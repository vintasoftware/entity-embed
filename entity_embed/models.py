import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils.numericalizer import FieldType


class StringEmbedCNN(nn.Module):
    """
    PyTorch nn.Module for embedding strings for fast edit distance computation,
    based on "Convolutional Embedding for Edit Distance (SIGIR 20)"
    (code: https://github.com/xinyandai/string-embed)

    The tensor shape expected here is produced by StringNumericalizer.
    """

    def __init__(self, field_config, embedding_size):
        super().__init__()

        self.alphabet_len = len(field_config.alphabet)
        self.max_str_len = field_config.max_str_len
        self.n_channels = field_config.n_channels
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.flat_size = (self.max_str_len // 2) * self.alphabet_len * self.n_channels
        if self.flat_size == 0:
            raise ValueError("Too small alphabet, self.flat_size == 0")

        dense_layers = [nn.Linear(self.flat_size, self.embedding_size)]
        if field_config.embed_dropout_p:
            dense_layers.append(nn.Dropout(p=field_config.embed_dropout_p))
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x, **kwargs):
        x = x.view(x.size(0), 1, -1)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x.size(0), self.flat_size)
        x = self.dense_net(x)

        return x


class SemanticEmbedNet(nn.Module):
    def __init__(self, field_config, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.dense_net = nn.Sequential(
            nn.Embedding.from_pretrained(field_config.vocab.vectors),
            nn.Dropout(p=field_config.embed_dropout_p),
        )

    def forward(self, x, **kwargs):
        return self.dense_net(x)


class MaskedAttention(nn.Module):
    """
    PyTorch nn.Module of an Attention mechanism for weighted averging of
    hidden states produced by a RNN. Based on mechanisms discussed in
    "Using millions of emoji occurrences to learn any-domain representations
    for detecting sentiment, emotion and sarcasm (EMNLP 17)"
    (code at https://github.com/huggingface/torchMoji)
    and
    "AutoBlock: A Hands-off Blocking Framework for Entity Matching (WSDM 20)".
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.attention_weights = nn.Parameter(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1))

    def forward(self, h, x, sequence_lengths, **kwargs):
        logits = h.matmul(self.attention_weights)
        scores = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_sequence_len = h.size(1)
        idxes = torch.arange(0, max_sequence_len, dtype=torch.int64, device=x.device).unsqueeze(0)
        mask = (idxes < sequence_lengths.unsqueeze(1)).float()

        # apply mask and renormalize attention scores (weights)
        masked_scores = scores * mask
        att_sums = masked_scores.sum(dim=1, keepdim=True)  # sums per sequence
        att_sums = att_sums.clamp(min=1e-5)  # prevents division by zero on empty sequences
        scores = masked_scores.div(att_sums)

        # apply attention weights
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations, scores


class MultitokenAttentionEmbed(nn.Module):
    def __init__(self, embed_net):
        super().__init__()

        self.embed_net = embed_net
        self.gru = nn.GRU(
            input_size=embed_net.embedding_size,
            hidden_size=embed_net.embedding_size // 2,  # due to bidirectional, must divide by 2
            bidirectional=True,
            batch_first=True,
        )
        self.attention_net = MaskedAttention(embedding_size=embed_net.embedding_size)

    def _forward(self, x, sequence_lengths):
        x_tokens = x.unbind(dim=1)
        x_tokens = [self.embed_net(x) for x in x_tokens]
        x = torch.stack(x_tokens, dim=1)

        # Pytorch can't handle zero length sequences,
        # but attention_net will use the actual sequence_lengths with zeros
        # https://github.com/pytorch/pytorch/issues/4582
        # https://github.com/pytorch/pytorch/issues/50192
        sequence_lengths_no_zero = sequence_lengths.clamp(min=1)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, sequence_lengths_no_zero.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h, __ = self.gru(packed_x)
        h, __ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)
        return self.attention_net(h, x, sequence_lengths=sequence_lengths)

    def forward(self, x, sequence_lengths, **kwargs):
        embeddings, __ = self._forward(x, sequence_lengths)
        return embeddings


class MultitokenAvgEmbed(nn.Module):
    def __init__(self, embed_net):
        super().__init__()

        self.embed_net = embed_net

    def forward(self, x, sequence_lengths, **kwargs):
        x_list = x.unbind(dim=1)
        x_list = [self.embed_net(x) for x in x_list]
        x = torch.stack(x_list, dim=1)

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_sequence_len = x.size(1)
        scores = torch.full(
            (max_sequence_len,), 1 / max_sequence_len, dtype=torch.float32, device=x.device
        )
        idxes = torch.arange(0, max_sequence_len, dtype=torch.int64, device=x.device).unsqueeze(0)
        mask = (idxes < sequence_lengths.unsqueeze(1)).float()

        # apply mask and renormalize
        masked_scores = scores * mask
        att_sums = masked_scores.sum(dim=1, keepdim=True)  # sums per sequence
        att_sums = att_sums.clamp(min=1e-5)  # prevents division by zero on empty sequences
        scores = masked_scores.div(att_sums)

        # compute average
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class EntityAvgPoolNet(nn.Module):
    def __init__(self, field_config_dict, embedding_size):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_size)

        if len(field_config_dict) > 1:
            self.weights = nn.Parameter(
                torch.full((len(field_config_dict),), 1 / len(field_config_dict))
            )
        else:
            self.weights = None

    def forward(self, field_embedding_dict, sequence_length_dict):
        if self.weights is not None:
            # layer norm
            x = torch.stack(list(field_embedding_dict.values()), dim=1)
            x = self.norm(x)

            return F.normalize((x * self.weights.unsqueeze(-1).expand_as(x)).sum(axis=1), dim=1)
        else:
            return F.normalize(list(field_embedding_dict.values())[0], dim=1)


class FieldsEmbedNet(nn.Module):
    def __init__(
        self,
        field_config_dict,
        embedding_size,
    ):
        super().__init__()
        self.field_config_dict = field_config_dict
        self.embedding_size = embedding_size
        self.embed_net_dict = nn.ModuleDict()

        for field, field_config in field_config_dict.items():
            if field_config.field_type in (
                FieldType.STRING,
                FieldType.MULTITOKEN,
            ):
                embed_net = StringEmbedCNN(
                    field_config=field_config,
                    embedding_size=embedding_size,
                )
            elif field_config.field_type in (
                FieldType.SEMANTIC_STRING,
                FieldType.SEMANTIC_MULTITOKEN,
            ):
                embed_net = SemanticEmbedNet(
                    field_config=field_config,
                    embedding_size=embedding_size,
                )
            else:
                raise ValueError(f"Unexpected field_config.field_type={field_config.field_type}")

            if field_config.field_type in (
                FieldType.MULTITOKEN,
                FieldType.SEMANTIC_MULTITOKEN,
            ):
                if field_config.use_attention:
                    self.embed_net_dict[field] = MultitokenAttentionEmbed(embed_net)
                else:
                    self.embed_net_dict[field] = MultitokenAvgEmbed(embed_net)
            elif field_config.field_type in (
                FieldType.STRING,
                FieldType.SEMANTIC_STRING,
            ):
                self.embed_net_dict[field] = embed_net

    def forward(self, tensor_dict, sequence_length_dict):
        field_embeddings = []

        for field, embed_net in self.embed_net_dict.items():
            embedding = embed_net(tensor_dict[field], sequence_lengths=sequence_length_dict[field])
            field_embeddings.append(embedding)

        # zero empty strings and sequences
        field_embeddings = torch.stack(field_embeddings, dim=1)
        field_mask = torch.stack(list(sequence_length_dict.values()), dim=1).clamp(max=1)
        field_embeddings = field_embeddings * field_mask.unsqueeze(dim=-1)

        field_embedding_dict = dict(zip(self.embed_net_dict.keys(), field_embeddings.unbind(dim=1)))
        return field_embedding_dict


class BlockerNet(nn.Module):
    def __init__(
        self,
        field_config_dict,
        embedding_size=300,
    ):
        super().__init__()
        self.field_config_dict = field_config_dict
        self.embedding_size = embedding_size
        self.field_embed_net = FieldsEmbedNet(
            field_config_dict=field_config_dict, embedding_size=embedding_size
        )
        self.avg_pool_net = EntityAvgPoolNet(
            field_config_dict=field_config_dict, embedding_size=embedding_size
        )

    def forward(self, tensor_dict, sequence_length_dict, return_field_embeddings=False):
        field_embedding_dict = self.field_embed_net(
            tensor_dict=tensor_dict, sequence_length_dict=sequence_length_dict
        )
        avg_embedding = self.avg_pool_net(
            field_embedding_dict=field_embedding_dict, sequence_length_dict=sequence_length_dict
        )
        if return_field_embeddings:
            return field_embedding_dict, avg_embedding
        else:
            return avg_embedding

    def fix_pool_weights(self):
        """
        Force pool weights between 0 and 1 and total sum as 1.
        """
        if self.avg_pool_net.weights is None:
            return

        with torch.no_grad():
            sd = self.avg_pool_net.state_dict()
            weights = sd["weights"]
            weights = weights.clamp(min=1e-5, max=1.0)
            weights = weights / weights.sum()
            sd["weights"] = weights
            self.avg_pool_net.load_state_dict(sd)

    def get_pool_weights(self):
        with torch.no_grad():
            if self.avg_pool_net.weights is None:
                return {list(self.field_config_dict.keys())[0]: 1.0}

            return {
                field: float(weight)
                for field, weight in zip(
                    self.field_config_dict.keys(),
                    self.avg_pool_net.state_dict()["weights"],
                )
            }
