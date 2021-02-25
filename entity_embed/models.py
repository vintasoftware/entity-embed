import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .data_utils.numericalizer import FieldType


class StringEmbedCNN(nn.Module):
    """
    PyTorch nn.Module for embedding strings for fast edit distance computation,
    based on "Convolutional Embedding for Edit Distance (SIGIR 20)"
    (code: https://github.com/xinyandai/string-embed)

    The tensor shape expected here is produced by StringNumericalizer.
    """

    def __init__(self, numericalize_info, embedding_size):
        super().__init__()

        self.alphabet_len = len(numericalize_info.alphabet)
        self.max_str_len = numericalize_info.max_str_len
        self.n_channels = numericalize_info.n_channels
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
        if numericalize_info.embed_dropout_p:
            dense_layers.append(nn.Dropout(p=numericalize_info.embed_dropout_p))
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x, **kwargs):
        x_len = len(x)
        x = x.view(x.size(0), 1, -1)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x_len, self.flat_size)
        x = self.dense_net(x)

        return x


class SemanticEmbedNet(nn.Module):
    def __init__(self, numericalize_info, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.dense_net = nn.Sequential(
            nn.Embedding.from_pretrained(numericalize_info.vocab.vectors),
            nn.Dropout(p=numericalize_info.embed_dropout_p),
        )

    def forward(self, x, **kwargs):
        return self.dense_net(x)


class Attention(nn.Module):
    """
    PyTorch nn.Module of an Attention mechanism for weighted averging of
    hidden states produced by a RNN. Based on mechanisms discussed in
    "Using millions of emoji occurrences to learn any-domain representations
    for detecting sentiment, emotion and sarcasm (EMNLP 17)"
    (code at https://github.com/huggingface/torchMoji)
    and
    "AutoBlock: A Hands-off Blocking Framework for Entity Matching (WSDM 20)"
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.attention_weights = torch.nn.Parameter(
            torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, x, **kwargs):
        scores = h.matmul(self.attention_weights)
        scores = self.softmax(scores)
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class MaskedAttention(nn.Module):
    """
    PyTorch nn.Module of an Attention mechanism for weighted averging of
    hidden states produced by a RNN. Based on mechanisms discussed in
    "Using millions of emoji occurrences to learn any-domain representations
    for detecting sentiment, emotion and sarcasm (EMNLP 17)"
    (code at https://github.com/huggingface/torchMoji)
    and
    "AutoBlock: A Hands-off Blocking Framework for Entity Matching (WSDM 20)".

    Different from the other Attention class, this one uses a mask
    to handle variable length inputs on the same batch.
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.attention_weights = nn.Parameter(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1))

    def forward(self, h, x, sequence_lengths, **kwargs):
        logits = h.matmul(self.attention_weights)
        scores = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = h.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < torch.LongTensor(sequence_lengths).unsqueeze(1)).float())
        if scores.data.is_cuda:
            mask = mask.cuda()

        # apply mask and renormalize attention scores (weights)
        masked_scores = scores * mask
        att_sums = masked_scores.sum(dim=1, keepdim=True)  # sums per sequence
        att_sums[att_sums == 0] = 1.0  # prevents division by zero on empty sequences
        scores = masked_scores.div(att_sums)

        # apply attention weights
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class MultitokenAttentionEmbed(nn.Module):
    def __init__(self, embedding_net, use_mask):
        super().__init__()

        self.embedding_net = embedding_net
        self.gru = nn.GRU(
            input_size=embedding_net.embedding_size,
            hidden_size=embedding_net.embedding_size // 2,  # due to bidirectional, must divide by 2
            bidirectional=True,
            batch_first=True,
        )
        if use_mask:
            self.attention_net = MaskedAttention(embedding_size=embedding_net.embedding_size)
        else:
            self.attention_net = Attention(embedding_size=embedding_net.embedding_size)

    def forward(self, x, sequence_lengths, **kwargs):
        x_tokens = x.unbind(dim=1)
        x_tokens = [self.embedding_net(x) for x in x_tokens]
        x = torch.stack(x_tokens, dim=1)

        # Pytorch can't handle zero length sequences,
        # but attention_net will use the actual sequence_lengths with zeros
        # https://github.com/pytorch/pytorch/issues/4582
        # https://github.com/pytorch/pytorch/issues/50192
        sequence_lengths_no_zero = [max(sl, 1) for sl in sequence_lengths]

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, sequence_lengths_no_zero, batch_first=True, enforce_sorted=False
        )
        packed_h, __ = self.gru(packed_x)
        h, __ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)
        return self.attention_net(h, x, sequence_lengths=sequence_lengths)


class MultitokenAverageEmbed(nn.Module):
    def __init__(self, embedding_net, use_mask):
        super().__init__()

        self.embedding_net = embedding_net
        self.use_mask = use_mask

    def forward(self, x, sequence_lengths, **kwargs):
        max_len = x.size(1)
        scores = torch.full((max_len,), 1 / max_len)
        if x.data.is_cuda:
            scores = scores.cuda()

        x_list = x.unbind(dim=1)
        x_list = [self.embedding_net(x) for x in x_list]
        x = torch.stack(x_list, dim=1)

        if self.use_mask:
            # Compute a mask for the attention on the padded sequences
            # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
            mask = Variable((idxes < torch.LongTensor(sequence_lengths).unsqueeze(1)).float())
            if x.data.is_cuda:
                mask = mask.cuda()

            # apply mask and renormalize
            masked_scores = scores * mask
            att_sums = masked_scores.sum(dim=1, keepdim=True)  # sums per sequence
            att_sums[att_sums == 0] = 1.0  # prevents division by zero on empty sequences
            scores = masked_scores.div(att_sums)

        # compute average
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class TupleSignature(nn.Module):
    def __init__(self, attr_info_dict):
        super().__init__()
        if len(attr_info_dict) > 1:
            self.weights = nn.Parameter(torch.full((len(attr_info_dict),), 1 / len(attr_info_dict)))
        else:
            self.weights = None

    def forward(self, attr_embedding_dict):
        if self.weights is not None:
            attr_embedding_list = list(attr_embedding_dict.values())
            x = torch.stack(attr_embedding_list, dim=1)
            x = F.normalize(x, dim=2)
            return F.normalize((x * self.weights.unsqueeze(-1).expand_as(x)).sum(axis=1), dim=1)
        else:
            return F.normalize(list(attr_embedding_dict.values())[0], dim=1)


class BlockerNet(nn.Module):
    def __init__(
        self,
        attr_info_dict,
        embedding_size=128,
    ):
        super().__init__()
        self.attr_info_dict = attr_info_dict
        self.embedding_size = embedding_size
        self.embedding_net_dict = nn.ModuleDict()

        for attr, numericalize_info in attr_info_dict.items():
            if numericalize_info.field_type in (
                FieldType.STRING,
                FieldType.MULTITOKEN,
            ):
                embedding_net = StringEmbedCNN(
                    numericalize_info=numericalize_info,
                    embedding_size=embedding_size,
                )
            elif numericalize_info.field_type in (
                FieldType.SEMANTIC_STRING,
                FieldType.SEMANTIC_MULTITOKEN,
            ):
                embedding_net = SemanticEmbedNet(
                    numericalize_info=numericalize_info,
                    embedding_size=embedding_size,
                )
            else:
                raise ValueError(
                    f"Unexpected numericalize_info.field_type={numericalize_info.field_type}"
                )

            if numericalize_info.field_type in (
                FieldType.MULTITOKEN,
                FieldType.SEMANTIC_MULTITOKEN,
            ):
                if numericalize_info.use_attention:
                    self.embedding_net_dict[attr] = MultitokenAttentionEmbed(
                        embedding_net, use_mask=numericalize_info.use_mask
                    )
                else:
                    self.embedding_net_dict[attr] = MultitokenAverageEmbed(
                        embedding_net, use_mask=numericalize_info.use_mask
                    )
            elif numericalize_info.field_type in (
                FieldType.STRING,
                FieldType.SEMANTIC_STRING,
            ):
                self.embedding_net_dict[attr] = embedding_net

        self.tuple_signature = TupleSignature(attr_info_dict)

    def forward(self, tensor_dict, sequence_length_dict):
        attr_embedding_dict = {}

        for attr, embedding_net in self.embedding_net_dict.items():
            attr_embedding = embedding_net(
                tensor_dict[attr], sequence_lengths=sequence_length_dict[attr]
            )
            attr_embedding_dict[attr] = attr_embedding

        return self.tuple_signature(attr_embedding_dict)

    def fix_signature_weights(self):
        """
        Force signature weights between 0 and 1 and total sum as 1.
        """
        if self.tuple_signature.weights is None:
            return

        with torch.no_grad():
            sd = self.tuple_signature.state_dict()
            weights = sd["weights"]
            one_tensor = torch.tensor([1.0]).to(weights.device)
            if torch.any((weights < 0) | (weights > 1)) or not torch.isclose(
                weights.sum(), one_tensor
            ):
                weights[weights < 0] = 0
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights /= weights.sum()
                else:
                    print("Warning: all weights turned to 0. Setting all equal.")
                    weights[[True] * len(weights)] = 1 / len(weights)
                sd["weights"] = weights
                self.tuple_signature.load_state_dict(sd)

    def get_signature_weights(self):
        with torch.no_grad():
            if self.tuple_signature.weights is None:
                return {list(self.attr_info_dict.keys())[0]: 1.0}

            return {
                attr: float(weight)
                for attr, weight in zip(
                    self.attr_info_dict.keys(),
                    self.tuple_signature.state_dict()["weights"],
                )
            }
