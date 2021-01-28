import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StringEmbedCNN(nn.Module):
    def __init__(self, alphabet_len, max_str_len, n_channels, embedding_size, embed_dropout_p):
        super().__init__()

        self.max_str_len = max_str_len
        self.n_channels = n_channels
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.flat_size = (max_str_len // 2) * alphabet_len * n_channels
        if self.flat_size == 0:
            raise ValueError("Too small alphabet, self.flat_size == 0")
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_size, self.embedding_size),
            nn.Dropout(p=embed_dropout_p),
        )

    def forward(self, x):
        x_len = len(x)
        x = x.reshape(-1, 1, self.max_str_len)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x_len, self.flat_size)
        x = self.fc1(x)

        return x


class AttentionModule(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.attention_weights = torch.nn.Parameter(
            torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, x, lengths):
        scores = h.matmul(self.attention_weights)
        scores = self.softmax(scores)
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class MaskedAttentionModule(nn.Module):
    """
    Class that implements a Self-Attention module that will be applied on the outputs of the GRU layer.
    Based on:
        https://github.com/huggingface/torchMoji/blob/198f7d4e0711a7d3cd01968812af0121c54477f8/torchmoji/attlayer.py
        https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983
        https://www.kaggle.com/andrelmfarias/bi-gru-with-self-attention-and-statistical-feat
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.attention_weights = nn.Parameter(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, x, lengths):
        scores = h.matmul(self.attention_weights)
        scores = self.softmax(scores)

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = h.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < torch.LongTensor(lengths).unsqueeze(1)).float())
        if scores.data.is_cuda:
            mask = mask.cuda()

        # apply mask and renormalize attention scores (weights)
        masked_scores = scores * mask
        att_sums = masked_scores.sum(dim=1, keepdim=True)  # sums per sequence
        scores = masked_scores.div(att_sums)

        # apply attention weights
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class MultitokenAttrAttentionModule(nn.Module):
    def __init__(self, embedding_size, use_mask):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size // 2,
            bidirectional=True,
            batch_first=True,
        )
        if use_mask:
            self.attention = MaskedAttentionModule(embedding_size=embedding_size)
        else:
            self.attention = AttentionModule(embedding_size=embedding_size)

    def forward(self, x, tensor_lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, tensor_lengths, batch_first=True, enforce_sorted=False
        )
        packed_h, __ = self.gru(packed_x)
        h, __ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)
        return self.attention(h, x, tensor_lengths)


class MultitokenAttentionEmbed(nn.Module):
    def __init__(self, embedding_net, use_mask):
        super().__init__()

        self.embedding_net = embedding_net
        self.attention_net = MultitokenAttrAttentionModule(
            embedding_size=embedding_net.embedding_size, use_mask=use_mask
        )

    def forward(self, x, tensor_lengths):
        x_list = x.unbind(dim=1)
        x_list = [self.embedding_net(x) for x in x_list]
        return self.attention_net(torch.stack(x_list, dim=1), tensor_lengths)


class MultitokenAverageEmbed(nn.Module):
    def __init__(self, embedding_net, use_mask):
        super().__init__()

        self.embedding_net = embedding_net
        self.use_mask = use_mask

    def forward(self, x, tensor_lengths):
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
            mask = Variable((idxes < torch.LongTensor(tensor_lengths).unsqueeze(1)).float())
            if x.data.is_cuda:
                mask = mask.cuda()

            # apply mask and renormalize
            masked_scores = scores * mask
            sums = masked_scores.sum(dim=1, keepdim=True)  # sums per sequence
            scores = masked_scores.div(sums)

        # compute average
        weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
        representations = weighted.sum(dim=1)

        return representations


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2, x3 = x
        return self.embedding_net(x1), self.embedding_net(x2), self.embedding_net(x3)


class TupleSignatureModule(nn.Module):
    def __init__(self, attr_list):
        super().__init__()
        self.weights = nn.Parameter(torch.full((len(attr_list),), 1 / len(attr_list)))

    def forward(self, attr_embedding_list):
        return (torch.stack(attr_embedding_list) * self.weights[:, None, None]).sum(axis=0)


def fix_signature_params(model):
    """
    Force signature params between 0 and 1 and total sum 1
    """
    with torch.no_grad():
        sd = model.tuple_signature.state_dict()
        weights = sd["weights"]
        one_tensor = torch.tensor([1.0]).to(weights.device)
        if torch.any((weights < 0) | (weights > 1)) or not torch.isclose(weights.sum(), one_tensor):
            weights[weights < 0] = 0
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights /= weights.sum()
            else:
                print("Warning: all weights turned to 0. Setting all equal.")
                weights[[True] * len(weights)] = 1 / len(weights)
            sd["weights"] = weights
            model.tuple_signature.load_state_dict(sd)


def get_current_signature_weights(attr_list, model):
    return list(zip(attr_list, model.tuple_signature.state_dict()["weights"]))


class BlockerNet(nn.Module):
    def __init__(
        self,
        attr_to_encoding_info,
        n_channels=8,
        embedding_size=128,
        embed_dropout_p=0.2,
        use_attention=True,
        use_mask=False,
    ):
        super().__init__()
        self.embedding_net_dict = nn.ModuleDict()
        for attr, one_hot_encoding_info in attr_to_encoding_info.items():
            embedding_net = StringEmbedCNN(
                alphabet_len=one_hot_encoding_info.alphabet_len,
                max_str_len=one_hot_encoding_info.max_str_len,
                n_channels=n_channels,
                embedding_size=embedding_size,
                embed_dropout_p=embed_dropout_p,
            )
            if not one_hot_encoding_info.is_multitoken:
                self.embedding_net_dict[attr] = embedding_net
            else:
                if use_attention:
                    self.embedding_net_dict[attr] = MultitokenAttentionEmbed(
                        embedding_net, use_mask
                    )
                else:
                    self.embedding_net_dict[attr] = MultitokenAverageEmbed(embedding_net, use_mask)
        attr_list = list(attr_to_encoding_info.keys())
        self.tuple_signature = TupleSignatureModule(attr_list)

    def forward(self, encoded_attr_tensor_list, tensor_lengths_list):
        embedding_net_list = self.embedding_net_dict.values()
        return F.normalize(
            self.tuple_signature(
                [
                    embedding_net(encoded_attr_tensor, tensor_lengths)
                    if isinstance(embedding_net, (MultitokenAttentionEmbed, MultitokenAverageEmbed))
                    else embedding_net(encoded_attr_tensor)
                    for embedding_net, encoded_attr_tensor, tensor_lengths in zip(
                        embedding_net_list, encoded_attr_tensor_list, tensor_lengths_list
                    )
                ]
            )
        )
