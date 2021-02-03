import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StringEmbedCNN(nn.Module):
    """
    PyTorch nn.Module for embedding strings for fast edit distance computation,
    based on "Convolutional Embedding for Edit Distance (SIGIR 20)"
    (code: https://github.com/xinyandai/string-embed)
    """

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

        dense_layers = [nn.Linear(self.flat_size, self.embedding_size)]
        if embed_dropout_p:
            dense_layers.append(nn.Dropout(p=embed_dropout_p))
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x, **kwargs):
        x_len = len(x)
        x = x.view(x.size(0), 1, -1)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x_len, self.flat_size)
        x = self.dense_net(x)

        return x


class Attention(nn.Module):
    """
    PyTorch nn.Module of an Attention mechanism for weighted averging of
    hidden states produced by a RNN. Based on mechanisms discussed in
    "Using millions of emoji occurrences to learn any-domain representations
    for detecting sentiment, emotion and sarcasm (EMNLP 17)" (code https://github.com/huggingface/torchMoji)
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
    for detecting sentiment, emotion and sarcasm (EMNLP 17)" (code https://github.com/huggingface/torchMoji)
    and
    "AutoBlock: A Hands-off Blocking Framework for Entity Matching (WSDM 20)".

    Different from the other Attention class, this one uses a mask
    to handle variable length inputs on the same batch.
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.attention_weights = nn.Parameter(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, x, tensor_lengths, **kwargs):
        scores = h.matmul(self.attention_weights)
        scores = self.softmax(scores)

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = h.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < torch.LongTensor(tensor_lengths).unsqueeze(1)).float())
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

    def forward(self, x, tensor_lengths, **kwargs):
        x_tokens = x.unbind(dim=1)
        x_tokens = [self.embedding_net(x) for x in x_tokens]
        x = torch.stack(x_tokens, dim=1)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, tensor_lengths, batch_first=True, enforce_sorted=False
        )
        packed_h, __ = self.gru(packed_x)
        h, __ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True)
        return self.attention_net(h, x, tensor_lengths=tensor_lengths)


class MultitokenAverageEmbed(nn.Module):
    def __init__(self, embedding_net, use_mask):
        super().__init__()

        self.embedding_net = embedding_net
        self.use_mask = use_mask

    def forward(self, x, tensor_lengths, **kwargs):
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


class TupleSignature(nn.Module):
    def __init__(self, attr_info_dict):
        super().__init__()
        self.weights = nn.Parameter(torch.full((len(attr_info_dict),), 1 / len(attr_info_dict)))

    def forward(self, attr_embedding_dict):
        attr_embedding_list = list(attr_embedding_dict.values())
        return (torch.stack(attr_embedding_list) * self.weights[:, None, None]).sum(axis=0)


class BlockerNet(nn.Module):
    def __init__(
        self,
        attr_info_dict,
        n_channels=8,
        embedding_size=128,
        embed_dropout_p=0.2,
        use_attention=True,
        use_mask=False,
    ):
        super().__init__()
        self.attr_info_dict = attr_info_dict
        self.embedding_size = embedding_size
        self.embedding_net_dict = nn.ModuleDict()

        for attr, one_hot_encoding_info in attr_info_dict.items():
            embedding_net = StringEmbedCNN(
                alphabet_len=len(one_hot_encoding_info.alphabet),
                max_str_len=one_hot_encoding_info.max_str_len,
                n_channels=n_channels,
                embedding_size=embedding_size,
                embed_dropout_p=embed_dropout_p,
            )
            if not one_hot_encoding_info.is_multitoken:
                self.embedding_net_dict[attr] = embedding_net
            else:
                embed_cls = MultitokenAttentionEmbed if use_attention else MultitokenAverageEmbed
                self.embedding_net_dict[attr] = embed_cls(embedding_net, use_mask)

        self.tuple_signature = TupleSignature(attr_info_dict)

    def forward(self, tensor_dict, tensor_lengths_dict):
        attr_embedding_dict = {}

        for attr, embedding_net in self.embedding_net_dict.items():
            attr_embedding = embedding_net(
                tensor_dict[attr], tensor_lengths=tensor_lengths_dict[attr]
            )
            attr_embedding_dict[attr] = attr_embedding

        return F.normalize(self.tuple_signature(attr_embedding_dict))
