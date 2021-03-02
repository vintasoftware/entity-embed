import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

import numpy as np
import regex
import torch
from torchtext.vocab import Vocab

logger = logging.getLogger(__name__)
AVAILABLE_VOCABS = [
    "charngram.100d",
    "fasttext.en.300d",
    "fasttext.simple.300d",
    "glove.42B.300d",
    "glove.840B.300d",
    "glove.twitter.27B.25d",
    "glove.twitter.27B.50d",
    "glove.twitter.27B.100d",
    "glove.twitter.27B.200d",
    "glove.6B.50d",
    "glove.6B.100d",
    "glove.6B.200d",
    "glove.6B.300d",
]


class FieldType(Enum):
    STRING = "string"
    MULTITOKEN = "multitoken"
    SEMANTIC_STRING = "semantic_string"
    SEMANTIC_MULTITOKEN = "semantic_multitoken"


@dataclass
class NumericalizeInfo:
    source_attr: str
    field_type: FieldType
    tokenizer: Callable[[str], List[str]]
    alphabet: List[str]
    max_str_len: int
    vocab: Vocab
    n_channels: int
    embed_dropout_p: float
    use_attention: bool
    use_mask: bool

    @property
    def is_multitoken(self):
        field_type = self.field_type
        if isinstance(field_type, str):
            field_type = FieldType[field_type]
        return field_type in (FieldType.MULTITOKEN, FieldType.SEMANTIC_MULTITOKEN)

    def __repr__(self):
        repr_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Callable):
                repr_dict[k] = f"{inspect.getmodule(v).__name__}.{v.__name__}"
            else:
                repr_dict[k] = v
        return "{klass}({attrs})".format(
            klass=self.__class__.__name__,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in repr_dict.items()),
        )


# Unicode \w without _ is [\w--_]
tokenizer_re = regex.compile(r"[\w--_]+|[^[\w--_]\s]+", flags=regex.V1)


def default_tokenizer(val):
    return tokenizer_re.findall(val)


class StringNumericalizer:
    is_multitoken = False

    def __init__(self, attr, numericalize_info):
        self.attr = attr
        self.alphabet = numericalize_info.alphabet
        self.max_str_len = numericalize_info.max_str_len
        self.char_to_ord = {c: i for i, c in enumerate(self.alphabet)}

    def _ord_encode(self, val):
        ord_encoded = []
        for c in val:
            try:
                ord_ = self.char_to_ord[c]
                ord_encoded.append(ord_)
            except KeyError:
                logger.warning(f"Found out of alphabet char at val={val}, char={c}")
        return ord_encoded

    def build_tensor(self, val):
        # encoded_arr is a one hot encoded bidimensional tensor
        # where the rows represent characters and the columns positions in the string.
        # This is the shape expected by StringEmbedCNN.
        ord_encoded_val = self._ord_encode(val)
        encoded_arr = np.zeros((len(self.alphabet), self.max_str_len), dtype=np.float32)
        if len(ord_encoded_val) > 0:
            encoded_arr[ord_encoded_val, range(len(ord_encoded_val))] = 1.0
        t = torch.from_numpy(encoded_arr)
        return t


class SemanticStringNumericalizer:
    is_multitoken = False

    def __init__(self, attr, numericalize_info):
        self.attr = attr
        self.vocab = numericalize_info.vocab

    def build_tensor(self, val):
        # encoded_arr is a lookup_tensor like in
        # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
        t = torch.tensor(self.vocab[val], dtype=torch.long)
        return t


class MultitokenNumericalizer:
    is_multitoken = True

    def __init__(self, attr, numericalize_info):
        self.attr = attr
        self.tokenizer = numericalize_info.tokenizer
        self.string_numericalizer = StringNumericalizer(
            attr=attr, numericalize_info=numericalize_info
        )

    def build_tensor(self, val):
        val_tokens = self.tokenizer(val)
        t_list = []
        for v in val_tokens:
            if v != "":
                t = self.string_numericalizer.build_tensor(v)
                t_list.append(t)

        if len(t_list) > 0:
            return torch.stack(t_list), len(t_list)
        else:
            t = self.string_numericalizer.build_tensor("")
            return torch.stack([t]), 0


class SemanticMultitokenNumericalizer(MultitokenNumericalizer):
    def __init__(self, attr, numericalize_info):
        self.attr = attr
        self.tokenizer = numericalize_info.tokenizer
        self.string_numericalizer = SemanticStringNumericalizer(
            attr=attr, numericalize_info=numericalize_info
        )


class RowNumericalizer:
    def __init__(
        self,
        attr_info_dict,
        attr_to_numericalizer,
    ):
        self.attr_info_dict = attr_info_dict
        self.attr_to_numericalizer = attr_to_numericalizer

    def build_tensor_dict(self, row):
        tensor_dict = {}
        sequence_length_dict = {}

        for attr, numericalizer in self.attr_to_numericalizer.items():
            # Get the source_attr from the NumericalizeInfo object for the
            # cases where the attr is different from the row's key
            source_attr = self.attr_info_dict[attr].source_attr
            if numericalizer.is_multitoken:
                t, sequence_length = numericalizer.build_tensor(row[source_attr])
            else:
                t = numericalizer.build_tensor(row[source_attr])
                sequence_length = None
            tensor_dict[attr] = t
            sequence_length_dict[attr] = sequence_length

        return tensor_dict, sequence_length_dict

    def __repr__(self):
        return f"<RowNumericalizer with attr_info_dict={self.attr_info_dict}>"
