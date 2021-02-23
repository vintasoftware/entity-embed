import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Union

import numpy as np
import regex
import torch
from torchtext.vocab import Vocab

from .utils import compute_alphabet_and_max_str_len, compute_vocab_counter, import_function

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


DEFAULT_ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")


class FieldType(Enum):
    STRING = "string"
    MULTITOKEN = "multitoken"
    SEMANTIC_STRING = "semantic_string"
    SEMANTIC_MULTITOKEN = "semantic_multitoken"


@dataclass
class NumericalizeInfo:
    field_type: FieldType
    tokenizer: Optional[Union[str, Callable[[str], List[str]]]] = "entity_embed.default_tokenizer"
    alphabet: Optional[List[str]] = field(default_factory=lambda: DEFAULT_ALPHABET)
    max_str_len: Optional[int] = None
    vocab: Optional[Union[str, Vocab]] = None

    @property
    def is_multitoken(self):
        field_type = self.field_type
        if isinstance(field_type, str):
            field_type = FieldType[field_type]
        return field_type in (FieldType.MULTITOKEN, FieldType.SEMANTIC_MULTITOKEN)


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
        return [self.char_to_ord[c] for c in val]

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
        row_dict=None,
    ):
        self.attr_info_dict = {}
        self.attr_to_numericalizer = {}

        for attr, numericalize_info in attr_info_dict.items():
            field_type = FieldType[numericalize_info.field_type]
            tokenizer = (
                import_function(numericalize_info.tokenizer)
                if numericalize_info.tokenizer
                else None
            )
            alphabet = numericalize_info.alphabet
            max_str_len = numericalize_info.max_str_len
            vocab = None

            # Check if tokenizer function is set
            if (
                field_type in (FieldType.MULTITOKEN, FieldType.SEMANTIC_MULTITOKEN)
                and tokenizer is None
            ):
                raise ValueError(
                    f"{attr=} has {field_type=} but {self.tokenizer=}. Please set a tokenizer."
                )

            # Compute vocab if necessary
            if field_type in (FieldType.SEMANTIC_STRING, FieldType.SEMANTIC_MULTITOKEN):
                if numericalize_info.vocab is None:
                    raise ValueError(
                        "Please set a torchtext pretrained vocab to use. "
                        f"Available ones are: {AVAILABLE_VOCABS}"
                    )
                vocab_counter = compute_vocab_counter(
                    attr_val_gen=(row[attr] for row in row_dict.values()),
                    tokenizer=tokenizer,
                )
                vocab = Vocab(vocab_counter)
                vocab.load_vectors(numericalize_info.vocab)

            # Compute alphabet and max_str_len if necessary
            if field_type in (FieldType.STRING, FieldType.MULTITOKEN) and (
                alphabet is None or max_str_len is None
            ):
                if row_dict is None:
                    raise ValueError(
                        f"Cannot compute alphabet and max_str_len for {attr=}. "
                        "row_dict cannot be None if alphabet or max_str_len is None. "
                        "Please set row_dict, a dictionary of id -> row with ALL your data "
                        "(train, test, valid). "
                        "Or call entity_embed.data_utils.compute_alphabet_and_max_str_len "
                        "over ALL your data (train, test, valid) to compute alphabet and "
                        "max_str_len."
                    )
                else:
                    logger.info(f"For {attr=}, computing actual alphabet and max_str_len")
                    (actual_alphabet, actual_max_str_len,) = compute_alphabet_and_max_str_len(
                        attr_val_gen=(row[attr] for row in row_dict.values()),
                        is_multitoken=numericalize_info.is_multitoken,
                        tokenizer=tokenizer,
                    )
                    if alphabet is None:
                        logger.info(f"For {attr=}, using {actual_alphabet=}")
                        alphabet = actual_alphabet
                    if max_str_len is None:
                        logger.info(f"For {attr=}, using {actual_max_str_len=}")
                        max_str_len = actual_max_str_len

            field_type_to_numericalizer_cls = {
                FieldType.STRING: StringNumericalizer,
                FieldType.MULTITOKEN: MultitokenNumericalizer,
                FieldType.SEMANTIC_STRING: SemanticStringNumericalizer,
                FieldType.SEMANTIC_MULTITOKEN: SemanticMultitokenNumericalizer,
            }
            numericalizer_cls = field_type_to_numericalizer_cls.get(field_type)
            if numericalizer_cls is None:
                raise ValueError(f"Unexpected {field_type=}")

            self.attr_info_dict[attr] = NumericalizeInfo(
                field_type=field_type,
                tokenizer=tokenizer,
                alphabet=alphabet,
                max_str_len=max_str_len,
                vocab=vocab,
            )
            self.attr_to_numericalizer[attr] = numericalizer_cls(
                attr=attr, numericalize_info=self.attr_info_dict[attr]
            )

    def build_tensor_dict(self, row):
        tensor_dict = {}
        sequence_length_dict = {}

        for attr, numericalizer in self.attr_to_numericalizer.items():
            if numericalizer.is_multitoken:
                t, sequence_length = numericalizer.build_tensor(row[attr])
            else:
                t = numericalizer.build_tensor(row[attr])
                sequence_length = None
            tensor_dict[attr] = t
            sequence_length_dict[attr] = sequence_length

        return tensor_dict, sequence_length_dict
