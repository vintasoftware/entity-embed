import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import regex
import torch
import torchtext
from torchtext.vocab import Vocab

from .utils import Enumerator, compute_alphabet_and_max_str_len, compute_vocab

logger = logging.getLogger(__name__)


@dataclass
class OneHotEncodingInfo:
    is_multitoken: bool
    is_semantic: bool
    tokenizer: Callable[[str], List[str]]
    alphabet: List[str]
    max_str_len: int
    vocab: Optional[Vocab]


# Unicode \w without _ is [\w--_]
tokenizer_re = regex.compile(r"[\w--_]+|[^[\w--_]\s]+", flags=regex.V1)


def default_tokenizer(val):
    return tokenizer_re.findall(val)


class AttrOneHotEncoder:
    def __init__(self, attr, one_hot_encoding_info, vocab):
        self.attr = attr
        self.is_multitoken = one_hot_encoding_info.is_multitoken
        self.is_semantic = one_hot_encoding_info.is_semantic
        self.tokenizer = one_hot_encoding_info.tokenizer
        if self.is_multitoken and self.tokenizer is None:
            raise ValueError(
                f"{attr=} has {self.is_multitoken=} but {self.tokenizer=}. "
                "Please set a tokenizer."
            )

        self.alphabet = one_hot_encoding_info.alphabet
        self.max_str_len = one_hot_encoding_info.max_str_len

        self.vocab = vocab
        self.char_to_ord = {c: i for i, c in enumerate(self.alphabet)}

    def _one_hot_encode(self, val):
        return [self.char_to_ord[c] for c in val]

    def _build_single_tensor(self, val):
        if self.is_semantic:
            # encoded_arr is a lookup_tensor like in
            # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
            return torch.tensor(self.vocab[val], dtype=torch.long)
        else:
            # encoded_arr is a one hot encoded bidimensional tensor
            # where the rows represent characters and the columns positions in the string.
            ord_encoded_val = self._one_hot_encode(val)
            encoded_arr = np.zeros((len(self.alphabet), self.max_str_len), dtype=np.float32)
            if len(ord_encoded_val) > 0:
                encoded_arr[ord_encoded_val, range(len(ord_encoded_val))] = 1.0
            return torch.from_numpy(encoded_arr)

    def build_tensor(self, val):
        if not self.is_multitoken:
            t = self._build_single_tensor(val)
            token_length = 1 if val != "" else 0
            return t, token_length
        else:
            val_tokens = self.tokenizer(val)
            if val_tokens:
                token_t_list = [self._build_single_tensor(v) for v in val_tokens]
            else:
                token_t_list = []

            if len(token_t_list) > 0:
                return torch.stack(token_t_list), len(token_t_list)
            else:
                return torch.stack([self._build_single_tensor("")]), len(token_t_list)


class RowOneHotEncoder:
    def __init__(
        self,
        attr_info_dict,
        row_dict=None,
    ):
        self.attr_info_dict = {}
        self.attr_to_encoder = {}

        for attr, one_hot_encoding_info in attr_info_dict.items():
            alphabet = one_hot_encoding_info.alphabet
            max_str_len = one_hot_encoding_info.max_str_len

            if alphabet is None or max_str_len is None:
                if row_dict is None:
                    raise ValueError(
                        f"Cannot compute alphabet / max_str_len for {attr=}. "
                        "row_dict cannot be None if any of alphabet / max_str_len is None. "
                        "Please set row_dict, a dictionary of id -> row with ALL your data (train, test, valid). "
                        "Or call entity_embed.data_utils.compute_alphabet_and_max_str_len "
                        "over ALL your data (train, test, valid) to compute alphabet and max_str_len for each attr. "
                    )
                else:
                    logger.info(f"For {attr=}, computing actual alphabet and max_str_len")
                    (actual_alphabet, actual_max_str_len,) = compute_alphabet_and_max_str_len(
                        attr_val_gen=(row[attr] for row in row_dict.values()),
                        is_multitoken=one_hot_encoding_info.is_multitoken,
                        tokenizer=one_hot_encoding_info.tokenizer,
                    )
                    if alphabet is None:
                        logger.info(f"For {attr=}, using {actual_alphabet=}")
                        alphabet = actual_alphabet
                    if max_str_len is None:
                        logger.info(f"For {attr=}, using {actual_max_str_len=}")
                        max_str_len = actual_max_str_len

            if one_hot_encoding_info.is_semantic:
                vocab_counter = compute_vocab(
                    attr_val_gen=(row[attr] for row in row_dict.values()),
                    tokenizer=one_hot_encoding_info.tokenizer,
                )
                vocab = torchtext.vocab.Vocab(vocab_counter)
                vocab.load_vectors("fasttext.en.300d")  # TODO: parametrize this
            else:
                vocab = None

            self.attr_info_dict[attr] = OneHotEncodingInfo(
                is_multitoken=one_hot_encoding_info.is_multitoken,
                is_semantic=one_hot_encoding_info.is_semantic,
                tokenizer=one_hot_encoding_info.tokenizer,
                alphabet=alphabet,
                max_str_len=max_str_len,
                vocab=vocab,
            )
            self.attr_to_encoder[attr] = AttrOneHotEncoder(
                attr=attr, one_hot_encoding_info=self.attr_info_dict[attr], vocab=vocab
            )

    def build_tensor_dict(self, row, log_empty_vals=False):
        tensor_dict = {}
        tensor_lengths_dict = {}

        for attr, encoder in self.attr_to_encoder.items():
            if not row[attr] and log_empty_vals:
                logger.warning(f"Found empty {attr=} at row={row}")
            t, t_len = encoder.build_tensor(row[attr])
            tensor_dict[attr] = t
            tensor_lengths_dict[attr] = t_len

        return tensor_dict, tensor_lengths_dict

    def build_attr_subset_encoder(self, attr_subset):
        new_row_encoder = RowOneHotEncoder(attr_info_dict={})
        new_row_encoder.attr_info_dict = {
            attr: attr_info
            for attr, attr_info in self.attr_info_dict.items()
            if attr in attr_subset
        }
        new_row_encoder.attr_to_encoder = {
            attr: encoder for attr, encoder in self.attr_to_encoder.items() if attr in attr_subset
        }
        return new_row_encoder
