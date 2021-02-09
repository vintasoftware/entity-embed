import logging
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import regex
import torch

from .utils import compute_alphabet_and_max_str_len

logger = logging.getLogger(__name__)


@dataclass
class OneHotEncodingInfo:
    is_multitoken: bool
    tokenizer: Callable[[str], List[str]]
    alphabet: List[str]
    max_str_len: int


# Unicode \w without _ is [\w--_]
tokenizer_re = regex.compile(r"[\w--_]+|[^[\w--_]\s]+", flags=regex.V1)


def default_tokenizer(val):
    return tokenizer_re.findall(val)


class AttrOneHotEncoder:
    def __init__(self, attr, one_hot_encoding_info):
        self.attr = attr
        self.is_multitoken = one_hot_encoding_info.is_multitoken
        self.tokenizer = one_hot_encoding_info.tokenizer
        if self.is_multitoken and self.tokenizer is None:
            raise ValueError(
                f"{attr=} has {self.is_multitoken=} but {self.tokenizer=}. "
                "Please set a tokenizer."
            )

        self.alphabet = one_hot_encoding_info.alphabet
        self.max_str_len = one_hot_encoding_info.max_str_len

        self.char_to_ord = {c: i for i, c in enumerate(self.alphabet)}

    def _one_hot_encode(self, val):
        return [self.char_to_ord[c] for c in val]

    def _build_single_tensor(self, val):
        # encoded_arr is a one hot encoded bidimensional tensor
        # where the rows represent characters and the columns positions in the string.
        ord_encoded_val = self._one_hot_encode(val)
        encoded_arr = np.zeros((len(self.alphabet), self.max_str_len), dtype=np.float32)
        if len(ord_encoded_val) > 0:
            encoded_arr[ord_encoded_val, range(len(ord_encoded_val))] = 1.0
        return torch.from_numpy(encoded_arr)

    def build_tensor(self, val):
        if not self.is_multitoken:
            return self._build_single_tensor(val)
        else:
            val_tokens = self.tokenizer(val)
            if val_tokens:
                token_t_list = [self._build_single_tensor(v) for v in val_tokens]
            else:
                token_t_list = []

            if len(token_t_list) > 0:
                return torch.stack(token_t_list)
            else:
                return torch.stack([self._build_single_tensor("")])


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
                        f"Cannot compute alphabet and max_str_len for {attr=}. "
                        "row_dict cannot be None if any of alphabet and max_str_len is None. "
                        "Please set row_dict, a dictionary of id -> row with ALL your data (train, test, valid). "
                        "Or call entity_embed.data_utils.compute_alphabet_and_max_str_len "
                        "over ALL your data (train, test, valid) to compute alphabet and max_str_len for each attr. "
                    )
                else:
                    logger.info(f"For {attr=}, computing actual alphabet and max_str_len")
                    actual_alphabet, actual_max_str_len = compute_alphabet_and_max_str_len(
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

            self.attr_info_dict[attr] = OneHotEncodingInfo(
                is_multitoken=one_hot_encoding_info.is_multitoken,
                tokenizer=one_hot_encoding_info.tokenizer,
                alphabet=alphabet,
                max_str_len=max_str_len,
            )
            self.attr_to_encoder[attr] = AttrOneHotEncoder(
                attr=attr, one_hot_encoding_info=self.attr_info_dict[attr]
            )

    def build_tensor_dict(self, row, log_empty_vals=False):
        tensor_dict = {}

        for attr, encoder in self.attr_to_encoder.items():
            if not row[attr] and log_empty_vals:
                logger.warning(f"Found empty {attr=} at row={row}")
            tensor_dict[attr] = encoder.build_tensor(row[attr])
        return tensor_dict
