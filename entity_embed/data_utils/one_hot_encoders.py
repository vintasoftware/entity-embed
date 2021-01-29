import logging
from dataclasses import dataclass

import numpy as np
import regex
import torch
import tqdm

logger = logging.getLogger(__name__)


@dataclass
class OneHotEncodingInfo:
    is_multitoken: bool
    alphabet_len: int
    max_str_len: int


# Unicode \w without _ is [\w--_]
tokenizer_re = regex.compile(r"[\w--_]+|[^[\w--_]\s]+", flags=regex.V1)


def _default_tokenizer_fn(val):
    return tokenizer_re.findall(val)


class AttrOneHotEncoder:
    def __init__(
        self,
        val_gen,
        attr,
        is_multitoken,
        tokenizer_fn=None,
        alphabet=None,
        max_str_len=None,
    ):
        self.attr = attr
        self.is_multitoken = is_multitoken
        if is_multitoken:
            if tokenizer_fn is not None:
                self.tokenizer_fn = tokenizer_fn
            else:
                self.tokenizer_fn = _default_tokenizer_fn

        if alphabet is not None:
            self.alphabet = alphabet

        if max_str_len is not None:
            self.max_str_len = max_str_len

        # Compute alphabet or max_str_len from val_gen if is None
        if alphabet is None or max_str_len is None:
            actual_alphabet = set()
            actual_max_str_len = 0
            for val in val_gen:
                actual_alphabet.update(list(val))
                if not self.is_multitoken:
                    str_len = len(val)
                else:
                    token_lens = [len(v) for v in self.tokenizer_fn(val)]
                    str_len = max(token_lens) if token_lens else -1
                actual_max_str_len = max(str_len, actual_max_str_len)

            if alphabet is None:
                self.alphabet = sorted(actual_alphabet)
            if max_str_len is None:
                self.max_str_len = actual_max_str_len

        # Ensure max_str_len is pair to enable pooling later
        if self.max_str_len % 2 != 0:
            logger.warning(
                f"{self.max_str_len=} must be pair to enable NN pooling. "
                f"Updating to {self.max_str_len + 1}"
            )
            self.max_str_len += 1

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
            val_tokens = self.tokenizer_fn(val)
            if val_tokens:
                token_t_list = [self._build_single_tensor(v) for v in val_tokens]
            else:
                token_t_list = []

            if len(token_t_list) > 0:
                return torch.stack(token_t_list)
            else:
                return torch.stack([self._build_single_tensor("")])

    @property
    def one_hot_encoding_info(self):
        return OneHotEncodingInfo(
            is_multitoken=self.is_multitoken,
            alphabet_len=len(self.alphabet),
            max_str_len=self.max_str_len,
        )


class RowOneHotEncoder:
    def __init__(
        self,
        row_dict,
        attr_list,
        is_multitoken_attr_list=None,
        tokenizer_fn=None,
        alphabet=None,
        show_progress=False,
    ):
        is_multitoken_attr_set = set(is_multitoken_attr_list) if is_multitoken_attr_list else set()
        self.attr_to_encoder = {}

        for attr in tqdm.tqdm(attr_list, disable=not show_progress):
            encoder = AttrOneHotEncoder(
                val_gen=(row[attr] for row in row_dict.values()),
                attr=attr,
                is_multitoken=attr in is_multitoken_attr_set,
                tokenizer_fn=tokenizer_fn if attr in is_multitoken_attr_set else None,
                alphabet=alphabet,
            )
            self.attr_to_encoder[attr] = encoder

    def build_tensor_dict(self, row, log_empty_vals=False):
        tensor_dict = {}

        for attr, encoder in self.attr_to_encoder.items():
            if not row[attr] and log_empty_vals:
                logger.warning(f"Found empty {attr=} at row={row}")
            tensor_dict[attr] = encoder.build_tensor(row[attr])
        return tensor_dict

    @property
    def attr_info_dict(self):
        return {
            attr: encoder.one_hot_encoding_info for attr, encoder in self.attr_to_encoder.items()
        }
