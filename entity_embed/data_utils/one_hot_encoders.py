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


# unicode \w without _ is [\w--_]
tokenizer_re = regex.compile(r"[\w--_]+|[^[\w--_]\s]+", flags=regex.V1)


def _default_tokenizer_fn(val):
    return tokenizer_re.findall(val)


class AttrOneHotEncoder:
    def __init__(
        self,
        val_list,
        attr,
        is_multitoken,
        tokenizer_fn=None,
        alphabet=None,
        max_str_len=None,
    ):
        # TODO: optimize to single pass over val_list for computing None params
        self.attr = attr
        self.is_multitoken = is_multitoken
        if is_multitoken:
            if tokenizer_fn is not None:
                self.tokenizer_fn = tokenizer_fn
            else:
                self.tokenizer_fn = _default_tokenizer_fn

        if alphabet is not None:
            self.alphabet = alphabet
        else:
            # Compute alphabet from val_list
            self.alphabet = sorted(set(c for val in val_list for c in val))

        if max_str_len is not None:
            # Ensure max_str_len is pair to enable pooling later
            if max_str_len % 2 != 0:
                logger.warning(
                    f"Provided {max_str_len=} is not a pair number. Changing to {max_str_len + 1}"
                )
                max_str_len += 1
            self.max_str_len = max_str_len
        else:
            # Compute max_str_len from val_list
            if not self.is_multitoken:
                self.max_str_len = max(len(val) for val in val_list)
            else:
                self.max_str_len = max(len(v) for val in val_list for v in self.tokenizer_fn(val))
            # Ensure max_str_len is pair to enable pooling later
            if self.max_str_len % 2 != 0:
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

    def get_one_hot_encoding_info(self):
        return OneHotEncodingInfo(
            is_multitoken=self.is_multitoken,
            alphabet_len=len(self.alphabet),
            max_str_len=self.max_str_len,
        )


def build_attr_one_hot_encoders(
    row_list,
    attr_list,
    is_multitoken_attr_list=None,
    tokenizer_fn=None,
    alphabet=None,
    show_progress=False,
):
    is_multitoken_attr_set = set(is_multitoken_attr_list) if is_multitoken_attr_list else set()
    attr_to_encoder = {}

    for attr in tqdm.tqdm(attr_list, disable=not show_progress):
        encoder = AttrOneHotEncoder(
            val_list=[row[attr] for row in row_list],
            attr=attr,
            is_multitoken=attr in is_multitoken_attr_set,
            tokenizer_fn=tokenizer_fn if attr in is_multitoken_attr_set else None,
            alphabet=alphabet,
        )
        attr_to_encoder[attr] = encoder

    return attr_to_encoder
