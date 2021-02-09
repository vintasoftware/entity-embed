import logging
import math
import random

import more_itertools
import torch
import torch.nn as nn
from Levenshtein import ratio
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from . import utils

logger = logging.getLogger(__name__)


class PairDataset(Dataset):
    def __init__(
        self,
        row_dict,
        cluster_attr,
        row_encoder,
        pos_pair_batch_size,
        neg_pair_batch_size,
        random_seed=42,
        log_empty_vals=False,
    ):
        self.row_dict = row_dict
        self.pair_list = list(utils.row_dict_to_id_pairs(row_dict, cluster_attr))
        self.id_to_cluster_id = {id_: row[cluster_attr] for id_, row in row_dict.items()}
        self.row_encoder = row_encoder
        self.random = random.Random(random_seed)
        self.log_empty_vals = log_empty_vals

        self.neg_batch_id_size = utils.pair_count_to_row_count(neg_pair_batch_size)
        actual_neg_pair_batch_size = (self.neg_batch_id_size * (self.neg_batch_id_size - 1)) // 2
        if actual_neg_pair_batch_size != neg_pair_batch_size:
            logger.warning(
                "Since PairDataset samples negative pairs by rows, "
                "neg_pair_batch_size must fit the equation y = (n * (n - 1)) / 2, "
                "where y is neg_pair_batch_size and n is number of rows. "
                f"Therefore, requested {neg_pair_batch_size=} is impossible. "
                f"Closest lower pair number is {actual_neg_pair_batch_size}. Using it."
            )

        pos_pair_list_batches_gen = more_itertools.chunked(self.pair_list, pos_pair_batch_size)
        self.pos_id_list_batches = [
            sorted(set(id_ for pair in pair_batch for id_ in pair))
            for pair_batch in pos_pair_list_batches_gen
        ]

    def __getitem__(self, idx):
        pos_id_batch = self.pos_id_list_batches[idx]
        neg_id_batch = self.random.sample(self.id_to_cluster_id.keys(), self.neg_batch_id_size)
        id_batch = pos_id_batch + neg_id_batch

        tensor_dict = {attr: [] for attr in self.row_encoder.attr_info_dict.keys()}
        tensor_lengths_dict = {attr: [] for attr in self.row_encoder.attr_info_dict.keys()}
        for id_ in id_batch:
            row_tensor_dict, row_tensor_lengths_dict = self.row_encoder.build_tensor_dict(
                self.row_dict[id_], log_empty_vals=self.log_empty_vals
            )
            if all(l == 0 for l in row_tensor_lengths_dict.values()):  # ignore empty rows
                continue
            for attr in self.row_encoder.attr_info_dict.keys():
                tensor_dict[attr].append(row_tensor_dict[attr])
                tensor_lengths_dict[attr].append(row_tensor_lengths_dict[attr])

        for attr, one_hot_encoding_info in self.row_encoder.attr_info_dict.items():
            if one_hot_encoding_info.is_multitoken:
                tensor_dict[attr] = nn.utils.rnn.pad_sequence(tensor_dict[attr], batch_first=True)
            else:
                tensor_dict[attr] = default_collate(tensor_dict[attr])

        label_batch = default_collate([self.id_to_cluster_id[id_] for id_ in id_batch])

        return (
            tensor_dict,
            tensor_lengths_dict,
            label_batch,
        )

    def __len__(self):
        return len(self.pos_id_list_batches)


class RowDataset(Dataset):
    def __init__(self, row_dict, row_encoder, batch_size, log_empty_vals=False):
        self.row_encoder = row_encoder
        self.row_list_batches = list(more_itertools.chunked(row_dict.values(), batch_size))
        self.log_empty_vals = log_empty_vals

    def __getitem__(self, idx):
        row_batch = self.row_list_batches[idx]

        tensor_dict = {attr: [] for attr in self.row_encoder.attr_info_dict.keys()}
        tensor_lengths_dict = {attr: [] for attr in self.row_encoder.attr_info_dict.keys()}
        for row in row_batch:
            row_tensor_dict, row_tensor_lengths_dict = self.row_encoder.build_tensor_dict(
                row, log_empty_vals=self.log_empty_vals
            )
            if all(l == 0 for l in row_tensor_lengths_dict.values()):  # ignore empty rows
                continue
            for attr in self.row_encoder.attr_info_dict.keys():
                tensor_dict[attr].append(row_tensor_dict[attr])
                tensor_lengths_dict[attr].append(row_tensor_lengths_dict[attr])

        for attr, one_hot_encoding_info in self.row_encoder.attr_info_dict.items():
            if one_hot_encoding_info.is_multitoken:
                tensor_dict[attr] = nn.utils.rnn.pad_sequence(tensor_dict[attr], batch_first=True)
            else:
                tensor_dict[attr] = default_collate(tensor_dict[attr])

        return tensor_dict, tensor_lengths_dict

    def __len__(self):
        return len(self.row_list_batches)
