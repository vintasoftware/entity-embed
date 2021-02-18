import logging
import random

import more_itertools
import torch.nn as nn
from ordered_set import OrderedSet
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from . import utils

logger = logging.getLogger(__name__)


def _collate_tensor_dict(row_batch, row_numericalizer):
    tensor_dict = {attr: [] for attr in row_numericalizer.attr_info_dict.keys()}
    sequence_length_dict = {attr: [] for attr in row_numericalizer.attr_info_dict.keys()}
    for row in row_batch:
        row_tensor_dict, row_sequence_length_dict = row_numericalizer.build_tensor_dict(row)
        for attr in row_numericalizer.attr_info_dict.keys():
            tensor_dict[attr].append(row_tensor_dict[attr])
            sequence_length_dict[attr].append(row_sequence_length_dict[attr])

    for attr, numericalize_info in row_numericalizer.attr_info_dict.items():
        if numericalize_info.is_multitoken:
            tensor_dict[attr] = nn.utils.rnn.pad_sequence(tensor_dict[attr], batch_first=True)
        else:
            tensor_dict[attr] = default_collate(tensor_dict[attr])
    return tensor_dict, sequence_length_dict


class PairDataset(Dataset):
    def __init__(
        self,
        row_dict,
        cluster_attr,
        row_numericalizer,
        pos_pair_batch_size,
        neg_pair_batch_size,
        random_seed=42,
    ):
        self.row_dict = row_dict
        self.pair_list = list(utils.row_dict_to_id_pairs(row_dict, cluster_attr))
        self.id_to_cluster_id = {id_: row[cluster_attr] for id_, row in row_dict.items()}
        self.cluster_dict = utils.row_dict_to_cluster_dict(row_dict, cluster_attr)
        self.row_numericalizer = row_numericalizer
        self.random = random.Random(random_seed)

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
        if self.neg_batch_id_size >= len(self.cluster_dict):
            raise ValueError(
                f"{neg_pair_batch_size=} too large. "
                "The largest value possible is (total of clusters * (total of clusters - 1) / 2), "
                f"which is {len(self.cluster_dict) * (len(self.cluster_dict) - 1) / 2}"
            )

        self.random.shuffle(self.pair_list)
        pos_pair_list_batches_gen = more_itertools.chunked(self.pair_list, pos_pair_batch_size)
        self.pos_id_list_batches = [
            OrderedSet(id_ for pair in pair_batch for id_ in pair)
            for pair_batch in pos_pair_list_batches_gen
        ]

    def __getitem__(self, idx):
        pos_id_batch = self.pos_id_list_batches[idx]
        cluster_keys_batch = self.random.sample(self.cluster_dict.keys(), self.neg_batch_id_size)
        neg_id_batch = OrderedSet(
            self.random.choice(self.cluster_dict[k]) for k in cluster_keys_batch
        )
        neg_id_batch -= pos_id_batch

        id_batch = pos_id_batch | neg_id_batch

        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            row_batch=(self.row_dict[id_] for id_ in id_batch),
            row_numericalizer=self.row_numericalizer,
        )
        label_batch = default_collate([self.id_to_cluster_id[id_] for id_ in id_batch])

        return (
            tensor_dict,
            sequence_length_dict,
            label_batch,
        )

    def __len__(self):
        return len(self.pos_id_list_batches)


class PairwiseDataset(Dataset):
    def __init__(
        self,
        row_dict,
        true_pos_pair_set,
        true_neg_pair_set,
        row_numericalizer,
        pair_batch_size,
        random_seed=42,
    ):
        self.row_dict = row_dict
        pair_list = [
            (pair, int(pair in true_pos_pair_set))
            for pair in list(true_pos_pair_set | true_neg_pair_set)
        ]
        self.row_numericalizer = row_numericalizer
        self.random = random.Random(random_seed)

        self.random.shuffle(pair_list)
        self.pair_batches = list(more_itertools.chunked(pair_list, pair_batch_size))

    def __getitem__(self, idx):
        left_id_list = []
        right_id_list = []
        target_list = []
        for (left_id, right_id), target in self.pair_batches[idx]:
            left_id_list.append(left_id)
            right_id_list.append(right_id)
            target_list.append(target)

        left_tensor_dict, left_sequence_length_dict = _collate_tensor_dict(
            row_batch=(self.row_dict[id_] for id_ in left_id_list),
            row_numericalizer=self.row_numericalizer,
        )
        right_tensor_dict, right_sequence_length_dict = _collate_tensor_dict(
            row_batch=(self.row_dict[id_] for id_ in right_id_list),
            row_numericalizer=self.row_numericalizer,
        )
        target = default_collate(target_list)

        return (
            left_tensor_dict,
            left_sequence_length_dict,
            right_tensor_dict,
            right_sequence_length_dict,
            target,
        )

    def __len__(self):
        return len(self.pair_batches)


class RowDataset(Dataset):
    def __init__(self, row_dict, row_numericalizer, batch_size):
        self.row_numericalizer = row_numericalizer
        self.row_list_batches = list(more_itertools.chunked(row_dict.values(), batch_size))

    def __getitem__(self, idx):
        row_batch = self.row_list_batches[idx]
        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            row_batch=row_batch,
            row_numericalizer=self.row_numericalizer,
        )
        return tensor_dict, sequence_length_dict

    def __len__(self):
        return len(self.row_list_batches)
