import logging
import random

import more_itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from . import utils

logger = logging.getLogger(__name__)


def _collate_tensor_dict(record_batch, record_numericalizer):
    tensor_dict = {field: [] for field in record_numericalizer.field_config_dict.keys()}
    sequence_length_dict = {field: [] for field in record_numericalizer.field_config_dict.keys()}
    for record in record_batch:
        record_tensor_dict, record_sequence_length_dict = record_numericalizer.build_tensor_dict(
            record
        )
        for field in record_numericalizer.field_config_dict.keys():
            tensor_dict[field].append(record_tensor_dict[field])
            sequence_length_dict[field].append(record_sequence_length_dict[field])

    for field, field_config in record_numericalizer.field_config_dict.items():
        if field_config.is_multitoken:
            tensor_dict[field] = nn.utils.rnn.pad_sequence(tensor_dict[field], batch_first=True)
        else:
            tensor_dict[field] = default_collate(tensor_dict[field])
        sequence_length_dict[field] = default_collate(sequence_length_dict[field])
    return tensor_dict, sequence_length_dict


class ClusterDataset(Dataset):
    def __init__(
        self,
        record_dict,
        cluster_field,
        record_numericalizer,
        batch_size,
        random_seed,
    ):
        self.record_dict = record_dict
        self.record_numericalizer = record_numericalizer
        self.batch_size = batch_size

        if self.batch_size % 2 != 0:
            raise ValueError("ClusterDataset batch_size must be divisble by 2")

        cluster_dict = utils.record_dict_to_cluster_dict(record_dict, cluster_field)
        self.pos_pair_set = utils.cluster_dict_to_id_pairs(cluster_dict)
        self.rnd = random.Random(random_seed)

        self.id_batch_list = self._compute_id_batch_list()

    def _compute_id_batch_list(self):
        pos_pair_list = list(self.pos_pair_set)
        self.rnd.shuffle(pos_pair_list)
        id_iter = more_itertools.collapse(pos_pair_list)
        return list(more_itertools.chunked(id_iter, self.batch_size))

    def __getitem__(self, idx):
        id_batch = self.id_batch_list[idx]
        record_batch = [self.record_dict[id_] for id_ in id_batch]
        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            record_batch=record_batch,
            record_numericalizer=self.record_numericalizer,
        )
        label_list = torch.arange(0, len(id_batch) // 2).repeat_interleave(2)
        return tensor_dict, sequence_length_dict, label_list

    def __len__(self):
        return len(self.id_batch_list)


class RecordDataset(Dataset):
    def __init__(self, record_dict, record_numericalizer, batch_size):
        self.record_numericalizer = record_numericalizer
        self.record_batch_list = list(more_itertools.chunked(record_dict.values(), batch_size))

    def __getitem__(self, idx):
        record_batch = self.record_batch_list[idx]
        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            record_batch=record_batch,
            record_numericalizer=self.record_numericalizer,
        )
        return tensor_dict, sequence_length_dict

    def __len__(self):
        return len(self.record_batch_list)
