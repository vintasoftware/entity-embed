import logging

import more_itertools
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from . import utils

logger = logging.getLogger(__name__)


def collate_cluster_tensor_dict(cluster_batch, row_numericalizer):
    tensor_dict, sequence_length_dict = _collate_tensor_dict(
        row_batch=[
            row for cluster_row_dict, label in cluster_batch for row in cluster_row_dict.values()
        ],
        row_numericalizer=row_numericalizer,
    )
    ids = [id_ for cluster_row_dict, label in cluster_batch for id_ in cluster_row_dict.keys()]
    labels = [
        label for cluster_row_dict, label in cluster_batch for row in cluster_row_dict.values()
    ]
    return default_collate(ids), tensor_dict, sequence_length_dict, default_collate(labels)


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


class ClusterDataset(Dataset):
    def __init__(
        self,
        row_dict,
        cluster_dict,
        row_numericalizer,
    ):
        self.row_dict = row_dict
        self.cluster_list = list(cluster_dict.items())
        self.row_numericalizer = row_numericalizer

    @classmethod
    def from_cluster_dict(
        cls,
        row_dict,
        cluster_attr,
        row_numericalizer,
    ):
        cluster_dict = utils.row_dict_to_cluster_dict(row_dict, cluster_attr)
        return ClusterDataset(
            row_dict=row_dict,
            cluster_dict=cluster_dict,
            row_numericalizer=row_numericalizer,
        )

    @classmethod
    def from_pairs(
        cls,
        row_dict,
        true_pair_set,
        row_numericalizer,
    ):
        __, cluster_dict = utils.id_pairs_to_cluster_mapping_and_dict(id_pairs=true_pair_set)
        transitive_true_pair_set = utils.cluster_dict_to_id_pairs(cluster_dict)
        transitive_new_pairs_count = len(transitive_true_pair_set - true_pair_set)
        if transitive_new_pairs_count > 0:
            logger.warning(
                f"true_pair_set has {transitive_new_pairs_count} less elements "
                "than transitive_true_pair_set. "
                "This means there are transitive true pairs not included in true_pair_set."
            )
        return ClusterDataset(
            row_dict=row_dict,
            cluster_dict=cluster_dict,
            row_numericalizer=row_numericalizer,
        )

    def __getitem__(self, idx):
        label, cluster = self.cluster_list[idx]
        cluster_row_dict = {id_: self.row_dict[id_] for id_ in cluster}
        return cluster_row_dict, label

    def __len__(self):
        return len(self.cluster_list)


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
