import logging
import random

import more_itertools
import torch.nn as nn
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


class ClusterDataset(Dataset):
    def __init__(
        self,
        row_dict,
        row_numericalizer,
        cluster_dict,
        cluster_mapping,
        batch_size,
        max_cluster_size_in_batch,
        random_seed,
    ):
        self.row_dict = row_dict
        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.rnd = random.Random(random_seed)
        self.cluster_list = cluster_dict.values()
        self.singleton_id_list = [cluster[0] for cluster in self.cluster_list if len(cluster) == 1]
        self.cluster_mapping = cluster_mapping
        self.batch_size = batch_size
        self.max_cluster_size_in_batch = max_cluster_size_in_batch
        self.rnd = random.Random(random_seed)

        self.id_batch_list = self._compute_id_batch_list()

    @classmethod
    def from_cluster_dict(
        cls,
        row_dict,
        cluster_attr,
        row_numericalizer,
        batch_size,
        max_cluster_size_in_batch,
        random_seed,
    ):
        cluster_dict = utils.row_dict_to_cluster_dict(row_dict, cluster_attr)
        cluster_mapping = {
            id_: cluster_id for cluster_id, cluster in cluster_dict.items() for id_ in cluster
        }
        return ClusterDataset(
            row_dict=row_dict,
            row_numericalizer=row_numericalizer,
            cluster_dict=cluster_dict,
            cluster_mapping=cluster_mapping,
            batch_size=batch_size,
            max_cluster_size_in_batch=max_cluster_size_in_batch,
            random_seed=random_seed,
        )

    @classmethod
    def from_pairs(
        cls,
        row_dict,
        true_pair_set,
        row_numericalizer,
        batch_size,
        max_cluster_size_in_batch,
        random_seed,
    ):
        # Note: there's a caveat in using pairs: you don't have singleton clusters!
        cluster_mapping, cluster_dict = utils.id_pairs_to_cluster_mapping_and_dict(
            id_pairs=true_pair_set
        )
        return ClusterDataset(
            row_dict=row_dict,
            row_numericalizer=row_numericalizer,
            cluster_dict=cluster_dict,
            cluster_mapping=cluster_mapping,
            batch_size=batch_size,
            max_cluster_size_in_batch=max_cluster_size_in_batch,
            random_seed=random_seed,
        )

    def _compute_id_batch_list(self):
        # copy cluster_list and singleton_id_list
        cluster_list = list(self.cluster_list)
        singleton_id_list = list(self.singleton_id_list)

        # break up large clusters
        cluster_list = [
            cluster_chunk
            for cluster in cluster_list
            for cluster_chunk in more_itertools.chunked(cluster, n=self.max_cluster_size_in_batch)
        ]

        # randomize cluster order
        self.rnd.shuffle(cluster_list)

        # prepare batches using both clusters and singletons
        id_batch_list = []
        while cluster_list:
            id_batch = []

            # add next cluster to batch
            next_cluster = cluster_list.pop()
            id_batch.extend(next_cluster)

            # fill batch with other clusters
            while cluster_list and len(id_batch) < self.batch_size:
                next_cluster = cluster_list.pop()
                if len(id_batch) + len(next_cluster) <= self.batch_size:
                    id_batch.extend(next_cluster)
                else:
                    # next_cluster is too large to fit the batch,
                    # add it back to cluster_list
                    cluster_list.append(next_cluster)
                    # and break this while
                    break

            # fill rest ot batch with singletons, if possible
            while singleton_id_list and len(id_batch) < self.batch_size:
                singleton_id = singleton_id_list.pop()
                id_batch.append(singleton_id)

            id_batch_list.append(id_batch)

        return id_batch_list

    def __getitem__(self, idx):
        id_batch = self.id_batch_list[idx]
        row_batch = [self.row_dict[id_] for id_ in id_batch]
        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            row_batch=row_batch,
            row_numericalizer=self.row_numericalizer,
        )
        label_list = [self.cluster_mapping[id_] for id_ in id_batch]
        return tensor_dict, sequence_length_dict, default_collate(label_list)

    def __len__(self):
        return len(self.id_batch_list)


class RowDataset(Dataset):
    def __init__(self, row_dict, row_numericalizer, batch_size):
        self.row_numericalizer = row_numericalizer
        self.row_batch_list = list(more_itertools.chunked(row_dict.values(), batch_size))

    def __getitem__(self, idx):
        row_batch = self.row_batch_list[idx]
        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            row_batch=row_batch,
            row_numericalizer=self.row_numericalizer,
        )
        return tensor_dict, sequence_length_dict

    def __len__(self):
        return len(self.row_batch_list)
