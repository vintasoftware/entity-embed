import logging
import random

import more_itertools
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
        max_cluster_size_in_batch,
        random_seed,
    ):
        self.record_dict = record_dict
        self.record_numericalizer = record_numericalizer
        self.batch_size = batch_size
        cluster_dict = utils.record_dict_to_cluster_dict(record_dict, cluster_field)
        self.cluster_mapping = {
            id_: cluster_id for cluster_id, cluster in cluster_dict.items() for id_ in cluster
        }
        self.cluster_list = cluster_dict.values()
        self.singleton_id_list = [cluster[0] for cluster in self.cluster_list if len(cluster) == 1]
        self.batch_size = batch_size
        self.max_cluster_size_in_batch = max(max_cluster_size_in_batch, 2)
        self.rnd = random.Random(random_seed)

        self.id_batch_list = self._compute_id_batch_list()

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
        record_batch = [self.record_dict[id_] for id_ in id_batch]
        tensor_dict, sequence_length_dict = _collate_tensor_dict(
            record_batch=record_batch,
            record_numericalizer=self.record_numericalizer,
        )
        label_list = [self.cluster_mapping[id_] for id_ in id_batch]
        return tensor_dict, sequence_length_dict, default_collate(label_list)

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
