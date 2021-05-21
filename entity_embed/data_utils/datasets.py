import logging
import random

import more_itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


def _collate_tensor_dict(record_batch, record_numericalizer):
    tensor_dict = {field: [] for field in record_numericalizer.field_config_dict.keys()}
    sequence_length_dict = {field: [] for field in record_numericalizer.field_config_dict.keys()}
    transformer_attention_mask_dict = {
        field: None for field in record_numericalizer.field_config_dict.keys()
    }

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
        elif field_config.is_semantic:
            transformer_pad_output = field_config.transformer_tokenizer.pad(
                {"input_ids": tensor_dict[field]},
                return_attention_mask=True,
            )
            tensor_dict[field] = transformer_pad_output["input_ids"]
            transformer_attention_mask_dict[field] = transformer_pad_output["attention_mask"]
        else:
            tensor_dict[field] = default_collate(tensor_dict[field])
        sequence_length_dict[field] = default_collate(sequence_length_dict[field])

    return tensor_dict, sequence_length_dict, transformer_attention_mask_dict


class BlockDataset(Dataset):
    def __init__(
        self,
        record_dict,
        pos_pair_set,
        record_numericalizer,
        batch_size,
        random_seed,
    ):
        self.record_dict = record_dict
        self.record_numericalizer = record_numericalizer
        self.batch_size = batch_size

        if self.batch_size % 2 != 0:
            raise ValueError("BlockDataset batch_size must be divisble by 2")

        self.pos_pair_set = pos_pair_set
        self.rnd = random.Random(random_seed)

        self.id_batch_list = self._compute_id_batch_list()

    def _compute_id_batch_list(self):
        # First sort pos_pair_set to ensure deterministic order
        pos_pair_list = sorted(self.pos_pair_set)
        # Then shuffle
        self.rnd.shuffle(pos_pair_list)
        # Collapse the pairs into sequential IDs.
        # Records that represent a positive pairs will be consecutive.
        id_iter = more_itertools.collapse(pos_pair_list)
        # Finally, divide into batches
        return list(more_itertools.chunked(id_iter, self.batch_size))

    def __getitem__(self, idx):
        id_batch = self.id_batch_list[idx]
        record_batch = [self.record_dict[id_] for id_ in id_batch]
        tensor_dict, sequence_length_dict, transformer_attention_mask_dict = _collate_tensor_dict(
            record_batch=record_batch,
            record_numericalizer=self.record_numericalizer,
        )
        # Consecutive IDs are positive pairs
        label_list = torch.arange(0, len(id_batch) // 2).repeat_interleave(2)
        return tensor_dict, sequence_length_dict, transformer_attention_mask_dict, label_list

    def __len__(self):
        return len(self.id_batch_list)


class MatcherDataset(Dataset):
    def __init__(
        self,
        record_dict,
        pos_pair_set,
        neg_pair_set,
        pair_numericalizer,
        batch_size,
        random_seed,
    ):
        self.record_dict = record_dict
        self.pair_numericalizer = pair_numericalizer
        self.pos_pair_set = pos_pair_set
        self.neg_pair_set = neg_pair_set
        self.batch_size = batch_size
        if random_seed is not None:
            self.rnd = random.Random(random_seed)
        else:
            self.rnd = None

        self.pair_batch_list, self.label_batch_list = self._compute_pair_label_batch_list()

    def _compute_pair_label_batch_list(self):
        # copy deterministically pos_pair_set and neg_pair_set
        pos_pair_list = sorted(self.pos_pair_set)
        neg_pair_list = sorted(self.neg_pair_set)

        # shuffle lists
        if self.rnd:
            self.rnd.shuffle(pos_pair_list)
            self.rnd.shuffle(neg_pair_list)

        # divide batches following pos/neg proportion
        pos_proportion = len(pos_pair_list) / (len(pos_pair_list) + len(neg_pair_list))
        pos_per_batch = max(int(pos_proportion * self.batch_size), 2)
        pair_batch_list = []
        label_batch_list = []

        while pos_pair_list or neg_pair_list:
            curr_batch_pos_count = 0
            pair_batch = []
            label_batch = []
            while len(pair_batch) < self.batch_size and (pos_pair_list or neg_pair_list):
                if pos_pair_list and (curr_batch_pos_count < pos_per_batch or not neg_pair_list):
                    pair_batch.append(pos_pair_list.pop())
                    label_batch.append(1.0)
                    curr_batch_pos_count += 1
                else:
                    pair_batch.append(neg_pair_list.pop())
                    label_batch.append(0.0)

            pair_batch_list.append(pair_batch)
            label_batch_list.append(label_batch)

        return pair_batch_list, label_batch_list

    def __getitem__(self, idx):
        id_batch_left, id_batch_right = list(zip(*self.pair_batch_list[idx]))
        record_batch_left = [self.record_dict[id_] for id_ in id_batch_left]
        record_batch_right = [self.record_dict[id_] for id_ in id_batch_right]
        pair_tensor_batch = self.pair_numericalizer.build_tensor_batch(
            record_batch_left, record_batch_right
        )
        label_batch = self.label_batch_list[idx]

        return (
            pair_tensor_batch,
            default_collate(label_batch),
        )

    def __len__(self):
        return len(self.pair_batch_list)


class RecordDataset(Dataset):
    def __init__(self, record_dict, record_numericalizer, batch_size):
        self.record_numericalizer = record_numericalizer
        self.record_batch_list = list(more_itertools.chunked(record_dict.values(), batch_size))

    def __getitem__(self, idx):
        record_batch = self.record_batch_list[idx]
        tensor_dict, sequence_length_dict, transformer_attention_mask_dict = _collate_tensor_dict(
            record_batch=record_batch,
            record_numericalizer=self.record_numericalizer,
        )
        return tensor_dict, sequence_length_dict, transformer_attention_mask_dict

    def __len__(self):
        return len(self.record_batch_list)
