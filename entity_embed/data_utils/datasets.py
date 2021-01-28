import logging
import math
import random

import more_itertools
import nlpaug.augmenter.char as nac
import torch
import torch.nn as nn
from Levenshtein import ratio
from nlpaug.util import Action
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


class AttrAugTripletDataset(Dataset):
    def __init__(
        self,
        encoder,
        val_list,
        aug_choice_weights=None,
        augmenters=None,
        pos_aug_char_p=0.3,
        neg_aug_char_p=0.7,
        aug_min_char=3,
        random_seed=42,
    ):
        self.encoder = encoder
        self.val_list = val_list
        self.random = random.Random(random_seed)

        if aug_choice_weights is None:
            self.aug_choice_weights = [0.6, 0.2, 0.05, 0.05, 0.05, 0.05]
        else:
            self.aug_choice_weights = aug_choice_weights
        self.aug_min_char = aug_min_char

        include_upper_case = any(c.isupper() for c in encoder.alphabet)

        if augmenters is None:
            self.augmenters = [
                (
                    nac.KeyboardAug(
                        aug_char_p=pos_aug_char_p,
                        include_upper_case=include_upper_case,
                        min_char=aug_min_char,
                    ),
                    nac.KeyboardAug(
                        aug_char_p=neg_aug_char_p,
                        include_upper_case=include_upper_case,
                        min_char=aug_min_char,
                    ),
                ),
                (
                    nac.OcrAug(aug_char_p=pos_aug_char_p, min_char=aug_min_char),
                    nac.OcrAug(aug_char_p=neg_aug_char_p, min_char=aug_min_char),
                ),
                *[
                    (
                        nac.RandomCharAug(
                            action=action,
                            aug_char_p=pos_aug_char_p,
                            include_upper_case=include_upper_case,
                            min_char=aug_min_char,
                        ),
                        nac.RandomCharAug(
                            action=action,
                            aug_char_p=neg_aug_char_p,
                            include_upper_case=include_upper_case,
                            min_char=aug_min_char,
                        ),
                    )
                    for action in [Action.INSERT, Action.SUBSTITUTE, Action.SWAP, Action.DELETE]
                ],
            ]
        else:
            self.augmenters = augmenters

    def get_item(self, anchor):
        [(pos_aug, neg_aug)] = self.random.choices(
            self.augmenters, weights=self.aug_choice_weights, k=1
        )
        pos = pos_aug.augment(anchor).lower()
        neg = neg_aug.augment(anchor).lower()
        if len(pos) > self.encoder.max_str_len:
            pos = pos[: self.encoder.max_str_len]
        if len(neg) > self.encoder.max_str_len:
            neg = neg[: self.encoder.max_str_len]

        return (
            pos,
            neg,
            self.encoder.build_tensor(anchor),
            self.encoder.build_tensor(pos),
            self.encoder.build_tensor(neg),
        )

    def __getitem__(self, idx):
        anchor = self.val_list[idx]
        pos, neg, anchor_t, pos_t, neg_t = self.get_item(anchor)
        pos_dist = 1 - ratio(anchor, pos)
        neg_dist = 1 - ratio(anchor, neg)
        if pos_dist > neg_dist:
            logger.warning(
                f"Inverted positive and negative distance for strings: {anchor=} {pos=} {neg=}"
            )
            pos_t, neg_t = neg_t, pos_t
            pos_dist, neg_dist = neg_dist, pos_dist
        return (
            anchor_t,
            pos_t,
            neg_t,
            pos_dist,
            neg_dist,
        )

    def __len__(self):
        return len(self.val_list)


class AugTripletDataset(Dataset):
    def __init__(
        self,
        row_list,
        attr_to_encoder,
        attr_to_aug_kwargs=None,
        attr_for_aug_list=None,
        random_seed=42,
    ):
        self.row_list = row_list
        self.attr_to_encoder = attr_to_encoder
        attr_to_aug_kwargs = attr_to_aug_kwargs or {}
        if attr_for_aug_list is None:
            self.attr_for_aug_list = list(row_list[0].keys())
        else:
            self.attr_for_aug_list = attr_for_aug_list

        self.attr_to_attr_dataset = {}
        for attr in self.attr_for_aug_list:
            attr_dataset = AttrAugTripletDataset(
                encoder=attr_to_encoder[attr],
                val_list=[],  # don't need to pass val_list, we'll call get_item manually
                random_seed=random_seed,
                **(attr_to_aug_kwargs.get(attr) or {}),
            )
            self.attr_to_attr_dataset[attr] = attr_dataset

    def __getitem__(self, idx):
        anchor_row = self.row_list[idx]
        anchor_tensor_list = []
        pos_tensor_list = []
        neg_tensor_list = []

        for attr, attr_dataset in self.attr_to_attr_dataset.items():
            anchor = anchor_row[attr]
            __, __, anchor_t, pos_t, neg_t = attr_dataset.get_item(anchor)
            anchor_tensor_list.append(anchor_t)
            pos_tensor_list.append(pos_t)
            neg_tensor_list.append(neg_t)

        return anchor_tensor_list, pos_tensor_list, neg_tensor_list

    def __len__(self):
        return len(self.row_list)


def _row_to_tensor_list(attr_to_encoder, row, log_empty_vals=False):
    tensor_list = []

    for attr, encoder in attr_to_encoder.items():
        if not row[attr]:
            logger.warning(f"Found empty {attr=} at id_={row['id']}")
        tensor_list.append(encoder.build_tensor(row[attr]))
    return tensor_list


def _pad_aware_collate(attr_to_encoder, tensor_list_batch):
    tensor_list_batch = zip(*tensor_list_batch)  # transpose, batch per column
    collated_tensor_list = []
    tensor_lengths_list = []

    for encoder, tensor_list in zip(attr_to_encoder.values(), tensor_list_batch):
        if encoder.is_multitoken:
            t = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
            collated_tensor_list.append(t)
            t_lengths = [ti.size(0) for ti in tensor_list]
            tensor_lengths_list.append(t_lengths)
        else:
            t = default_collate(tensor_list)
            collated_tensor_list.append(t)
            tensor_lengths_list.append(None)
    return collated_tensor_list, tensor_lengths_list


class PairDataset(Dataset):
    def __init__(
        self,
        row_dict,
        attr_to_encoder,
        pair_set,
        id_to_cluster_id,
        pos_pair_batch_size,
        neg_pair_batch_size,
        log_empty_vals=False,
        random_seed=42,
    ):
        self.row_dict = row_dict
        self.attr_to_encoder = attr_to_encoder
        self.pair_list = list(pair_set)
        self.id_to_cluster_id = id_to_cluster_id
        self.id_list = list(id_to_cluster_id.keys())
        self.log_empty_vals = log_empty_vals
        self.random = random.Random(random_seed)

        # compute positive solution for y of y = (n * (n - 1)) / 2
        self.neg_batch_id_size = int((1 + math.sqrt(1 + 8 * neg_pair_batch_size)) / 2)
        actual_neg_pair_batch_size = (self.neg_batch_id_size * (self.neg_batch_id_size - 1)) // 2
        if actual_neg_pair_batch_size != neg_pair_batch_size:
            logger.warning(
                "Since PairDataset samples negative pairs by rows, "
                "neg_pair_batch_size must fit the equation y = (n * (n - 1)) / 2, "
                "where n is number of rows, and y is neg_pair_batch_size. "
                f"Therefore, requested {neg_pair_batch_size=} is impossible. "
                f"Closest lower pair number is {actual_neg_pair_batch_size}."
            )

        if pos_pair_batch_size > len(self.pair_list):
            raise ValueError(
                f"{pos_pair_batch_size=} is greater than {len(self.pair_list)=}, "
                "please use a smaller pos_pair_batch_size."
            )
        pos_pair_list_batches_gen = more_itertools.chunked(self.pair_list, pos_pair_batch_size)
        self.pos_id_list_batches = [
            sorted(set(id_ for pair in pair_batch for id_ in pair))
            for pair_batch in pos_pair_list_batches_gen
        ]

    def __getitem__(self, idx):
        pos_id_batch = self.pos_id_list_batches[idx]
        neg_id_batch = self.random.sample(self.id_list, self.neg_batch_id_size)
        id_batch = pos_id_batch + neg_id_batch

        tensor_list_batch = [
            _row_to_tensor_list(
                self.attr_to_encoder, self.row_dict[id_], log_empty_vals=self.log_empty_vals
            )
            for id_ in id_batch
        ]
        label_batch = [self.id_to_cluster_id[id_] for id_ in id_batch]

        collated_tensor_list, tensor_lengths = _pad_aware_collate(
            self.attr_to_encoder, tensor_list_batch
        )

        return collated_tensor_list, tensor_lengths, default_collate(label_batch)

    def __len__(self):
        return len(self.pos_id_list_batches)


class RowDataset(Dataset):
    def __init__(self, attr_to_encoder, row_list, batch_size, log_empty_vals=False):
        self.attr_to_encoder = attr_to_encoder
        self.log_empty_vals = log_empty_vals

        self.row_list_batches = list(more_itertools.chunked(row_list, batch_size))

    def __getitem__(self, idx):
        row_batch = self.row_list_batches[idx]

        tensor_list_batch = [
            _row_to_tensor_list(self.attr_to_encoder, row, log_empty_vals=self.log_empty_vals)
            for row in row_batch
        ]
        collated_tensor_list, tensor_lengths_list = _pad_aware_collate(
            self.attr_to_encoder, tensor_list_batch
        )

        return collated_tensor_list, tensor_lengths_list


    def __len__(self):
        return len(self.row_list_batches)
