import logging

import pytorch_lightning as pl
import torch

from .data_utils import utils
from .data_utils.datasets import ClusterDataset, RowDataset
from .helpers import build_loader_kwargs

logger = logging.getLogger(__name__)


class DeduplicationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        row_dict,
        cluster_attr,
        row_numericalizer,
        batch_size,
        eval_batch_size,
        train_cluster_len,
        valid_cluster_len,
        test_cluster_len,
        train_loader_kwargs=None,
        eval_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__()
        self.row_dict = row_dict
        self.cluster_attr = cluster_attr
        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_cluster_len = train_cluster_len
        self.valid_cluster_len = valid_cluster_len
        self.test_cluster_len = test_cluster_len
        self.train_loader_kwargs = build_loader_kwargs(train_loader_kwargs)
        self.eval_loader_kwargs = build_loader_kwargs(eval_loader_kwargs)
        self.random_seed = random_seed

        self.valid_true_pair_set = None
        self.test_true_pair_set = None
        self.train_row_dict = None
        self.valid_row_dict = None
        self.test_row_dict = None

    def setup(self, stage=None):
        cluster_dict = utils.row_dict_to_cluster_dict(self.row_dict, self.cluster_attr)

        train_cluster_dict, valid_cluster_dict, test_cluster_dict = utils.split_clusters(
            cluster_dict,
            train_len=self.train_cluster_len,
            valid_len=self.valid_cluster_len,
            test_len=self.test_cluster_len,
            random_seed=self.random_seed,
        )
        self.valid_true_pair_set = utils.cluster_dict_to_id_pairs(valid_cluster_dict)
        self.test_true_pair_set = utils.cluster_dict_to_id_pairs(test_cluster_dict)
        logger.info("Train pair count: %s", utils.count_cluster_dict_pairs(train_cluster_dict))
        logger.info("Valid pair count: %s", len(self.valid_true_pair_set))
        logger.info("Test pair count: %s", len(self.test_true_pair_set))

        (
            self.train_row_dict,
            self.valid_row_dict,
            self.test_row_dict,
        ) = utils.cluster_dicts_to_row_dicts(
            row_dict=self.row_dict,
            train_cluster_dict=train_cluster_dict,
            valid_cluster_dict=valid_cluster_dict,
            test_cluster_dict=test_cluster_dict,
        )

        # If not test, drop test values
        if stage == "fit":
            self.test_true_pair_set = None
            self.test_row_dict = None
        elif stage == "test":
            self.valid_true_pair_set = None
            self.train_row_dict = None
            self.valid_row_dict = None

    def train_dataloader(self):
        train_cluster_dataset = ClusterDataset.from_cluster_dict(
            row_dict=self.train_row_dict,
            cluster_attr=self.cluster_attr,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.batch_size,
            max_cluster_size_in_batch=self.batch_size // 3,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving contrastive learning:
            random_seed=self.random_seed + self.trainer.current_epoch,
        )
        train_cluster_loader = torch.utils.data.DataLoader(
            train_cluster_dataset,
            batch_size=None,  # batch size is set on ClusterDataset
            shuffle=False,  # shuffling is implemented on ClusterDataset
            **self.train_loader_kwargs,
        )
        return train_cluster_loader

    def val_dataloader(self):
        valid_row_dataset = RowDataset(
            row_dict=self.valid_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.eval_batch_size,
        )
        valid_row_loader = torch.utils.data.DataLoader(
            valid_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return valid_row_loader

    def test_dataloader(self):
        test_row_dataset = RowDataset(
            row_dict=self.test_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.eval_batch_size,
        )
        test_row_loader = torch.utils.data.DataLoader(
            test_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return test_row_loader


class LinkageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        row_dict,
        left_id_set,
        right_id_set,
        row_numericalizer,
        batch_size,
        eval_batch_size,
        cluster_attr=None,
        true_pair_set=None,
        train_cluster_len=None,
        valid_cluster_len=None,
        test_cluster_len=None,
        train_true_pair_set=None,
        valid_true_pair_set=None,
        test_true_pair_set=None,
        train_loader_kwargs=None,
        eval_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__()

        self.row_dict = row_dict
        self.left_id_set = left_id_set
        self.right_id_set = right_id_set
        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_loader_kwargs = build_loader_kwargs(train_loader_kwargs)
        self.eval_loader_kwargs = build_loader_kwargs(eval_loader_kwargs)
        self.random_seed = random_seed
        if train_true_pair_set:
            if valid_true_pair_set is None:
                raise ValueError(
                    "valid_true_pair_set can't be None when train_true_pair_set is provided"
                )
            if test_true_pair_set is None:
                raise ValueError(
                    "test_true_pair_set can't be None when train_true_pair_set is provided"
                )
            self.train_true_pair_set = train_true_pair_set
            self.valid_true_pair_set = valid_true_pair_set
            self.test_true_pair_set = test_true_pair_set
        elif true_pair_set:
            if train_cluster_len is None:
                raise ValueError("train_cluster_len can't be None")
            if valid_cluster_len is None:
                raise ValueError("valid_cluster_len can't be None")
            __, cluster_dict = utils.id_pairs_to_cluster_mapping_and_dict(true_pair_set)
            self._split_clusters(
                cluster_dict=cluster_dict,
                train_cluster_len=train_cluster_len,
                valid_cluster_len=valid_cluster_len,
                test_cluster_len=test_cluster_len,
            )
        elif cluster_attr:
            if train_cluster_len is None:
                raise ValueError("train_cluster_len can't be None")
            if valid_cluster_len is None:
                raise ValueError("valid_cluster_len can't be None")
            cluster_dict = utils.row_dict_to_cluster_dict(row_dict, cluster_attr)
            self._split_clusters(
                cluster_dict=cluster_dict,
                train_cluster_len=train_cluster_len,
                valid_cluster_len=valid_cluster_len,
                test_cluster_len=test_cluster_len,
            )
        else:
            raise Exception("Please set one of train_true_pair_set, true_pair_set or cluster_attr")
        self.train_row_dict = None
        self.valid_row_dict = None
        self.test_row_dict = None

    def _split_clusters(self, cluster_dict, train_cluster_len, valid_cluster_len, test_cluster_len):
        train_cluster_dict, valid_cluster_dict, test_cluster_dict = utils.split_clusters(
            cluster_dict,
            train_len=train_cluster_len,
            valid_len=valid_cluster_len,
            test_len=test_cluster_len,
            random_seed=self.random_seed,
        )
        self.train_true_pair_set = utils.cluster_dict_to_id_pairs(
            cluster_dict=train_cluster_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
        )
        self.valid_true_pair_set = utils.cluster_dict_to_id_pairs(
            cluster_dict=valid_cluster_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
        )
        self.test_true_pair_set = utils.cluster_dict_to_id_pairs(
            cluster_dict=test_cluster_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
        )

    def setup(self, stage=None):
        logger.info("Train pair count: %s", len(self.train_true_pair_set))
        logger.info("Valid pair count: %s", len(self.valid_true_pair_set))
        logger.info("Test pair count: %s", len(self.test_true_pair_set))

        if stage == "fit":
            self.train_row_dict = {
                id_: self.row_dict[id_] for pair in self.train_true_pair_set for id_ in pair
            }
            self.valid_row_dict = {
                id_: self.row_dict[id_] for pair in self.valid_true_pair_set for id_ in pair
            }
        elif stage == "test":
            self.test_row_dict = {
                id_: self.row_dict[id_] for pair in self.test_true_pair_set for id_ in pair
            }

    def train_dataloader(self):
        train_cluster_dataset = ClusterDataset.from_pairs(
            row_dict=self.train_row_dict,
            true_pair_set=self.train_true_pair_set,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.batch_size,
            max_cluster_size_in_batch=self.batch_size // 3,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving contrastive learning:
            random_seed=self.random_seed + self.trainer.current_epoch,
        )
        train_cluster_loader = torch.utils.data.DataLoader(
            train_cluster_dataset,
            batch_size=None,  # batch size is set on ClusterDataset
            shuffle=False,  # shuffling is implemented on ClusterDataset
            **self.train_loader_kwargs,
        )
        return train_cluster_loader

    def val_dataloader(self):
        valid_row_dataset = RowDataset(
            row_dict=self.valid_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.eval_batch_size,
        )
        valid_row_loader = torch.utils.data.DataLoader(
            valid_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return valid_row_loader

    def test_dataloader(self):
        test_row_dataset = RowDataset(
            row_dict=self.test_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.eval_batch_size,
        )
        test_row_loader = torch.utils.data.DataLoader(
            test_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return test_row_loader

    def separate_dict_left_right(self, d):
        return utils.separate_dict_left_right(
            d, left_id_set=self.left_id_set, right_id_set=self.right_id_set
        )
