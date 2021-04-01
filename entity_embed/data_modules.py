import logging

import pytorch_lightning as pl
import torch

from .data_utils import utils
from .data_utils.datasets import ClusterDataset, RowDataset
from .helpers import build_loader_kwargs

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_FIELD = "__source"
DEFAULT_LEFT_SOURCE = "left"


def _check_for_common_rows(
    train_row_dict,
    valid_row_dict,
    test_row_dict,
):
    train_valid_common_ids = train_row_dict.keys() & (valid_row_dict.keys())
    if train_valid_common_ids:
        raise ValueError(
            "There are common IDs between "
            f"train_row_dict and valid_row_dict: {train_valid_common_ids}"
        )
    train_test_common_ids = train_row_dict.keys() & (test_row_dict.keys())
    if train_test_common_ids:
        raise ValueError(
            "There are common IDs between "
            f"train_row_dict and test_row_dict: {train_test_common_ids}"
        )
    valid_test_common_ids = valid_row_dict.keys() & (test_row_dict.keys())
    if valid_test_common_ids:
        raise ValueError(
            "There are common IDs between "
            f"valid_row_dict and test_row_dict: {valid_test_common_ids}"
        )


class DeduplicationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_row_dict,
        valid_row_dict,
        test_row_dict,
        cluster_field,
        row_numericalizer,
        batch_size,
        eval_batch_size,
        train_loader_kwargs=None,
        eval_loader_kwargs=None,
        random_seed=42,
        check_for_common_rows=True,
    ):
        super().__init__()

        if check_for_common_rows:
            _check_for_common_rows(
                train_row_dict,
                valid_row_dict,
                test_row_dict,
            )

        self.cluster_field = cluster_field
        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_loader_kwargs = build_loader_kwargs(train_loader_kwargs)
        self.eval_loader_kwargs = build_loader_kwargs(eval_loader_kwargs)
        self.random_seed = random_seed
        self.train_row_dict = train_row_dict
        self.valid_row_dict = valid_row_dict
        self.test_row_dict = test_row_dict

        self.train_pos_pair_set = None
        self.valid_pos_pair_set = None
        self.test_pos_pair_set = None

    def _set_pair_sets(self, stage):
        if stage == "fit":
            train_cluster_dict = utils.row_dict_to_cluster_dict(
                self.train_row_dict, self.cluster_field
            )
            valid_cluster_dict = utils.row_dict_to_cluster_dict(
                self.valid_row_dict, self.cluster_field
            )

            self.train_pos_pair_set = utils.cluster_dict_to_id_pairs(train_cluster_dict)
            self.valid_pos_pair_set = utils.cluster_dict_to_id_pairs(valid_cluster_dict)
        elif stage == "test":
            test_cluster_dict = utils.row_dict_to_cluster_dict(
                self.test_row_dict, self.cluster_field
            )

            self.test_pos_pair_set = utils.cluster_dict_to_id_pairs(test_cluster_dict)

    def setup(self, stage=None):
        self._set_pair_sets(stage)

        if stage == "fit":
            logger.info("Train positive pair count: %s", len(self.train_pos_pair_set))
            logger.info("Valid positive pair count: %s", len(self.valid_pos_pair_set))
        elif stage == "test":
            logger.info("Test positive pair count: %s", len(self.test_pos_pair_set))

    def train_dataloader(self):
        train_cluster_dataset = ClusterDataset.from_cluster_dict(
            row_dict=self.train_row_dict,
            cluster_field=self.cluster_field,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.batch_size,
            max_cluster_size_in_batch=self.batch_size // 3,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving contrastive learning:
            random_seed=self.random_seed + self.trainer.current_epoch
            if self.trainer
            else self.random_seed,
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
        train_row_dict,
        valid_row_dict,
        test_row_dict,
        source_field,
        left_source,
        cluster_field,
        row_numericalizer,
        batch_size,
        eval_batch_size,
        train_loader_kwargs=None,
        eval_loader_kwargs=None,
        random_seed=42,
        check_for_common_rows=True,
    ):
        super().__init__()

        if check_for_common_rows:
            _check_for_common_rows(
                train_row_dict,
                valid_row_dict,
                test_row_dict,
            )

        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_loader_kwargs = build_loader_kwargs(train_loader_kwargs)
        self.eval_loader_kwargs = build_loader_kwargs(eval_loader_kwargs)
        self.random_seed = random_seed
        self.train_row_dict = train_row_dict
        self.valid_row_dict = valid_row_dict
        self.test_row_dict = test_row_dict
        self.source_field = source_field
        self.left_source = left_source
        self.cluster_field = cluster_field

        # set later on setup
        self.train_pos_pair_set = None
        self.valid_pos_pair_set = None
        self.test_pos_pair_set = None

    def _set_pair_sets(self, stage):
        if stage == "fit":
            train_left_id_set, train_right_id_set = utils.row_dict_to_left_right_id_set(
                row_dict=self.train_row_dict,
                source_field=self.source_field,
                left_source=self.left_source,
            )
            valid_left_id_set, valid_right_id_set = utils.row_dict_to_left_right_id_set(
                row_dict=self.valid_row_dict,
                source_field=self.source_field,
                left_source=self.left_source,
            )
            train_cluster_dict = utils.row_dict_to_cluster_dict(
                self.train_row_dict, self.cluster_field
            )
            valid_cluster_dict = utils.row_dict_to_cluster_dict(
                self.valid_row_dict, self.cluster_field
            )

            self.train_pos_pair_set = utils.cluster_dict_to_id_pairs(
                train_cluster_dict, left_id_set=train_left_id_set, right_id_set=train_right_id_set
            )
            self.valid_pos_pair_set = utils.cluster_dict_to_id_pairs(
                valid_cluster_dict, left_id_set=valid_left_id_set, right_id_set=valid_right_id_set
            )
        elif stage == "test":
            test_left_id_set, test_right_id_set = utils.row_dict_to_left_right_id_set(
                row_dict=self.test_row_dict,
                source_field=self.source_field,
                left_source=self.left_source,
            )
            test_cluster_dict = utils.row_dict_to_cluster_dict(
                self.test_row_dict, self.cluster_field
            )
            self.test_pos_pair_set = utils.cluster_dict_to_id_pairs(
                test_cluster_dict, left_id_set=test_left_id_set, right_id_set=test_right_id_set
            )

    def setup(self, stage=None):
        self._set_pair_sets(stage)

        if stage == "fit":
            logger.info("Train positive pair count: %s", len(self.train_pos_pair_set))
            logger.info("Valid positive pair count: %s", len(self.valid_pos_pair_set))
        elif stage == "test":
            logger.info("Test positive pair count: %s", len(self.test_pos_pair_set))

    def train_dataloader(self):
        train_cluster_dataset = ClusterDataset.from_cluster_dict(
            row_dict=self.train_row_dict,
            cluster_field=self.cluster_field,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.batch_size,
            max_cluster_size_in_batch=self.batch_size // 3,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving contrastive learning:
            random_seed=self.random_seed + self.trainer.current_epoch
            if self.trainer
            else self.random_seed,
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
