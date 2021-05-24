import logging

import pytorch_lightning as pl
import torch

from .data_utils import utils
from .data_utils.datasets import BlockDataset, RecordDataset
from .helpers import build_loader_kwargs

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_FIELD = "__source"
DEFAULT_LEFT_SOURCE = "left"


def _check_for_common_records(
    train_record_dict,
    valid_record_dict,
    test_record_dict,
):
    train_valid_common_ids = train_record_dict.keys() & (valid_record_dict.keys())
    if train_valid_common_ids:
        raise ValueError(
            "There are common IDs between "
            f"train_record_dict and valid_record_dict: {train_valid_common_ids}"
        )
    train_test_common_ids = train_record_dict.keys() & (test_record_dict.keys())
    if train_test_common_ids:
        raise ValueError(
            "There are common IDs between "
            f"train_record_dict and test_record_dict: {train_test_common_ids}"
        )
    valid_test_common_ids = valid_record_dict.keys() & (test_record_dict.keys())
    if valid_test_common_ids:
        raise ValueError(
            "There are common IDs between "
            f"valid_record_dict and test_record_dict: {valid_test_common_ids}"
        )


class DeduplicationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_record_dict,
        valid_record_dict,
        test_record_dict,
        cluster_field,
        record_numericalizer,
        batch_size,
        eval_batch_size,
        train_loader_kwargs=None,
        eval_loader_kwargs=None,
        random_seed=42,
        check_for_common_records=True,
    ):
        super().__init__()

        if check_for_common_records:
            _check_for_common_records(
                train_record_dict,
                valid_record_dict,
                test_record_dict,
            )

        self.record_numericalizer = record_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_loader_kwargs = build_loader_kwargs(train_loader_kwargs)
        self.eval_loader_kwargs = build_loader_kwargs(eval_loader_kwargs)
        self.random_seed = random_seed
        self.train_record_dict = train_record_dict
        self.valid_record_dict = valid_record_dict
        self.test_record_dict = test_record_dict

        self._train_pos_pair_set = utils.record_dict_to_id_pairs(train_record_dict, cluster_field)
        self._valid_pos_pair_set = utils.record_dict_to_id_pairs(valid_record_dict, cluster_field)
        self._test_pos_pair_set = utils.record_dict_to_id_pairs(test_record_dict, cluster_field)

    def _set_pair_sets(self, stage):
        if stage == "fit":
            self.train_pos_pair_set = self._train_pos_pair_set
            self.valid_pos_pair_set = self._valid_pos_pair_set
        elif stage == "test":
            self.test_pos_pair_set = self._test_pos_pair_set

    def setup(self, stage=None):
        self._set_pair_sets(stage)

        if stage == "fit":
            logger.info("Train positive pair count: %s", len(self.train_pos_pair_set))
            logger.info("Valid positive pair count: %s", len(self.valid_pos_pair_set))
        elif stage == "test":
            logger.info("Test positive pair count: %s", len(self.test_pos_pair_set))

    def train_dataloader(self):
        train_block_dataset = BlockDataset(
            record_dict=self.train_record_dict,
            pos_pair_set=self.train_pos_pair_set,
            record_numericalizer=self.record_numericalizer,
            batch_size=self.batch_size,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving learning:
            random_seed=(
                self.random_seed + self.trainer.current_epoch if self.trainer else self.random_seed
            ),
        )
        train_block_loader = torch.utils.data.DataLoader(
            train_block_dataset,
            batch_size=None,  # batch size is set on BlockDataset
            shuffle=False,  # shuffling is implemented on BlockDataset
            **self.train_loader_kwargs,
        )
        return train_block_loader

    def val_dataloader(self):
        valid_record_dataset = RecordDataset(
            record_dict=self.valid_record_dict,
            record_numericalizer=self.record_numericalizer,
            batch_size=self.eval_batch_size,
        )
        valid_record_loader = torch.utils.data.DataLoader(
            valid_record_dataset,
            batch_size=None,  # batch size is set on RecordDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return valid_record_loader

    def test_dataloader(self):
        test_record_dataset = RecordDataset(
            record_dict=self.test_record_dict,
            record_numericalizer=self.record_numericalizer,
            batch_size=self.eval_batch_size,
        )
        test_record_loader = torch.utils.data.DataLoader(
            test_record_dataset,
            batch_size=None,  # batch size is set on RecordDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return test_record_loader


class LinkageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_record_dict,
        valid_record_dict,
        test_record_dict,
        train_pos_pair_set,
        valid_pos_pair_set,
        test_pos_pair_set,
        source_field,
        left_source,
        record_numericalizer,
        batch_size,
        eval_batch_size,
        train_loader_kwargs=None,
        eval_loader_kwargs=None,
        random_seed=42,
        check_for_common_records=True,
    ):
        super().__init__()

        if check_for_common_records:
            _check_for_common_records(
                train_record_dict,
                valid_record_dict,
                test_record_dict,
            )

        self.record_numericalizer = record_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_loader_kwargs = build_loader_kwargs(train_loader_kwargs)
        self.eval_loader_kwargs = build_loader_kwargs(eval_loader_kwargs)
        self.random_seed = random_seed
        self.train_record_dict = train_record_dict
        self.valid_record_dict = valid_record_dict
        self.test_record_dict = test_record_dict
        self.source_field = source_field
        self.left_source = left_source

        self._train_pos_pair_set = train_pos_pair_set
        self._valid_pos_pair_set = valid_pos_pair_set
        self._test_pos_pair_set = test_pos_pair_set

    def _set_pair_sets(self, stage):
        if stage == "fit":
            self.train_pos_pair_set = self._train_pos_pair_set
            self.valid_pos_pair_set = self._valid_pos_pair_set
        elif stage == "test":
            self.test_pos_pair_set = self._test_pos_pair_set

    def setup(self, stage=None):
        self._set_pair_sets(stage)

        if stage == "fit":
            logger.info("Train positive pair count: %s", len(self.train_pos_pair_set))
            logger.info("Valid positive pair count: %s", len(self.valid_pos_pair_set))
            logger.info("Valid negative pair count: %s", len(self.valid_pos_pair_set))
        elif stage == "test":
            logger.info("Test positive pair count: %s", len(self.test_pos_pair_set))
            logger.info("Test negative pair count: %s", len(self.test_pos_pair_set))

    def train_dataloader(self):
        train_block_dataset = BlockDataset(
            record_dict=self.train_record_dict,
            pos_pair_set=self.train_pos_pair_set,
            record_numericalizer=self.record_numericalizer,
            batch_size=self.batch_size,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving learning:
            random_seed=(
                self.random_seed + self.trainer.current_epoch if self.trainer else self.random_seed
            ),
        )
        train_block_loader = torch.utils.data.DataLoader(
            train_block_dataset,
            batch_size=None,  # batch size is set on BlockDataset
            shuffle=False,  # shuffling is implemented on BlockDataset
            **self.train_loader_kwargs,
        )
        return train_block_loader

    def val_dataloader(self):
        valid_record_dataset = RecordDataset(
            record_dict=self.valid_record_dict,
            record_numericalizer=self.record_numericalizer,
            batch_size=self.eval_batch_size,
        )
        valid_record_loader = torch.utils.data.DataLoader(
            valid_record_dataset,
            batch_size=None,  # batch size is set on RecordDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return valid_record_loader

    def test_dataloader(self):
        test_record_dataset = RecordDataset(
            record_dict=self.test_record_dict,
            record_numericalizer=self.record_numericalizer,
            batch_size=self.eval_batch_size,
        )
        test_record_loader = torch.utils.data.DataLoader(
            test_record_dataset,
            batch_size=None,  # batch size is set on RecordDataset
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        return test_record_loader
