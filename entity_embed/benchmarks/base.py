import csv
import logging
import os
import urllib.request
import zipfile
from abc import ABC
from typing import List
from urllib.error import HTTPError

from ..data_utils import utils
from ..entity_embed import LinkageDataModule

logger = logging.getLogger(__name__)


class DeepmatcherBenchmark(ABC):
    base_url: str
    file_name: str
    base_csv_path: str = "exp_data"
    table_csv_paths: List[str]
    train_csv_path: str = "train.csv"
    valid_csv_path: str = "valid.csv"
    test_csv_path: str = "test.csv"
    csv_encoding: str = "utf-8"
    dataset_name: str

    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.cache_dir_name = self.dataset_name

        self._download()
        self._extract_zip()
        self.id_enumerator = utils.Enumerator()
        self.row_dict, self.left_id_set, self.right_id_set = self._read_row_dict_and_id_sets()
        self.train_true_pair_set, self.train_false_pair_set = self._read_pair_sets(
            pair_csv_path=self.train_csv_path
        )
        self.valid_true_pair_set, self.valid_false_pair_set = self._read_pair_sets(
            pair_csv_path=self.valid_csv_path
        )
        self.test_true_pair_set, self.test_false_pair_set = self._read_pair_sets(
            pair_csv_path=self.test_csv_path
        )

    @property
    def local_dir_path(self) -> str:
        return os.path.join(self.data_dir_path, self.cache_dir_name)

    @property
    def local_file_path(self) -> str:
        return os.path.join(self.local_dir_path, self.file_name)

    @property
    def url(self) -> str:
        return os.path.join(self.base_url, self.file_name)

    def _download_from_url(self):
        logging.info(f"Downloading {self.url}")
        try:
            os.makedirs(self.local_dir_path)
            urllib.request.urlretrieve(self.url, self.local_file_path)
        except HTTPError as err:
            raise RuntimeError(f"Failed download from {self.url}") from err

    def _download(self):
        if os.path.isfile(self.local_file_path):  # if exists, ignore download
            return
        self._download_from_url()

    def _extract_zip(self):
        logging.info(f"Extracting {self.dataset_name}...")
        with zipfile.ZipFile(self.local_file_path, "r") as zf:
            zf.extractall(self.local_dir_path)

    def _read_row_dict_and_id_sets(self):
        logging.info(f"Reading {self.dataset_name} row_dict...")
        if len(self.table_csv_paths) > 2:
            raise ValueError("table_csv_paths with more than two paths not supported.")

        row_dict = {}
        left_id_set = set()
        right_id_set = set()

        for table_name, id_set, csv_path in zip(
            ["left", "right"], [left_id_set, right_id_set], self.table_csv_paths
        ):
            csv_path = os.path.join(self.local_dir_path, self.base_csv_path, csv_path)
            with open(csv_path, "r", encoding=self.csv_encoding) as f:
                for row in csv.DictReader(f):
                    row["__source"] = table_name
                    row["id"] = self.id_enumerator[f"{table_name}-{row['id']}"]
                    row_dict[row["id"]] = row
                    id_set.add(row["id"])

        return row_dict, left_id_set, right_id_set

    def _read_pair_sets(self, pair_csv_path):
        logging.info(f"Reading {self.dataset_name} {pair_csv_path}...")
        true_pair_set = set()
        false_pair_set = set()

        csv_path = os.path.join(self.local_dir_path, self.base_csv_path, pair_csv_path)
        with open(csv_path, "r", encoding=self.csv_encoding) as f:
            for row in csv.DictReader(f):
                id_left = self.id_enumerator[f'left-{(row["ltable_id"]).strip()}']
                id_right = self.id_enumerator[f'right-{(row["rtable_id"]).strip()}']

                if int(row["label"]) == 1:
                    true_pair_set.add((id_left, id_right))
                else:
                    false_pair_set.add((id_left, id_right))

        return true_pair_set, false_pair_set

    def build_datamodule(self, row_numericalizer, batch_size, eval_batch_size, random_seed):
        return LinkageDataModule(
            row_dict=self.row_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
            row_numericalizer=row_numericalizer,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            train_true_pair_set=self.train_true_pair_set,
            valid_true_pair_set=self.valid_true_pair_set,
            test_true_pair_set=self.test_true_pair_set,
            random_seed=random_seed,
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}> from {self.url}"
