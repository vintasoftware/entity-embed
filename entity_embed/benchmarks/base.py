import csv
import logging
import os
import urllib.request
import zipfile
from abc import ABC
from typing import List
from urllib.error import HTTPError

from ..data_modules import DEFAULT_LEFT_SOURCE, DEFAULT_SOURCE_FIELD
from ..data_utils import utils

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
        self.source_field = DEFAULT_SOURCE_FIELD
        self.left_source = DEFAULT_LEFT_SOURCE
        self.right_source = "right"

        self._download()
        self._extract_zip()
        self.id_enumerator = utils.Enumerator()
        self.record_dict = self._read_record_dict()
        self.train_pos_pair_set, self.train_neg_pair_set = self._read_pair_sets(
            pair_csv_path=self.train_csv_path
        )
        self.valid_pos_pair_set, self.valid_neg_pair_set = self._read_pair_sets(
            pair_csv_path=self.valid_csv_path
        )
        self.test_pos_pair_set, self.test_neg_pair_set = self._read_pair_sets(
            pair_csv_path=self.test_csv_path
        )
        self.train_record_dict = {
            id_: self.record_dict[id_]
            for pair in self.train_pos_pair_set | self.train_neg_pair_set
            for id_ in pair
        }
        self.valid_record_dict = {
            id_: self.record_dict[id_]
            for pair in self.valid_pos_pair_set | self.valid_neg_pair_set
            for id_ in pair
        }
        self.test_record_dict = {
            id_: self.record_dict[id_]
            for pair in self.test_pos_pair_set | self.test_neg_pair_set
            for id_ in pair
        }

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

    def _read_record_dict(self):
        logging.info(f"Reading {self.dataset_name} record_dict...")
        if len(self.table_csv_paths) > 2:
            raise ValueError("table_csv_paths with more than two paths not supported.")

        record_dict = {}

        for table_name, csv_path in zip(
            [self.left_source, self.right_source], self.table_csv_paths
        ):
            csv_path = os.path.join(self.local_dir_path, self.base_csv_path, csv_path)
            with open(csv_path, "r", encoding=self.csv_encoding) as f:
                for record in csv.DictReader(f):
                    record[self.source_field] = table_name
                    record["id"] = self.id_enumerator[f"{table_name}-{record['id']}"]
                    record_dict[record["id"]] = record

        return record_dict

    def _read_pair_sets(self, pair_csv_path):
        logging.info(f"Reading {self.dataset_name} {pair_csv_path}...")
        pos_pair_set = set()
        neg_pair_set = set()

        csv_path = os.path.join(self.local_dir_path, self.base_csv_path, pair_csv_path)
        with open(csv_path, "r", encoding=self.csv_encoding) as f:
            for row in csv.DictReader(f):
                id_left = self.id_enumerator[f'left-{(row["ltable_id"]).strip()}']
                id_right = self.id_enumerator[f'right-{(row["rtable_id"]).strip()}']

                if int(row["label"]) == 1:
                    pos_pair_set.add((id_left, id_right))
                else:
                    neg_pair_set.add((id_left, id_right))

        return pos_pair_set, neg_pair_set

    def __repr__(self):
        return f"<{self.__class__.__name__}> from {self.url}"
