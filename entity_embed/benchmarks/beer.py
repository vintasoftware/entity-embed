from typing import List

from .base import DeepmatcherBenchmark


class BeerBenchmark(DeepmatcherBenchmark):
    base_url: str = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/"
    file_name: str = "beer_exp_data.zip"
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "Beer"
