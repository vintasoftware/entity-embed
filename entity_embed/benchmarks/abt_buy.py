from typing import List

from .base import DeepmatcherBenchmark


class AbtBuyBenchmark(DeepmatcherBenchmark):
    base_url: str = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/"
    file_name: str = "abt_buy_exp_data.zip"
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "Abt-Buy"
