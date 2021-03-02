from typing import List

from .base import DeepmatcherBenchmark


class ITunesAmazonStructuredBenchmark(DeepmatcherBenchmark):
    base_url: str = (
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/"
    )
    file_name: str = "itunes_amazon_exp_data.zip"
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "iTunes-Amazon-Structured"
