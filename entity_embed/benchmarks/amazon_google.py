from typing import List

from .base import DeepmatcherBenchmark


class AmazonGoogleBenchmark(DeepmatcherBenchmark):
    base_url: str = (
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/"
    )
    file_name: str = "amazon_google_exp_data.zip"
    base_csv_path: str = "."
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "Amazon-Google"
