from typing import List

from .base import DeepmatcherBenchmark


class DBLPScholarStructuredBenchmark(DeepmatcherBenchmark):
    base_url: str = (
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/"
    )
    file_name: str = "dblp_scholar_exp_data.zip"
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "DBLP-Scholar-Structured"
