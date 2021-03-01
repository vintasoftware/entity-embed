from typing import List

from .base import DeepmatcherBenchmark


class CompanyBenchmark(DeepmatcherBenchmark):
    base_url: str = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/"
    file_name: str = "company_exp_data.zip"
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "Company"
