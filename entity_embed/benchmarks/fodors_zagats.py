from typing import List

from .base import DeepmatcherBenchmark


class FodorsZagatsBenchmark(DeepmatcherBenchmark):
    base_url: str = (
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/"
    )
    file_name: str = "fodors_zagat_exp_data.zip"
    table_csv_paths: List[str] = ["tableA.csv", "tableB.csv"]
    dataset_name: str = "Fodors-Zagats"
