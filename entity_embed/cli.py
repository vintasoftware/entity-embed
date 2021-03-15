import csv
import logging
import sys

import click
import pytorch_lightning as pl
from entity_embed import validate_best
from entity_embed.data_utils.helpers import AttrInfoDictParser
from entity_embed.data_utils.utils import Enumerator
from entity_embed.entity_embed import DeduplicationDataModule, EntityEmbed, LinkageDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_datamodule(parser_args_dict):
    id_enumerator = Enumerator()
    row_dict = {}

    csv_filepath = parser_args_dict["csv_filepath"]
    encoding = parser_args_dict["csv_encoding"]
    with open(csv_filepath, "r", encoding=encoding) as row_dict_csv_file:
        for row in csv.DictReader(row_dict_csv_file):
            row["id"] = id_enumerator[row["id"]]
            row_dict[row["id"]] = row

    logger.info(f"Finished reading {csv_filepath}")

    attr_info_json_filepath = parser_args_dict["attr_info_json_filepath"]
    with open(attr_info_json_filepath, "r") as attr_info_json_file:
        row_numericalizer = AttrInfoDictParser.from_json(attr_info_json_file, row_dict=row_dict)

    logger.info(f"Finished reading {attr_info_json_filepath}")

    datamodule_args = {
        "row_dict": row_dict,
        "cluster_attr": parser_args_dict["cluster_attr"],
        "row_numericalizer": row_numericalizer,
        "batch_size": parser_args_dict["batch_size"],
        "eval_batch_size": parser_args_dict["eval_batch_size"],
        "train_cluster_len": parser_args_dict["train_len"],
        "valid_cluster_len": parser_args_dict["valid_len"],
        "test_cluster_len": parser_args_dict["test_len"],
        "only_plural_clusters": parser_args_dict["only_plural_clusters"],
    }

    if parser_args_dict.get("left"):
        left_id_set = set()
        right_id_set = set()
        for id, row in row_dict.items():
            try:
                if row["__source"] == parser_args_dict["left"]:
                    left_id_set.add(id)
                else:
                    right_id_set.add(id)
            except KeyError:
                raise KeyError(
                    f'You must provide a "__source" column on {csv_filepath} '
                    "in order to determine left_id_set and right_id_set on LinkageDataModule"
                )
        datamodule_args["left_id_set"] = left_id_set
        datamodule_args["right_id_set"] = right_id_set
        datamodule_cls = LinkageDataModule
    else:
        datamodule_cls = DeduplicationDataModule

    if parser_args_dict.get("num_workers") or parser_args_dict.get("multiprocessing_context"):
        for k in ["pair_loader_kwargs", "row_loader_kwargs"]:
            datamodule_args[k] = {}
            for inner_k in ["num_workers", "multiprocessing_context"]:
                if parser_args_dict[inner_k]:
                    datamodule_args[k][inner_k] = parser_args_dict[inner_k]

    if parser_args_dict.get("random_seed"):
        datamodule_args["random_seed"] = parser_args_dict["random_seed"]

    logger.info("Building datamodule...")

    return datamodule_cls(**datamodule_args)


def _build_model(datamodule, parser_args_dict):
    model_args = {
        "datamodule": datamodule,
    }

    if parser_args_dict["embedding_size"]:
        model_args["embedding_size"] = parser_args_dict["embedding_size"]

    if parser_args_dict["lr"]:
        model_args["learning_rate"] = parser_args_dict["lr"]

    if parser_args_dict["ann_k"]:
        model_args["ann_k"] = parser_args_dict["ann_k"]

    if parser_args_dict["sim_threshold_list"]:
        model_args["sim_threshold_list"] = parser_args_dict["sim_threshold_list"]

    if (
        parser_args_dict["m"]
        or parser_args_dict["max_m0"]
        or parser_args_dict["ef_construction"]
        or parser_args_dict["n_threads"]
    ):
        model_args["index_build_kwargs"] = {}
        for k in ["m", "max_m0", "ef_construction", "n_threads"]:
            if parser_args_dict[k]:
                model_args["index_build_kwargs"][k] = parser_args_dict[k]

    if parser_args_dict["ef_search"] or parser_args_dict["n_threads"]:
        model_args["index_search_kwargs"] = {}
        for k in ["ef_search", "n_threads"]:
            if parser_args_dict[k]:
                model_args["index_search_kwargs"][k] = parser_args_dict[k]

    logger.info("Building model...")

    return EntityEmbed(**model_args)


def _build_trainer(parser_args_dict):
    monitor = parser_args_dict["early_stopping_monitor"]
    min_delta = parser_args_dict["early_stopping_min_delta"]
    patience = parser_args_dict["early_stopping_patience"]
    mode = parser_args_dict["early_stopping_mode"] or (
        "min" if "pair_entity_ratio_at" in monitor else "max"
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode=mode,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        save_top_k=1,
        verbose=True,
        filename=parser_args_dict["model_save_filepath"],
    )

    trainer_args = {
        "gpus": parser_args_dict["gpus"],
        "max_epochs": parser_args_dict["max_epochs"],
        "check_val_every_n_epoch": parser_args_dict["check_val_every_n_epoch"],
        "callbacks": [early_stop_callback, checkpoint_callback],
    }

    if parser_args_dict["tb_name"] and parser_args_dict["tb_save_dir"]:
        trainer_args["logger"] = TensorBoardLogger(
            parser_args_dict["tb_save_dir"],
            name=parser_args_dict["tb_name"],
        )
    elif parser_args_dict["tb_name"] or parser_args_dict["tb_save_dir"]:
        raise KeyError(
            'Please provide both "tb_name" and "tb_save_dir" to enable '
            "TensorBoardLogger or omit both to disable it"
        )

    return pl.Trainer(**trainer_args)


@click.command()
@click.option(
    "-model_save_filepath",
    type=str,
    help="Path where to save the best validation model using PyTorch Lightning",
)
@click.option("-tb_save_dir", type=str, help="TensorBoard save directory")
@click.option("-tb_name", type=str, help="TensorBoard experiment name")
@click.option(
    "-check_val_every_n_epoch",
    type=int,
    default=1,
    help="Run validation every N epochs.",
)
@click.option("-max_epochs", type=int, required=True, help="Max number of epochs to run")
@click.option(
    "-gpus", type=int, default=1, help="Number of GPUs to use (currently only tested with 1)"
)
@click.option(
    "-early_stopping_mode",
    type=str,
    help="Mode for early stopping. Values are `max` or `min`. "
    "Based on `early_stopping_monitor` metric",
)
@click.option(
    "-early_stopping_patience",
    type=int,
    required=True,
    help="Number of validation runs with no improvement after which training will be stopped",
)
@click.option(
    "-early_stopping_min_delta",
    type=float,
    required=True,
    help="Minimum change in the monitored metric to qualify as an improvement",
)
@click.option(
    "-early_stopping_monitor",
    type=str,
    required=True,
    help="Metric to be monitored for early stoping. E.g. `valid_recall_at_0.3`. "
    "The float on `at_X` must be one of `sim_threshold_list`",
)
@click.option(
    "-ann_k",
    type=int,
    help="When finding duplicates, use this number as the K for the Approximate Nearest Neighbors",
)
@click.option("-ef_search", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option(
    "-ef_construction", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io"
)
@click.option("-max_m0", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option("-m", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option(
    "-multiprocessing_context",
    type=str,
    help="Context name for multiprocessing for PyTorch Lightning datamodules, "
    "like `spawn`, `fork`, `forkserver` (currently only tested with `fork`)",
)
@click.option(
    "-num_workers",
    type=int,
    help="Number of workers to use in PyTorch Lightning datamodules "
    "and also number of threads to use in ANN",
)
@click.option(
    "-sim_threshold",
    "--sim_threshold_list",
    type=float,
    multiple=True,
    help="Cosine Similarity thresholds to use when computing validation and testing metrics. "
    "For each of these thresholds, validation and testing metrics "
    "(precision, recall, etc.) are computed, "
    "but ignoring any ANN neighbors with cosine similarity BELOW the threshold",
)
@click.option("-lr", type=str, help="Learning Rate for training")
@click.option("-embedding_size", type=int, help="Embedding Dimensionality, for example: 300")
@click.option(
    "-test_len",
    type=int,
    required=True,
    help="Number of CLUSTERS from the dataset to use for testing",
)
@click.option(
    "-valid_len",
    type=int,
    required=True,
    help="Number of CLUSTERS from the dataset to use for validation",
)
@click.option(
    "-train_len",
    type=int,
    required=True,
    help="Number of CLUSTERS from the dataset to use for training",
)
@click.option(
    "-only_plural_clusters",
    type=bool,
    help="Use only clusters with more than one element for training, validation, testing",
)
@click.option("-random_seed", type=int, help="Random seed to help with reproducibility")
@click.option(
    "-left",
    type=str,
    help="Set this when doing Record Linkage. "
    "Consider any row with this value in the `__source` column as the left dataset. "
    "The rows with other `__source` values are considered the right dataset",
)
@click.option("-eval_batch_size", type=int, required=True, help="Evaluation batch size, in ROWS")
@click.option("-batch_size", type=int, required=True, help="Training batch size, in CLUSTERS")
@click.option("-csv_encoding", type=str, default="utf-8", help="Encoding of the dataset CSV file")
@click.option("-csv_filepath", type=str, required=True, help="Path of the dataset CSV file")
@click.option(
    "-cluster_attr",
    type=str,
    required=True,
    help="Column of the CSV dataset that contains the true cluster assignment. "
    "Equivalent to the label in tabular classification",
)
@click.option(
    "-attr_info_json_filepath",
    type=str,
    required=True,
    help="Path of the JSON configuration file "
    "that defines how columns will be processed by the neural network",
)
def main(**kwargs):
    """
    Transform entities like companies, products, etc. into vectors
    to support scalable Record Linkage / Entity Resolution
    using Approximate Nearest Neighbors.
    """

    # Duplicate "num_workers" key into "n_threads" key since _build_datamodule
    # uses "num_workers" and _build_model uses "n_threads"
    kwargs["n_threads"] = kwargs["num_workers"]

    datamodule = _build_datamodule(kwargs)
    model = _build_model(datamodule, kwargs)

    trainer = _build_trainer(kwargs)
    trainer.fit(model, datamodule)
    validate_best(trainer)
    trainer.test(ckpt_path="best", verbose=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
