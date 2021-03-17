import csv
import logging
import os

import click
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from . import ANNEntityIndex, ANNLinkageIndex, validate_best
from .data_utils import utils
from .data_utils.helpers import AttrInfoDictParser
from .entity_embed import DeduplicationDataModule, EntityEmbed, LinkageDataModule, LinkageEmbed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fix_workers_kwargs(kwargs):
    # Accept -1 as "num_workers"
    if kwargs["num_workers"] == -1:
        kwargs["num_workers"] = os.cpu_count()
    # Duplicate "num_workers" key into "n_threads" key since _build_datamodule
    # uses "num_workers" and _build_model uses "n_threads"
    kwargs["n_threads"] = kwargs["num_workers"]


def _set_random_seeds(kwargs):
    if kwargs.get("random_seed") is not None:
        random_seed = kwargs["random_seed"]
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


def _build_row_dict(csv_filepath, kwargs):
    csv_encoding = kwargs["csv_encoding"]
    row_dict = {}

    with open(csv_filepath, "r", newline="", encoding=csv_encoding) as row_dict_csv_file:
        for id_, row in enumerate(csv.DictReader(row_dict_csv_file)):
            row_dict[id_] = row

    logger.info(f"Finished reading {csv_filepath}")
    return row_dict


def _build_row_numericalizer(row_list, kwargs):
    attr_info_json_filepath = kwargs["attr_info_json_filepath"]

    with open(attr_info_json_filepath, "r") as attr_info_json_file:
        row_numericalizer = AttrInfoDictParser.from_json(attr_info_json_file, row_list=row_list)

    logger.info(f"Finished reading {attr_info_json_filepath}")
    return row_numericalizer


def _build_left_right_id_sets(row_dict, source_attr, left_source):
    left_id_set = set()
    right_id_set = set()
    for id, row in row_dict.items():
        if row[source_attr] == left_source:
            left_id_set.add(id)
        else:
            right_id_set.add(id)
    return left_id_set, right_id_set


def _build_datamodule(row_dict, row_numericalizer, kwargs):
    left_source = kwargs.get("left_source")
    if left_source:  # is record linkage
        try:
            source_attr = kwargs["source_attr"]
        except KeyError as e:
            raise KeyError('You must provide a "source_attr" to perform Record Linkage') from e

    datamodule_args = {
        "row_dict": row_dict,
        "cluster_attr": kwargs["cluster_attr"],
        "row_numericalizer": row_numericalizer,
        "batch_size": kwargs["batch_size"],
        "eval_batch_size": kwargs["eval_batch_size"],
        "train_cluster_len": kwargs["train_len"],
        "valid_cluster_len": kwargs["valid_len"],
        "test_cluster_len": kwargs["test_len"],
        "only_plural_clusters": kwargs["only_plural_clusters"],
    }

    if left_source:  # is record linkage
        left_id_set, right_id_set = _build_left_right_id_sets(row_dict, source_attr, left_source)
        datamodule_args["left_id_set"] = left_id_set
        datamodule_args["right_id_set"] = right_id_set
        datamodule_cls = LinkageDataModule
    else:
        datamodule_cls = DeduplicationDataModule

    if kwargs.get("num_workers") or kwargs.get("multiprocessing_context"):
        for k in ["pair_loader_kwargs", "row_loader_kwargs"]:
            datamodule_args[k] = {}
            for inner_k in ["num_workers", "multiprocessing_context"]:
                if kwargs[inner_k]:
                    datamodule_args[k][inner_k] = kwargs[inner_k]

    if kwargs.get("random_seed"):
        datamodule_args["random_seed"] = kwargs["random_seed"]

    logger.info("Building datamodule...")

    return datamodule_cls(**datamodule_args)


def _build_model(row_numericalizer, kwargs):
    model_args = {"row_numericalizer": row_numericalizer, "eval_with_clusters": True}

    if kwargs["embedding_size"]:
        model_args["embedding_size"] = kwargs["embedding_size"]

    if kwargs["lr"]:
        model_args["learning_rate"] = kwargs["lr"]

    if kwargs["ann_k"]:
        model_args["ann_k"] = kwargs["ann_k"]

    if kwargs["sim_threshold_list"]:
        model_args["sim_threshold_list"] = kwargs["sim_threshold_list"]

    model_args["index_build_kwargs"] = {}
    for k in ["m", "max_m0", "ef_construction", "n_threads"]:
        if kwargs[k]:
            model_args["index_build_kwargs"][k] = kwargs[k]

    model_args["index_search_kwargs"] = {}
    for k in ["ef_search", "n_threads"]:
        if kwargs[k]:
            model_args["index_search_kwargs"][k] = kwargs[k]

    logger.info("Building model...")

    return EntityEmbed(**model_args)


def _build_trainer(kwargs):
    monitor = kwargs["early_stopping_monitor"]
    min_delta = kwargs["early_stopping_min_delta"]
    patience = kwargs["early_stopping_patience"]
    mode = kwargs["early_stopping_mode"] or ("min" if "pair_entity_ratio_at" in monitor else "max")

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
        mode=mode,
        verbose=True,
        dirpath=kwargs["model_save_dirpath"],
    )

    trainer_args = {
        "gpus": kwargs["gpus"],
        "max_epochs": kwargs["max_epochs"],
        "check_val_every_n_epoch": kwargs["check_val_every_n_epoch"],
        "callbacks": [early_stop_callback, checkpoint_callback],
    }

    if kwargs["tb_name"] and kwargs["tb_save_dir"]:
        trainer_args["logger"] = TensorBoardLogger(
            kwargs["tb_save_dir"],
            name=kwargs["tb_name"],
        )
    elif kwargs["tb_name"] or kwargs["tb_save_dir"]:
        raise KeyError(
            'Please provide both "tb_name" and "tb_save_dir" to enable '
            "TensorBoardLogger or omit both to disable it"
        )

    return pl.Trainer(**trainer_args)


@click.command()
@click.option(
    "-model_save_dirpath",
    type=str,
    help="Directory path where to save the best validation model checkpoint"
    " using PyTorch Lightning",
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
    default="fork",
    help="Context name for multiprocessing for PyTorch Lightning datamodules, "
    "like `spawn`, `fork`, `forkserver` (currently only tested with `fork`)",
)
@click.option(
    "-num_workers",
    type=int,
    help="Number of workers to use in PyTorch Lightning datamodules "
    "and also number of threads to use in ANN. Set -1 to use all available CPUs",
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
@click.option("-lr", type=float, help="Learning Rate for training")
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
    "-source_attr",
    type=str,
    help="Set this when doing Record Linkage. "
    "Column of the CSV dataset that contains the indication of the left or right source "
    "for Record Linkage",
)
@click.option(
    "-left_source",
    type=str,
    help="Set this when doing Record Linkage. "
    "Consider any row with this value in the `source_attr` column as the left_source dataset. "
    "The rows with other `source_attr` values are considered the right dataset",
)
@click.option("-eval_batch_size", type=int, required=True, help="Evaluation batch size, in ROWS")
@click.option("-batch_size", type=int, required=True, help="Training batch size, in CLUSTERS")
@click.option(
    "-csv_encoding", type=str, default="utf-8", help="Encoding of the input dataset CSV file"
)
@click.option(
    "-labeled_input_csv_filepath",
    type=str,
    required=True,
    help="Path of the LABELED input dataset CSV file",
)
@click.option(
    "-unlabeled_input_csv_filepath",
    type=str,
    required=True,
    help="Path of the UNLABELED input dataset CSV file",
)
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
def train(**kwargs):
    """
    Transform entities like companies, products, etc. into vectors
    to support scalable Record Linkage / Entity Resolution
    using Approximate Nearest Neighbors.
    """
    _fix_workers_kwargs(kwargs)
    _set_random_seeds(kwargs)
    row_dict_labeled = _build_row_dict(
        csv_filepath=kwargs["labeled_input_csv_filepath"], kwargs=kwargs
    )
    row_dict_unlabeled = _build_row_dict(
        csv_filepath=kwargs["unlabeled_input_csv_filepath"], kwargs=kwargs
    )
    row_list_all = list(row_dict_labeled.values()) + list(row_dict_unlabeled.values())
    row_numericalizer = _build_row_numericalizer(row_list=row_list_all, kwargs=kwargs)
    del row_list_all, row_dict_unlabeled
    datamodule = _build_datamodule(
        row_dict=row_dict_labeled, row_numericalizer=row_numericalizer, kwargs=kwargs
    )
    model = _build_model(row_numericalizer=row_numericalizer, kwargs=kwargs)

    trainer = _build_trainer(kwargs)
    trainer.fit(model, datamodule)
    del model, datamodule
    valid_metrics = validate_best(trainer)
    logger.info(valid_metrics)
    test_metrics = trainer.test(ckpt_path="best", verbose=False)
    logger.info(test_metrics)

    logger.info(f"Saved best model at path {trainer.checkpoint_callback.best_model_path}")

    return 0


def _load_model(kwargs):
    is_record_linkage = "left_source" in kwargs
    if is_record_linkage:
        model_cls = LinkageEmbed
    else:
        model_cls = EntityEmbed
    return model_cls.load_from_checkpoint(kwargs["model_save_filepath"], datamodule=None)


def _assign_clusters(row_dict, model, kwargs):
    eval_batch_size = kwargs["eval_batch_size"]
    num_workers = kwargs["num_workers"]
    multiprocessing_context = kwargs["multiprocessing_context"]
    ann_k = kwargs["ann_k"]
    sim_threshold = kwargs["sim_threshold"]
    left_source = kwargs.get("left_source")
    if left_source:  # is record linkage
        try:
            source_attr = kwargs["source_attr"]
        except KeyError as e:
            raise KeyError('You must provide a "source_attr" to perform Record Linkage') from e
    cluster_attr = kwargs["cluster_attr"]

    index_build_kwargs = {}
    for k in ["m", "max_m0", "ef_construction", "n_threads"]:
        if kwargs[k]:
            index_build_kwargs[k] = kwargs[k]

    index_search_kwargs = {}
    for k in ["ef_search", "n_threads"]:
        if kwargs[k]:
            index_search_kwargs[k] = kwargs[k]

    if left_source:  # is record linkage
        left_id_set, right_id_set = _build_left_right_id_sets(row_dict, source_attr, left_source)
        cluster_mapping = _find_clusters_rl(
            row_dict=row_dict,
            left_id_set=left_id_set,
            right_id_set=right_id_set,
            model=model,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            ann_k=ann_k,
            sim_threshold=sim_threshold,
            index_build_kwargs=index_build_kwargs,
            index_search_kwargs=index_search_kwargs,
        )
    else:
        cluster_mapping = _find_clusters_er(
            row_dict=row_dict,
            model=model,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            ann_k=ann_k,
            sim_threshold=sim_threshold,
            index_build_kwargs=index_build_kwargs,
            index_search_kwargs=index_search_kwargs,
        )

    utils.assign_clusters(
        row_dict=row_dict, cluster_attr=cluster_attr, cluster_mapping=cluster_mapping
    )


def _find_clusters_rl(
    row_dict,
    left_id_set,
    right_id_set,
    model,
    eval_batch_size,
    num_workers,
    multiprocessing_context,
    ann_k,
    sim_threshold,
    index_build_kwargs,
    index_search_kwargs,
):
    left_vector_dict, right_vector_dict = model.predict(
        row_dict=row_dict,
        left_id_set=left_id_set,
        right_id_set=right_id_set,
        batch_size=eval_batch_size,
        loader_kwargs={
            "num_workers": num_workers,
            "multiprocessing_context": multiprocessing_context,
        },
        show_progress=True,
    )
    ann_index = ANNLinkageIndex(embedding_size=model.embedding_size)
    ann_index.insert_vector_dict(
        left_vector_dict=left_vector_dict, right_vector_dict=right_vector_dict
    )
    ann_index.build(index_build_kwargs=index_build_kwargs)
    cluster_mapping, __ = ann_index.search_clusters(
        k=ann_k,
        sim_threshold=sim_threshold,
        left_vector_dict=left_vector_dict,
        right_vector_dict=right_vector_dict,
        index_search_kwargs=index_search_kwargs,
    )
    return cluster_mapping


def _find_clusters_er(
    row_dict,
    model,
    eval_batch_size,
    num_workers,
    multiprocessing_context,
    ann_k,
    sim_threshold,
    index_build_kwargs,
    index_search_kwargs,
):
    vector_dict = model.predict(
        row_dict=row_dict,
        batch_size=eval_batch_size,
        loader_kwargs={
            "num_workers": num_workers,
            "multiprocessing_context": multiprocessing_context,
        },
        show_progress=True,
    )
    ann_index = ANNEntityIndex(embedding_size=model.embedding_size)
    ann_index.insert_vector_dict(vector_dict)
    ann_index.build(index_build_kwargs=index_build_kwargs)
    cluster_mapping = ann_index.search_clusters(
        k=ann_k, sim_threshold=sim_threshold, index_search_kwargs=index_search_kwargs
    )
    return cluster_mapping


def _write_csv(row_dict, kwargs):
    with open(kwargs["output_csv_filepath"], "w", newline="", encoding=kwargs["csv_encoding"]) as f:
        writer = csv.DictWriter(f, fieldnames=next(iter(row_dict.values())).keys())
        writer.writeheader()
        writer.writerows(row_dict.values())


@click.command()
@click.option(
    "-model_save_filepath",
    type=str,
    help="Path where the model checkpoint was saved",
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
    default="fork",
    help="Context name for multiprocessing for PyTorch Lightning datamodules, "
    "like `spawn`, `fork`, `forkserver` (currently only tested with `fork`)",
)
@click.option(
    "-num_workers",
    type=int,
    help="Number of workers to use in PyTorch Lightning datamodules "
    "and also number of threads to use in ANN. Set -1 to use all available CPUs",
)
@click.option(
    "-sim_threshold",
    type=float,
    multiple=False,
    help="A SINGLE Cosine Similarity threshold to use when finding duplicates. "
    "Any ANN neighbors with cosine similarity BELOW this threshold is ignored",
)
@click.option("-random_seed", type=int, help="Random seed to help with reproducibility")
@click.option(
    "-source_attr",
    type=str,
    help="Set this when doing Record Linkage. "
    "Column of the CSV dataset that contains the indication of the left or right source "
    "for Record Linkage",
)
@click.option(
    "-left_source",
    type=str,
    help="Set this when doing Record Linkage. "
    "Consider any row with this value in the `source_attr` column as the left_source dataset. "
    "The rows with other `source_attr` values are considered the right dataset",
)
@click.option("-eval_batch_size", type=int, required=True, help="Evaluation batch size, in ROWS")
@click.option(
    "-csv_encoding",
    type=str,
    default="utf-8",
    help="Encoding of the input and output dataset CSV files",
)
@click.option(
    "-unlabeled_input_csv_filepath",
    type=str,
    required=True,
    help="Path of the unlabeled input dataset CSV file",
)
@click.option(
    "-attr_info_json_filepath",
    type=str,
    required=True,
    help="Path of the JSON configuration file "
    "that defines how columns will be processed by the neural network",
)
@click.option(
    "-output_csv_filepath",
    type=str,
    required=True,
    help="Path of the output CSV file that will contain the `cluster_attr` with the found values. "
    "The CSV will be equal to the dataset CSV but with the additional `cluster_attr` column",
)
@click.option(
    "-cluster_attr",
    type=str,
    required=True,
    help="Column of the CSV dataset that will contain the cluster assignment. "
    "Equivalent to the label in tabular classification",
)
def predict(**kwargs):
    _fix_workers_kwargs(kwargs)
    _set_random_seeds(kwargs)
    model = _load_model(kwargs)
    row_dict = _build_row_dict(
        csv_filepath=kwargs["unlabeled_input_csv_filepath"],
        kwargs=kwargs,
    )
    _assign_clusters(row_dict=row_dict, model=model, kwargs=kwargs)
    _write_csv(row_dict=row_dict, kwargs=kwargs)

    logger.info(
        f"File {kwargs['output_csv_filepath']} is now labeled at column {kwargs['cluster_attr']}"
    )

    return 0
