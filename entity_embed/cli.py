import csv
import json
import logging
import os
import random

import click
import numpy as np
import torch

from . import DeduplicationDataModule, EntityEmbed, LinkageDataModule, LinkageEmbed
from .data_utils.field_config_parser import FieldConfigDictParser

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
    random_seed = kwargs["random_seed"]
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def _build_record_dict(csv_filepath, kwargs):
    csv_encoding = kwargs["csv_encoding"]
    cluster_field = kwargs.get("cluster_field")
    record_dict = {}

    with open(csv_filepath, "r", newline="", encoding=csv_encoding) as record_dict_csv_file:
        for record in csv.DictReader(record_dict_csv_file):
            if cluster_field in record:
                # force cluster_field to be an int, if there's a cluster_field
                record[cluster_field] = int(record[cluster_field])
            # force id field to be an int
            record["id"] = int(record["id"])
            record_dict[record["id"]] = record

    logger.info(f"Finished reading {csv_filepath}")
    return record_dict


def _build_record_numericalizer(record_list, kwargs):
    field_config_json = kwargs["field_config_json"]

    with open(field_config_json, "r") as field_config_json_file:
        record_numericalizer = FieldConfigDictParser.from_json(
            field_config_json_file, record_list=record_list
        )

    logger.info(f"Finished reading {field_config_json}")
    return record_numericalizer


def _is_record_linkage(kwargs):
    left_source = kwargs.get("left_source")
    source_field = kwargs.get("source_field")
    if (left_source and not source_field) or (not left_source and source_field):
        raise KeyError(
            'You must provide BOTH "source_field" and "left_source" to perform Record Linkage. '
            "Either remove both or provide both."
        )
    else:
        return bool(left_source)


def _build_datamodule(
    train_record_dict, valid_record_dict, test_record_dict, record_numericalizer, kwargs
):
    datamodule_args = {
        "train_record_dict": train_record_dict,
        "valid_record_dict": valid_record_dict,
        "test_record_dict": test_record_dict,
        "cluster_field": kwargs["cluster_field"],
        "record_numericalizer": record_numericalizer,
        "batch_size": kwargs["batch_size"],
        "eval_batch_size": kwargs["eval_batch_size"],
    }

    if _is_record_linkage(kwargs):
        datamodule_cls = LinkageDataModule
        datamodule_args.update(
            {
                "source_field": kwargs["source_field"],
                "left_source": kwargs["left_source"],
            }
        )
    else:
        datamodule_cls = DeduplicationDataModule

    if kwargs.get("num_workers") or kwargs.get("multiprocessing_context"):
        for k in ["train_loader_kwargs", "eval_loader_kwargs"]:
            datamodule_args[k] = {}
            for inner_k in ["num_workers", "multiprocessing_context"]:
                if kwargs[inner_k]:
                    datamodule_args[k][inner_k] = kwargs[inner_k]

    if kwargs.get("random_seed"):
        datamodule_args["random_seed"] = kwargs["random_seed"]

    logger.info("Building datamodule...")
    return datamodule_cls(**datamodule_args)


def _build_model(record_numericalizer, kwargs):
    model_args = {"record_numericalizer": record_numericalizer}

    if _is_record_linkage(kwargs):
        model_cls = LinkageEmbed
        model_args.update(
            {
                "source_field": kwargs["source_field"],
                "left_source": kwargs["left_source"],
            }
        )
    else:
        model_cls = EntityEmbed

    if kwargs["embedding_size"]:
        model_args["embedding_size"] = kwargs["embedding_size"]

    if kwargs["lr"]:
        model_args["learning_rate"] = kwargs["lr"]

    if kwargs["ann_k"]:
        model_args["ann_k"] = kwargs["ann_k"]

    if kwargs["sim_threshold"]:
        model_args["sim_threshold_list"] = kwargs["sim_threshold"]

    model_args["index_build_kwargs"] = {}
    for k in ["m", "max_m0", "ef_construction", "n_threads"]:
        if kwargs[k]:
            model_args["index_build_kwargs"][k] = kwargs[k]

    model_args["index_search_kwargs"] = {}
    for k in ["ef_search", "n_threads"]:
        if kwargs[k]:
            model_args["index_search_kwargs"][k] = kwargs[k]

    logger.info("Building model...")

    return model_cls(**model_args)


def _fit_model(model, datamodule, kwargs):
    monitor = kwargs["early_stop_monitor"]
    mode = kwargs["early_stop_mode"] or ("min" if "pair_entity_ratio_at" in monitor else "max")
    return model.fit(
        datamodule,
        min_epochs=kwargs["min_epochs"],
        max_epochs=kwargs["max_epochs"],
        check_val_every_n_epoch=kwargs["check_val_every_n_epoch"],
        early_stop_monitor=monitor,
        early_stop_min_delta=kwargs["early_stop_min_delta"],
        early_stop_patience=kwargs["early_stop_patience"],
        early_stop_mode=mode,
        early_stop_verbose=True,
        model_save_top_k=1,
        model_save_dir=kwargs["model_save_dir"],
        model_save_verbose=True,
        tb_save_dir=kwargs["tb_save_dir"],
        tb_name=kwargs["tb_name"],
        use_gpu=kwargs["use_gpu"],
    )


@click.command()
@click.option(
    "--field_config_json",
    type=str,
    required=True,
    help="Path of the JSON configuration file "
    "that defines how fields will be processed by the neural network",
)
@click.option(
    "--train_csv",
    type=str,
    required=True,
    help="Path of the TRAIN input dataset CSV file",
)
@click.option(
    "--valid_csv",
    type=str,
    required=True,
    help="Path of the VALID input dataset CSV file",
)
@click.option(
    "--test_csv",
    type=str,
    required=True,
    help="Path of the TEST input dataset CSV file",
)
@click.option(
    "--unlabeled_csv",
    type=str,
    required=True,
    help="Path of the UNLABELED input dataset CSV file",
)
@click.option(
    "--csv_encoding", type=str, default="utf-8", help="Encoding of the input dataset CSV file"
)
@click.option(
    "--cluster_field",
    type=str,
    required=True,
    help="Column of the CSV dataset that contains the true cluster assignment. "
    "Equivalent to the label in tabular classification. "
    "Files train_csv, valid_csv, test_csv MUST HAVE cluster_field column. File test_csv MUST NOT.",
)
@click.option(
    "--source_field",
    type=str,
    help="Set this when doing Record Linkage. "
    "Column of the CSV dataset that contains the indication of the left or right source "
    "for Record Linkage",
)
@click.option(
    "--left_source",
    type=str,
    help="Set this when doing Record Linkage. "
    "Consider any record with this value in the `source_field` as the `left_source` dataset. "
    "The records with other `source_field` values are considered the right dataset",
)
@click.option(
    "--embedding_size", type=int, default=300, help="Embedding Dimensionality, for example: 300"
)
@click.option("--lr", type=float, default=0.001, help="Learning Rate for training")
@click.option("--min_epochs", type=int, default=5, help="Min number of epochs to run")
@click.option("--max_epochs", type=int, default=100, help="Max number of epochs to run")
@click.option(
    "--early_stop_monitor",
    type=str,
    default="valid_recall_at_0.3",
    help="Metric to be monitored for early stoping. E.g. `valid_recall_at_0.3`. "
    "The float on `at_X` must be one of `sim_threshold`",
)
@click.option(
    "--early_stop_min_delta",
    type=float,
    default=0.0,
    help="Minimum change in the monitored metric to qualify as an improvement",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=20,
    help="Number of validation runs with no improvement after which training will be stopped",
)
@click.option(
    "--early_stop_mode",
    type=str,
    default="max",
    help="Mode for early stopping. Values are `max` or `min`. "
    "Based on `early_stop_monitor` metric",
)
@click.option("--tb_save_dir", type=str, help="TensorBoard save directory")
@click.option("--tb_name", type=str, help="TensorBoard experiment name")
@click.option(
    "--check_val_every_n_epoch",
    type=int,
    default=1,
    help="Run validation every N epochs.",
)
@click.option("--batch_size", type=int, required=True, help="Training batch size, in CLUSTERS")
@click.option(
    "--eval_batch_size", type=int, required=True, help="Evaluation batch size, in RECORDS"
)
@click.option(
    "--num_workers",
    type=int,
    default=-1,
    help="Number of workers to use in PyTorch Lightning datamodules "
    "and also number of threads to use in ANN. Set -1 to use all available CPUs",
)
@click.option(
    "--multiprocessing_context",
    type=str,
    default="fork",
    help="Context name for multiprocessing for PyTorch Lightning datamodules, "
    "like `spawn`, `fork`, `forkserver` (currently only tested with `fork`)",
)
@click.option(
    "--sim_threshold",
    type=float,
    multiple=True,
    help="Cosine similarity thresholds to use when computing validation and testing metrics. "
    "For each of these thresholds, validation and testing metrics "
    "(precision, recall, etc.) are computed, "
    "but ignoring any ANN neighbors with cosine similarity BELOW the threshold",
)
@click.option(
    "--ann_k",
    type=int,
    help="When finding duplicates, use this number as the K for the Approximate Nearest Neighbors",
)
@click.option("--m", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option("--max_m0", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option(
    "--ef_construction", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io"
)
@click.option("--ef_search", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option(
    "--random_seed", type=int, default=42, help="Random seed to help with reproducibility"
)
@click.option(
    "--model_save_dir",
    type=str,
    required=True,
    help="Directory path where to save the best validation model checkpoint"
    " using PyTorch Lightning",
)
@click.option("--use_gpu", type=bool, default=True, help="Use GPU for training")
def train(**kwargs):
    """
    Transform entities like companies, products, etc. into vectors
    to support scalable Record Linkage / Entity Resolution
    using Approximate Nearest Neighbors.
    """
    _fix_workers_kwargs(kwargs)
    _set_random_seeds(kwargs)
    train_record_dict = _build_record_dict(csv_filepath=kwargs["train_csv"], kwargs=kwargs)
    valid_record_dict = _build_record_dict(csv_filepath=kwargs["valid_csv"], kwargs=kwargs)
    test_record_dict = _build_record_dict(csv_filepath=kwargs["test_csv"], kwargs=kwargs)
    unlabeled_record_dict = _build_record_dict(csv_filepath=kwargs["unlabeled_csv"], kwargs=kwargs)
    record_list_all = [
        *train_record_dict.values(),
        *valid_record_dict.values(),
        *test_record_dict.values(),
        *unlabeled_record_dict.values(),
    ]
    record_numericalizer = _build_record_numericalizer(record_list=record_list_all, kwargs=kwargs)
    del record_list_all, unlabeled_record_dict
    datamodule = _build_datamodule(
        train_record_dict=train_record_dict,
        valid_record_dict=valid_record_dict,
        test_record_dict=test_record_dict,
        record_numericalizer=record_numericalizer,
        kwargs=kwargs,
    )
    model = _build_model(record_numericalizer=record_numericalizer, kwargs=kwargs)
    trainer = _fit_model(model, datamodule, kwargs)
    logger.info("Validating best model:")
    valid_metrics = model.validate(datamodule)
    logger.info(valid_metrics)
    logger.info("Testing best model:")
    test_metrics = model.test(datamodule)
    logger.info(test_metrics)
    logger.info("Saved best model at path:")
    logger.info(trainer.checkpoint_callback.best_model_path)

    return 0


def _load_model(kwargs):
    if _is_record_linkage(kwargs):
        model_cls = LinkageEmbed
    else:
        model_cls = EntityEmbed

    model = model_cls.load_from_checkpoint(kwargs["model_save_filepath"], datamodule=None)
    if kwargs["use_gpu"]:
        model = model.to(torch.device("cuda"))
    else:
        model = model.to(torch.device("cpu"))
    return model


def _predict_pairs(record_dict, model, kwargs):
    eval_batch_size = kwargs["eval_batch_size"]
    num_workers = kwargs["num_workers"]
    multiprocessing_context = kwargs["multiprocessing_context"]
    ann_k = kwargs["ann_k"]
    sim_threshold = kwargs["sim_threshold"]

    index_build_kwargs = {}
    for k in ["m", "max_m0", "ef_construction", "n_threads"]:
        if kwargs[k]:
            index_build_kwargs[k] = kwargs[k]

    index_search_kwargs = {}
    for k in ["ef_search", "n_threads"]:
        if kwargs[k]:
            index_search_kwargs[k] = kwargs[k]

    if _is_record_linkage(kwargs):
        found_pair_set = model.predict_pairs(
            record_dict=record_dict,
            batch_size=eval_batch_size,
            ann_k=ann_k,
            sim_threshold=sim_threshold,
            loader_kwargs={
                "num_workers": num_workers,
                "multiprocessing_context": multiprocessing_context,
            },
            index_build_kwargs=index_build_kwargs,
            index_search_kwargs=index_search_kwargs,
        )
    else:
        found_pair_set = model.predict_pairs(
            record_dict=record_dict,
            batch_size=eval_batch_size,
            ann_k=ann_k,
            sim_threshold=sim_threshold,
            loader_kwargs={
                "num_workers": num_workers,
                "multiprocessing_context": multiprocessing_context,
            },
            index_build_kwargs=index_build_kwargs,
            index_search_kwargs=index_search_kwargs,
        )
    return list(found_pair_set)


def _write_json(found_pairs, kwargs):
    with open(kwargs["output_json"], "w", encoding="utf-8") as f:
        json.dump(found_pairs, f, indent=4)


@click.command()
@click.option(
    "--model_save_filepath",
    type=str,
    required=True,
    help="Path where the model checkpoint was saved. Get this from entity_embed_train output",
)
@click.option(
    "--unlabeled_csv",
    type=str,
    required=True,
    help="Path of the unlabeled input dataset CSV file",
)
@click.option(
    "--csv_encoding",
    type=str,
    default="utf-8",
    help="Encoding of the input and output dataset CSV files",
)
@click.option(
    "--source_field",
    type=str,
    help="Set this when doing Record Linkage. "
    "Column of the CSV dataset that contains the indication of the left or right source "
    "for Record Linkage",
)
@click.option(
    "--left_source",
    type=str,
    help="Set this when doing Record Linkage. "
    "Consider any record with this value in the `source_field` as the `left_source` dataset. "
    "The records with other `source_field` values are considered the right dataset",
)
@click.option(
    "--eval_batch_size", type=int, required=True, help="Evaluation batch size, in RECORDS"
)
@click.option(
    "--num_workers",
    type=int,
    default=-1,
    help="Number of workers to use in PyTorch Lightning datamodules "
    "and also number of threads to use in ANN. Set -1 to use all available CPUs",
)
@click.option(
    "--multiprocessing_context",
    type=str,
    default="fork",
    help="Context name for multiprocessing for PyTorch Lightning datamodules, "
    "like `spawn`, `fork`, `forkserver` (currently only tested with `fork`)",
)
@click.option(
    "--sim_threshold",
    type=float,
    required=True,
    help="A SINGLE cosine similarity threshold to use when finding duplicates. "
    "Any ANN neighbors with cosine similarity BELOW this threshold is ignored",
)
@click.option(
    "--ann_k",
    type=int,
    default=100,
    help="When finding duplicates, use this number as the K for the Approximate Nearest Neighbors",
)
@click.option("--m", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option("--max_m0", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option(
    "--ef_construction", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io"
)
@click.option("--ef_search", type=int, help="Parameter for the ANN. See N2 docs: n2.readthedocs.io")
@click.option(
    "--random_seed", type=int, default=42, help="Random seed to help with reproducibility"
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path of the output JSON file that will contain the candidate duplicate pairs. "
    "Remember Entity Embed is focused on recall. "
    "You must use some classifier to filter these and find the best matching pairs.",
)
@click.option("--use_gpu", type=bool, default=True, help="Use GPU for predicting pairs")
def predict(**kwargs):
    _fix_workers_kwargs(kwargs)
    _set_random_seeds(kwargs)
    model = _load_model(kwargs)
    record_dict = _build_record_dict(
        csv_filepath=kwargs["unlabeled_csv"],
        kwargs=kwargs,
    )
    found_pairs = _predict_pairs(record_dict=record_dict, model=model, kwargs=kwargs)
    _write_json(found_pairs=found_pairs, kwargs=kwargs)

    logger.info(f"Found {len(found_pairs)} candidate pairs, writing to JSON file at:")
    logger.info(kwargs["output_json"])

    return 0
