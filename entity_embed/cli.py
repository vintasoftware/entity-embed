import argparse
import csv
import sys

from entity_embed.data_utils.helpers import AttrInfoDictParser
from entity_embed.data_utils.utils import Enumerator
from entity_embed.entity_embed import DeduplicationDataModule, EntityEmbed, LinkageDataModule


def _build_datamodule(parser_args):
    id_enumerator = Enumerator()
    row_dict = {}

    with open(parser_args.row_dict_csv_filepath, "r", newline="") as row_dict_csv_file:
        for row in csv.DictReader(row_dict_csv_file):
            row["id"] = id_enumerator[row["id"]]
            row_dict[row["id"]] = row

    with open(parser_args.attr_info_json_filepath, "r") as attr_info_json_file:
        row_numericalizer = AttrInfoDictParser.from_json(attr_info_json_file, row_dict=row_dict)

    datamodule_args = {
        "row_dict": row_dict,
        "cluster_attr": parser_args.cluster_attr,
        "row_numericalizer": row_numericalizer,
        "batch_size": parser_args.batch_size,
        "row_batch_size": parser_args.row_batch_size,
    }

    if parser_args.left:
        left_id_set = set()
        right_id_set = set()
        for id, row in row_dict.items():
            try:
                if row["__source"] == parser_args.left:
                    left_id_set.add(id)
                else:
                    right_id_set.add(id)
            except KeyError:
                raise KeyError(
                    f'You must provide a "__source" column on {parser_args.row_dict_csv_filepath} '
                    "in order to determine left_id_set and right_id_set on LinkageDataModule"
                )
        datamodule_args["left_id_set"] = left_id_set
        datamodule_args["right_id_set"] = right_id_set
        datamodule_cls = LinkageDataModule
    else:
        datamodule_args["train_cluster_len"] = parser_args.train_len
        datamodule_args["valid_cluster_len"] = parser_args.valid_len
        datamodule_args["test_cluster_len"] = parser_args.test_len
        datamodule_cls = DeduplicationDataModule

    if parser_args.only_plural_clusters:
        datamodule_args["only_plural_clusters"] = parser_args.only_plural_clusters

    if parser_args.num_workers or parser_args.multiprocessing_context:
        for k in ["pair_loader_kwargs", "row_loader_kwargs"]:
            datamodule_args[k] = {}
            if parser_args.num_workers:
                datamodule_args[k]["num_workers"] = parser_args.num_workers
            if parser_args.multiprocessing_context:
                datamodule_args[k]["multiprocessing_context"] = parser_args.multiprocessing_context

    if parser_args.random_seed:
        datamodule_args["random_seed"] = parser_args.random_seed

    return datamodule_cls(**datamodule_args)


def _build_model(datamodule, parser_args):
    model_args = {
        "datamodule": datamodule,
    }

    if parser_args.embedding_size:
        model_args["embedding_size"] = parser_args.embedding_size

    if parser_args.lr:
        model_args["learning_rate"] = parser_args.lr

    if parser_args.ann_k:
        model_args["ann_k"] = parser_args.ann_k

    if parser_args.sim_threshold_list:
        model_args["sim_threshold_list"] = parser_args.sim_threshold_list

    if (
        parser_args.m
        or parser_args.max_m0
        or parser_args.ef_construction
        or parser_args.num_workers
    ):
        model_args["index_build_kwargs"] = {}
        if parser_args.m:
            model_args["index_build_kwargs"]["m"] = parser_args.m
        if parser_args.m:
            model_args["index_build_kwargs"]["max_m0"] = parser_args.max_m0
        if parser_args.m:
            model_args["index_build_kwargs"]["ef_construction"] = parser_args.ef_construction
        if parser_args.num_workers:
            model_args["index_build_kwargs"]["n_threads"] = parser_args.num_workers

    if parser_args.ef_search or parser_args.num_workers:
        model_args["index_search_kwargs"] = {}
        if parser_args.ef_search:
            model_args["index_search_kwargs"]["ef_search"] = parser_args.ef_search
        if parser_args.num_workers:
            model_args["index_search_kwargs"]["n_threads"] = parser_args.num_workers

    return EntityEmbed(**model_args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-attr_info_json_filepath", type=str)
    parser.add_argument("-cluster_attr", type=str)
    parser.add_argument("-row_dict_csv_filepath", type=str)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-row_batch_size", type=int)

    parser.add_argument("-left", type=str)
    parser.add_argument("-random_seed", type=int)
    parser.add_argument("-only_plural_clusters", type=bool)

    parser.add_argument("-train_len", type=int)
    parser.add_argument("-valid_len", type=int)
    parser.add_argument("-test_len", type=int)

    parser.add_argument("-embedding_size", type=int)
    parser.add_argument("-lr", type=str)
    parser.add_argument("-ann_k", type=int)
    parser.add_argument("-sim_threshold_list", type=list)

    parser.add_argument("-num_workers", type=int)
    parser.add_argument("-multiprocessing_context", type=str)

    parser.add_argument("-m", type=int)
    parser.add_argument("-max_m0", type=int)
    parser.add_argument("-ef_construction", type=int)
    parser.add_argument("-ef_search", type=int)

    args = parser.parse_args()

    datamodule = _build_datamodule(args)
    model = _build_model(datamodule, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
