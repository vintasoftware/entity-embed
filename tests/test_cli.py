import csv
import json
import os
import tempfile

import mock
import pytest
from click.testing import CliRunner
from entity_embed import cli
from entity_embed.data_utils.numericalizer import RowNumericalizer


@pytest.fixture
def attr_info_json_filepath():
    attr_info_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    with tempfile.NamedTemporaryFile("w", delete=False) as attr_info_json_file:
        json.dump(attr_info_dict, attr_info_json_file)

    yield attr_info_json_file.name

    os.remove(attr_info_json_file.name)


ROW_DICT_VALUES = [
    {
        "id": "1",
        "name": "foo product",
        "price": 1.00,
        "__source": "foo",
    },
    {
        "id": "2",
        "name": "the foo product from world",
        "price": 1.20,
        "__source": "bar",
    },
    {
        "id": "3",
        "name": "the foo product from world",
        "price": 1.00,
        "__source": "foo",
    },
    {
        "id": "4",
        "name": "foo product",
        "price": 1.00,
        "__source": "foo",
    },
    {
        "id": "5",
        "name": "foo product",
        "price": 1.30,
        "__source": "bar",
    },
    {
        "id": "6",
        "name": "foo pr",
        "price": 1.30,
        "__source": "bar",
    },
    {
        "id": "7",
        "name": "foo pr",
        "price": 1.30,
        "__source": "foo",
    },
    {
        "id": "8",
        "name": "foo product",
        "price": 1.30,
        "__source": "bar",
    },
    {
        "id": "9",
        "name": "foo propaganda",
        "price": 1.30,
        "__source": "bar",
    },
    {
        "id": "10",
        "name": "foo propaganda",
        "price": 1.30,
        "__source": "foo",
    },
]


@pytest.fixture
def csv_filepath():
    with tempfile.NamedTemporaryFile("w", delete=False) as row_dict_csv_file:
        csv_writer = csv.writer(row_dict_csv_file)
        csv_writer.writerow(ROW_DICT_VALUES[0].keys())
        for row in ROW_DICT_VALUES:
            csv_writer.writerow(row.values())

    yield row_dict_csv_file.name

    os.remove(row_dict_csv_file.name)


@mock.patch("entity_embed.cli.validate_best")
@mock.patch("entity_embed.cli._build_trainer")
def test_cli(
    mock_build_trainer,
    mock_validate_best,
    attr_info_json_filepath,
    csv_filepath,
):
    runner = CliRunner()
    result = runner.invoke(
        cli.train,
        [
            "-attr_info_json_filepath",
            attr_info_json_filepath,
            "-csv_filepath",
            csv_filepath,
            "-cluster_attr",
            "name",
            "-batch_size",
            10,
            "-eval_batch_size",
            10,
            "-sim_threshold",
            0.2,
            "-sim_threshold",
            0.4,
            "-sim_threshold",
            0.6,
            "-test_len",
            1,
            "-valid_len",
            2,
            "-train_len",
            2,
            "-early_stopping_monitor",
            "foo",
            "-early_stopping_min_delta",
            0.00,
            "-early_stopping_patience",
            20,
            "-max_epochs",
            10,
            "-check_val_every_n_epoch",
            1,
            "-model_save_filepath",
            "weights.ckpt",
        ],
    )

    assert result.exit_code == 0

    expected_parser_args_dict = {
        "attr_info_json_filepath": attr_info_json_filepath,
        "csv_filepath": csv_filepath,
        "csv_encoding": "utf-8",
        "cluster_attr": "name",
        "batch_size": 10,
        "eval_batch_size": 10,
        "early_stopping_monitor": "foo",
        "early_stopping_min_delta": 0,
        "early_stopping_patience": 20,
        "max_epochs": 10,
        "check_val_every_n_epoch": 1,
        "tb_save_dir": None,
        "tb_name": None,
        "gpus": 1,
        "early_stopping_mode": None,
        "ef_search": None,
        "ef_construction": None,
        "max_m0": None,
        "m": None,
        "multiprocessing_context": None,
        "num_workers": None,
        "n_threads": None,
        "sim_threshold_list": (0.2, 0.4, 0.6),
        "ann_k": None,
        "lr": None,
        "embedding_size": None,
        "test_len": 1,
        "valid_len": 2,
        "train_len": 2,
        "random_seed": None,
        "left": None,
        "model_save_filepath": "weights.ckpt",
    }

    mock_trainer = mock_build_trainer.return_value
    mock_build_trainer.assert_called_once_with(expected_parser_args_dict)
    mock_trainer.fit.assert_called_once()
    mock_validate_best.assert_called_once_with(mock_trainer)
    mock_trainer.test.assert_called_once_with(ckpt_path="best", verbose=False)


@mock.patch("entity_embed.entity_embed.LinkageDataModule.__init__", return_value=None)
def test_build_linkage_datamodule(
    mock_linkage_datamodule,
    attr_info_json_filepath,
    csv_filepath,
):
    cli._build_datamodule(
        {
            "attr_info_json_filepath": attr_info_json_filepath,
            "csv_filepath": csv_filepath,
            "csv_encoding": "utf-8",
            "cluster_attr": "name",
            "batch_size": 10,
            "eval_batch_size": 10,
            "left": "foo",
            "test_len": 1,
            "valid_len": 2,
            "train_len": 2,
            "random_seed": 30,
            "num_workers": 16,
            "multiprocessing_context": None,
        }
    )

    mock_linkage_datamodule.assert_called_once_with(
        row_dict=mock.ANY,
        cluster_attr="name",
        row_numericalizer=mock.ANY,
        batch_size=10,
        eval_batch_size=10,
        train_cluster_len=2,
        valid_cluster_len=2,
        test_cluster_len=1,
        left_id_set={0, 2, 3, 6, 9},
        right_id_set={1, 4, 5, 7, 8},
        random_seed=30,
        pair_loader_kwargs={"num_workers": 16},
        row_loader_kwargs={"num_workers": 16},
    )
    call_args = mock_linkage_datamodule.call_args.kwargs
    assert isinstance(call_args["row_numericalizer"], RowNumericalizer)


def test_build_linkage_datamodule_without_source_raises(attr_info_json_filepath):
    wrong_row_dict_values = []
    for row in ROW_DICT_VALUES:
        wrong_row_dict_values.append(
            {
                "id": row["id"],
                "name": row["name"],
                "price": row["price"],
            }
        )

    with tempfile.NamedTemporaryFile("w", delete=False) as row_dict_csv_file:
        csv_writer = csv.writer(row_dict_csv_file)
        csv_writer.writerow(wrong_row_dict_values[0].keys())
        for row in wrong_row_dict_values:
            csv_writer.writerow(row.values())

    with pytest.raises(KeyError):
        cli._build_datamodule(
            {
                "attr_info_json_filepath": attr_info_json_filepath,
                "csv_filepath": row_dict_csv_file.name,
                "csv_encoding": "utf-8",
                "cluster_attr": "name",
                "batch_size": 10,
                "eval_batch_size": 10,
                "left": "foo",
                "test_len": 1,
                "valid_len": 2,
                "train_len": 2,
            }
        )

    os.remove(row_dict_csv_file.name)


@mock.patch("entity_embed.cli.pl.Trainer")
@mock.patch("entity_embed.cli.TensorBoardLogger")
@mock.patch("entity_embed.cli.ModelCheckpoint")
@mock.patch("entity_embed.cli.EarlyStopping")
def test_build_trainer(
    mock_early_stopping,
    mock_checkpoint,
    mock_logger,
    mock_trainer,
):
    trainer = cli._build_trainer(
        {
            "early_stopping_monitor": "pair_entity_ratio_at_f0",
            "early_stopping_min_delta": 0.1,
            "early_stopping_patience": 20,
            "early_stopping_mode": None,
            "gpus": 2,
            "max_epochs": 20,
            "check_val_every_n_epoch": 2,
            "tb_name": "foo",
            "tb_save_dir": "bar",
            "model_save_filepath": "weights.ckpt",
        }
    )

    mock_early_stopping.assert_called_once_with(
        monitor="pair_entity_ratio_at_f0",
        min_delta=0.1,
        patience=20,
        verbose=True,
        mode="min",
    )

    mock_checkpoint.assert_called_once_with(
        monitor="pair_entity_ratio_at_f0",
        verbose=True,
        filename="weights.ckpt",
        save_top_k=1,
    )

    mock_logger.assert_called_once_with("bar", name="foo")

    mock_trainer.assert_called_once_with(
        gpus=2,
        max_epochs=20,
        check_val_every_n_epoch=2,
        callbacks=[mock_early_stopping.return_value, mock_checkpoint.return_value],
        logger=mock_logger.return_value,
    )

    assert trainer == mock_trainer.return_value


@mock.patch("entity_embed.cli.ModelCheckpoint")
@mock.patch("entity_embed.cli.EarlyStopping")
def test_build_trainer_with_only_tb_name_raises(
    mock_early_stopping,
    mock_checkpoint,
):
    with pytest.raises(KeyError):
        cli._build_trainer(
            {
                "early_stopping_monitor": "pair_entity_ratio_at_f0",
                "early_stopping_min_delta": 0.1,
                "early_stopping_patience": 20,
                "early_stopping_mode": None,
                "gpus": 2,
                "max_epochs": 20,
                "check_val_every_n_epoch": 2,
                "tb_name": "foo",
                "tb_save_dir": None,
                "model_save_filepath": "weights.ckpt",
            }
        )

    mock_early_stopping.assert_called_once_with(
        monitor="pair_entity_ratio_at_f0",
        min_delta=0.1,
        patience=20,
        verbose=True,
        mode="min",
    )

    mock_checkpoint.assert_called_once_with(
        monitor="pair_entity_ratio_at_f0",
        verbose=True,
        filename="weights.ckpt",
        save_top_k=1,
    )


@mock.patch("entity_embed.entity_embed.EntityEmbed.__init__", return_value=None)
def test_build_model(mock_entity_embed):
    mock_datamodule = mock.MagicMock()
    cli._build_model(
        mock_datamodule,
        {
            "embedding_size": 125,
            "lr": 0.2,
            "ann_k": 10,
            "sim_threshold_list": (0.2, 0.4, 0.6, 0.8),
            "m": None,
            "max_m0": None,
            "ef_construction": 128,
            "n_threads": 4,
            "ef_search": -1,
        },
    )
    mock_entity_embed.assert_called_once_with(
        datamodule=mock_datamodule,
        embedding_size=125,
        learning_rate=0.2,
        ann_k=10,
        sim_threshold_list=(0.2, 0.4, 0.6, 0.8),
        index_build_kwargs={"ef_construction": 128, "n_threads": 4},
        index_search_kwargs={"ef_search": -1, "n_threads": 4},
    )
