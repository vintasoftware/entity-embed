import csv
import json
import os
import tempfile

import mock
import pytest
from click.testing import CliRunner
from entity_embed.cli import _build_datamodule, _build_trainer, main
from entity_embed.entity_embed import LinkageDataModule


@pytest.fixture
def datamodule_files():
    attr_info_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    row_dict_values = [
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

    with tempfile.NamedTemporaryFile("w", delete=False) as attr_info_json_file:
        json.dump(attr_info_dict, attr_info_json_file)

    with tempfile.NamedTemporaryFile("w", delete=False) as row_dict_csv_file:
        csv_writer = csv.writer(row_dict_csv_file)
        csv_writer.writerow(row_dict_values[0].keys())
        for row in row_dict_values:
            csv_writer.writerow(row.values())

    yield (attr_info_json_file.name, row_dict_csv_file.name)

    os.remove(attr_info_json_file.name)
    os.remove(row_dict_csv_file.name)


@mock.patch("entity_embed.cli.validate_best")
@mock.patch("entity_embed.cli._build_trainer")
def test_cli(
    mock_build_trainer,
    mock_validate_best,
    datamodule_files,
):
    attr_info_json_filepath, csv_filepath = datamodule_files
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-attr_info_json_filepath",
            attr_info_json_filepath,
            "-csv_filepath",
            csv_filepath,
            "-cluster_attr",
            "name",
            "-batch_size",
            10,
            "-row_batch_size",
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
            "-monitor",
            "foo",
            "-min_delta",
            0.00,
            "-patience",
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
        "row_batch_size": 10,
        "monitor": "foo",
        "min_delta": 0,
        "patience": 20,
        "max_epochs": 10,
        "check_val_every_n_epoch": 1,
        "tb_log_dir": None,
        "tb_name": None,
        "gpus": 1,
        "mode": None,
        "ef_search": None,
        "ef_construction": None,
        "max_m0": None,
        "m": None,
        "multiprocessing_context": None,
        "num_workers": None,
        "sim_threshold_list": (0.2, 0.4, 0.6),
        "ann_k": None,
        "lr": None,
        "embedding_size": None,
        "test_len": 1,
        "valid_len": 2,
        "train_len": 2,
        "only_plural_clusters": None,
        "random_seed": None,
        "left": None,
        "model_save_filepath": "weights.ckpt",
    }

    mock_trainer = mock_build_trainer.return_value
    mock_build_trainer.assert_called_once_with(expected_parser_args_dict)
    mock_trainer.fit.assert_called_once()
    mock_validate_best.assert_called_once_with(mock_trainer)
    mock_trainer.test.assert_called_once_with(ckpt_path="best", verbose=False)


def test_build_linkage_datamodule(datamodule_files):
    attr_info_json_filepath, csv_filepath = datamodule_files
    datamodule = _build_datamodule(
        {
            "attr_info_json_filepath": attr_info_json_filepath,
            "csv_filepath": csv_filepath,
            "csv_encoding": "utf-8",
            "cluster_attr": "name",
            "batch_size": 10,
            "row_batch_size": 10,
            "left": "foo",
            "test_len": 1,
            "valid_len": 2,
            "train_len": 2,
            "only_plural_clusters": False,
        }
    )
    assert isinstance(datamodule, LinkageDataModule)


@mock.patch("entity_embed.cli.pl.Trainer")
@mock.patch("entity_embed.cli.ModelCheckpoint")
@mock.patch("entity_embed.cli.TensorBoardLogger")
@mock.patch("entity_embed.cli.EarlyStopping")
def test_build_trainer(
    mock_early_stopping,
    mock_logger,
    mock_checkpoint,
    mock_trainer,
):
    trainer = _build_trainer(
        {
            "monitor": "pair_entity_ratio_at_f0",
            "min_delta": 0.1,
            "patience": 20,
            "mode": None,
            "gpus": 2,
            "max_epochs": 20,
            "check_val_every_n_epoch": 2,
            "tb_name": "foo",
            "tb_log_dir": "bar",
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
