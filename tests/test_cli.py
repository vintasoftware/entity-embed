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


LABELED_ROW_DICT_VALUES = [
    {
        "cluster": "1",
        "name": "foo product",
        "price": "1.00",
        "__source": "foo",
    },
    {
        "cluster": "1",
        "name": "the foo product from world",
        "price": "1.20",
        "__source": "bar",
    },
    {
        "cluster": "1",
        "name": "foo product",
        "price": "1.00",
        "__source": "foo",
    },
    {
        "cluster": "2",
        "id": "5",
        "name": "bar product",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "cluster": "2",
        "name": "bar pr",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "cluster": "2",
        "name": "bar pr",
        "price": "1.30",
        "__source": "foo",
    },
    {
        "cluster": "3",
        "name": "dee",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "cluster": "3",
        "name": "dee pr",
        "price": "1.30",
        "__source": "foo",
    },
    {
        "cluster": "4",
        "name": "999",
        "price": "10.00",
        "__source": "foo",
    },
    {
        "cluster": "4",
        "name": "9999",
        "price": "10.00",
        "__source": "foo",
    },
]
UNLABELED_ROW_DICT_VALUES = [
    {
        "name": "good product",
        "price": "1.50",
        "__source": "foo",
    },
    {
        "name": "bad product",
        "price": "1.90",
        "__source": "bar",
    },
    {
        "name": "badd product",
        "price": "1.90",
        "__source": "foo",
    },
]


def yield_temporary_csv_filename(row_dict_values):
    with tempfile.NamedTemporaryFile("w", delete=False) as row_dict_csv_file:
        csv_writer = csv.writer(row_dict_csv_file)
        csv_writer.writerow(row_dict_values[0].keys())
        for row in row_dict_values:
            csv_writer.writerow(row.values())

    yield row_dict_csv_file.name

    os.remove(row_dict_csv_file.name)


@pytest.fixture
def labeled_input_csv_filepath():
    yield from yield_temporary_csv_filename(LABELED_ROW_DICT_VALUES)


@pytest.fixture
def unlabeled_input_csv_filepath():
    yield from yield_temporary_csv_filename(UNLABELED_ROW_DICT_VALUES)


@mock.patch("entity_embed.cli.validate_best")
@mock.patch("pytorch_lightning.Trainer")
@mock.patch("os.cpu_count", return_value=16)
@mock.patch("torch.manual_seed")
@mock.patch("numpy.random.seed")
@mock.patch("random.seed")
def test_cli_train(
    mock_random_seed,
    mock_np_random_seed,
    mock_torch_random_seed,
    mock_cpu_count,
    mock_trainer,
    mock_validate_best,
    attr_info_json_filepath,
    labeled_input_csv_filepath,
    unlabeled_input_csv_filepath,
):
    runner = CliRunner()
    result = runner.invoke(
        cli.train,
        [
            "--attr_info_json_filepath",
            attr_info_json_filepath,
            "--labeled_input_csv_filepath",
            labeled_input_csv_filepath,
            "--unlabeled_input_csv_filepath",
            unlabeled_input_csv_filepath,
            "--csv_encoding",
            "utf-8",
            "--cluster_attr",
            "cluster",
            "--source_attr",
            "__source",
            "--left_source",
            "foo",
            "--embedding_size",
            300,
            "--lr",
            0.001,
            "--train_len",
            2,
            "--valid_len",
            1,
            "--test_len",
            1,
            "--max_epochs",
            100,
            "--early_stopping_monitor",
            "valid_recall_at_0.9",
            "--early_stopping_min_delta",
            0.01,
            "--early_stopping_patience",
            20,
            "--early_stopping_mode",
            "max",
            "--tb_save_dir",
            "tb_logs",
            "--tb_name",
            "test_experiment",
            "--check_val_every_n_epoch",
            1,
            "--batch_size",
            16,
            "--eval_batch_size",
            64,
            "--num_workers",
            -1,
            "--multiprocessing_context",
            "fork",
            "--sim_threshold",
            0.6,
            "--sim_threshold",
            0.9,
            "--ann_k",
            3,
            "--m",
            64,
            "--max_m0",
            96,
            "--ef_construction",
            150,
            "--ef_search",
            -1,
            "--random_seed",
            42,
            "--model_save_dirpath",
            "trained-models",
        ],
    )

    assert result.exit_code == 0

    expected_args_dict = {
        "attr_info_json_filepath": attr_info_json_filepath,
        "labeled_input_csv_filepath": labeled_input_csv_filepath,
        "unlabeled_input_csv_filepath": unlabeled_input_csv_filepath,
        "csv_encoding": "utf-8",
        "cluster_attr": "cluster",
        "source_attr": "__source",
        "left_source": "foo",
        "embedding_size": 300,
        "lr": 0.001,
        "train_len": 2,
        "valid_len": 1,
        "test_len": 1,
        "max_epochs": 100,
        "early_stopping_monitor": "valid_recall_at_0.9",
        "early_stopping_min_delta": 0.01,
        "early_stopping_patience": 20,
        "early_stopping_mode": "max",
        "tb_save_dir": "tb_logs",
        "tb_name": "test_experiment",
        "check_val_every_n_epoch": 1,
        "batch_size": 16,
        "eval_batch_size": 64,
        "num_workers": 16,  # assigned from os.cpu_count() mock
        "multiprocessing_context": "fork",
        "sim_threshold": (0.6, 0.9),
        "ann_k": 3,
        "m": 64,
        "max_m0": 96,
        "ef_construction": 150,
        "ef_search": -1,
        "random_seed": 42,
        "model_save_dirpath": "trained-models",
        "n_threads": 16,  # assigned
    }

    # random asserts
    mock_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_np_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_torch_random_seed.assert_called_once_with(expected_args_dict["random_seed"])

    # trainer asserts
    __, trainer_kwargs = mock_trainer.call_args
    early_stopping_cb, model_ckpt_cb = trainer_kwargs["callbacks"]
    tb_logger = trainer_kwargs["logger"]

    for early_stopping_attr, early_stopping_kwarg in [
        ("monitor", "early_stopping_monitor"),
        ("min_delta", "early_stopping_min_delta"),
        ("patience", "early_stopping_patience"),
        ("mode", "early_stopping_mode"),
    ]:
        assert (
            getattr(early_stopping_cb, early_stopping_attr)
            == expected_args_dict[early_stopping_kwarg]
        )
    assert early_stopping_cb.verbose

    for model_ckpt_attr, model_ckpt_kwarg in [
        ("monitor", "early_stopping_monitor"),
        ("mode", "early_stopping_mode"),
    ]:
        assert getattr(model_ckpt_cb, model_ckpt_attr) == expected_args_dict[model_ckpt_kwarg]
    assert model_ckpt_cb.dirpath.endswith(expected_args_dict["model_save_dirpath"])
    assert model_ckpt_cb.save_top_k == 1
    assert model_ckpt_cb.verbose

    for tb_logger_attr, tb_logger_kwarg in [("save_dir", "tb_save_dir"), ("name", "tb_name")]:
        assert getattr(tb_logger, tb_logger_attr) == expected_args_dict[tb_logger_kwarg]

    for trainer_kwarg in ["max_epochs", "check_val_every_n_epoch"]:
        assert trainer_kwargs[trainer_kwarg] == expected_args_dict[trainer_kwarg]
    assert trainer_kwargs["gpus"] == 1
    assert trainer_kwargs["reload_dataloaders_every_epoch"]

    # fit assert
    mock_trainer.return_value.fit.assert_called_once()
    (model, datamodule), __ = mock_trainer.return_value.fit.call_args

    # validate and test asserts
    mock_validate_best.assert_called_once_with(mock_trainer.return_value)
    mock_trainer.return_value.test.assert_called_once_with(ckpt_path="best", verbose=False)


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
    for row in LABELED_ROW_DICT_VALUES:
        wrong_row_dict_values.append(
            {
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
