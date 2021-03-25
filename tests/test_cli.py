import copy
import csv
import json
import logging

import entity_embed
import mock
import pytest
from click.testing import CliRunner
from entity_embed import DeduplicationDataModule, EntityEmbed, LinkageDataModule, LinkageEmbed, cli
from entity_embed.data_utils.numericalizer import FieldType


@pytest.fixture
def attr_config_json_filepath(tmp_path):
    filepath = tmp_path / "attr_config.json"
    attr_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    with open(filepath, "w") as f:
        json.dump(attr_config_dict, f)

    yield filepath


LABELED_ROW_DICT_VALUES = [
    {
        "cluster": 1,
        "name": "foo product",
        "price": "1.00",
        "__source": "foo",
    },
    {
        "cluster": 1,
        "name": "the foo product",
        "price": "1.20",
        "__source": "bar",
    },
    {
        "cluster": 1,
        "name": "foo product",
        "price": "1.00",
        "__source": "foo",
    },
    {
        "cluster": 2,
        "name": "bar product",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "cluster": 2,
        "name": "bar pr",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "cluster": 2,
        "name": "bar pr",
        "price": "1.30",
        "__source": "foo",
    },
    {
        "cluster": 3,
        "name": "dee",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "cluster": 3,
        "name": "dee pr",
        "price": "1.30",
        "__source": "foo",
    },
    {
        "cluster": 4,
        "name": "999",
        "price": "10.00",
        "__source": "foo",
    },
    {
        "cluster": 4,
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


def yield_temporary_csv_filepath(row_dict_values, tmp_path, filename):
    filepath = tmp_path / filename

    with open(filepath, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row_dict_values[0].keys())
        for row in row_dict_values:
            csv_writer.writerow(row.values())

    yield filepath


@pytest.fixture
def labeled_input_csv_filepath(tmp_path):
    yield from yield_temporary_csv_filepath(LABELED_ROW_DICT_VALUES, tmp_path, "labeled.csv")


@pytest.fixture
def unlabeled_input_csv_filepath(tmp_path):
    yield from yield_temporary_csv_filepath(UNLABELED_ROW_DICT_VALUES, tmp_path, "unlabeled.csv")


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
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
    mode,
    attr_config_json_filepath,
    labeled_input_csv_filepath,
    unlabeled_input_csv_filepath,
    caplog,
):
    with caplog.at_level(logging.INFO):
        runner = CliRunner()
        result = runner.invoke(
            cli.train,
            [
                "--attr_config_json_filepath",
                attr_config_json_filepath,
                "--labeled_input_csv_filepath",
                labeled_input_csv_filepath,
                "--unlabeled_input_csv_filepath",
                unlabeled_input_csv_filepath,
                "--csv_encoding",
                "utf-8",
                "--cluster_attr",
                "cluster",
                *(
                    [
                        "--source_attr",
                        "__source",
                        "--left_source",
                        "foo",
                    ]
                    if mode == "linkage"
                    else []
                ),
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

    assert result.exit_code == 0, result.stdout_bytes.decode("utf-8")

    expected_args_dict = {
        "attr_config_json_filepath": attr_config_json_filepath,
        "labeled_input_csv_filepath": labeled_input_csv_filepath,
        "unlabeled_input_csv_filepath": unlabeled_input_csv_filepath,
        "csv_encoding": "utf-8",
        "cluster_attr": "cluster",
        **({"source_attr": "__source", "left_source": "foo"} if mode == "linkage" else {}),
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
    expected_attr_config_name_dict = {
        "source_attr": "name",
        "field_type": FieldType.MULTITOKEN,
        "tokenizer": entity_embed.data_utils.numericalizer.default_tokenizer,
        "alphabet": entity_embed.data_utils.numericalizer.DEFAULT_ALPHABET,
        "max_str_len": 8,
        "vocab": None,
        "n_channels": 8,
        "embed_dropout_p": 0.2,
        "use_attention": True,
    }
    if mode == "linkage":
        expected_left_id_set = {
            id_ for id_, row in enumerate(LABELED_ROW_DICT_VALUES) if row["__source"] == "foo"
        }
        expected_right_id_set = {
            id_ for id_, row in enumerate(LABELED_ROW_DICT_VALUES) if row["__source"] == "bar"
        }

    # random asserts
    mock_cpu_count.assert_called_once()
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

    # model asserts
    if mode == "linkage":
        assert isinstance(model, LinkageEmbed)
    else:
        assert isinstance(model, EntityEmbed)
    assert all(
        getattr(model.row_numericalizer.attr_config_dict["name"], k) == expected
        for k, expected in expected_attr_config_name_dict.items()
    )
    assert model.eval_with_clusters
    assert model.embedding_size == expected_args_dict["embedding_size"]
    assert model.learning_rate == expected_args_dict["lr"]
    assert model.ann_k == expected_args_dict["ann_k"]
    assert model.sim_threshold_list == expected_args_dict["sim_threshold"]
    assert model.index_build_kwargs == {
        k: expected_args_dict[k] for k in ["m", "max_m0", "ef_construction", "n_threads"]
    }
    assert model.index_search_kwargs == {
        k: expected_args_dict[k] for k in ["ef_search", "n_threads"]
    }

    # datamodule asserts
    assert datamodule.row_dict == dict(enumerate(LABELED_ROW_DICT_VALUES))
    assert all(
        getattr(model.row_numericalizer.attr_config_dict["name"], k) == expected
        for k, expected in expected_attr_config_name_dict.items()
    )
    assert datamodule.batch_size == expected_args_dict["batch_size"]
    assert datamodule.eval_batch_size == expected_args_dict["eval_batch_size"]
    if mode == "linkage":
        assert datamodule.left_id_set == expected_left_id_set
        assert datamodule.right_id_set == expected_right_id_set
    assert datamodule.train_loader_kwargs == {
        k: expected_args_dict[k] for k in ["num_workers", "multiprocessing_context"]
    }
    assert datamodule.eval_loader_kwargs == {
        k: expected_args_dict[k] for k in ["num_workers", "multiprocessing_context"]
    }
    assert datamodule.random_seed == expected_args_dict["random_seed"]

    # validate and test asserts
    mock_validate_best.assert_called_once_with(mock_trainer.return_value)
    mock_trainer.return_value.test.assert_called_once_with(ckpt_path="best", verbose=False)

    # assert outputs
    assert str(mock_validate_best.return_value) in caplog.records[-4].message
    assert str(mock_trainer.return_value.test.return_value) in caplog.records[-3].message
    assert "Saved best model at path:" in caplog.records[-2].message
    assert (
        str(mock_trainer.return_value.checkpoint_callback.best_model_path)
        in caplog.records[-1].message
    )


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
@mock.patch("os.cpu_count", return_value=16)
@mock.patch("torch.manual_seed")
@mock.patch("numpy.random.seed")
@mock.patch("random.seed")
@mock.patch("entity_embed.data_utils.utils.assign_clusters")
def test_cli_predict(
    mock_assign_clusters,
    mock_random_seed,
    mock_np_random_seed,
    mock_torch_random_seed,
    mock_cpu_count,
    mode,
    attr_config_json_filepath,
    unlabeled_input_csv_filepath,
    caplog,
    tmp_path,
):
    if mode == "linkage":
        expected_model_cls = LinkageEmbed
    else:
        expected_model_cls = EntityEmbed
    with mock.patch(
        f"entity_embed.{expected_model_cls.__name__}.load_from_checkpoint"
    ) as model_load, caplog.at_level(logging.INFO):
        expected_cluster_mapping = {0: 0, 1: 1, 2: 1}
        expected_cluster_dict = {0: [0], 1: [1, 2]}
        model_load.return_value.predict_clusters.return_value = (
            expected_cluster_mapping,
            expected_cluster_dict,
        )
        expected_output_csv_filepath = tmp_path / f"labeled-{mode}.csv"

        runner = CliRunner()
        result = runner.invoke(
            cli.predict,
            [
                "--model_save_filepath",
                "trained-model.ckpt",
                "--attr_config_json_filepath",
                attr_config_json_filepath,
                "--unlabeled_input_csv_filepath",
                unlabeled_input_csv_filepath,
                "--csv_encoding",
                "utf-8",
                *(
                    [
                        "--source_attr",
                        "__source",
                        "--left_source",
                        "foo",
                    ]
                    if mode == "linkage"
                    else []
                ),
                "--eval_batch_size",
                64,
                "--num_workers",
                -1,
                "--multiprocessing_context",
                "fork",
                "--sim_threshold",
                0.6,
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
                "--output_csv_filepath",
                expected_output_csv_filepath,
                "--cluster_attr",
                "cluster",
            ],
        )

    assert result.exit_code == 0, result.stdout_bytes.decode("utf-8")

    expected_args_dict = {
        "model_save_filepath": "trained-model.ckpt",
        "attr_config_json_filepath": attr_config_json_filepath,
        "unlabeled_input_csv_filepath": unlabeled_input_csv_filepath,
        "csv_encoding": "utf-8",
        **({"source_attr": "__source", "left_source": "foo"} if mode == "linkage" else {}),
        "eval_batch_size": 64,
        "num_workers": 16,  # assigned from os.cpu_count() mock
        "multiprocessing_context": "fork",
        "sim_threshold": 0.6,
        "ann_k": 3,
        "m": 64,
        "max_m0": 96,
        "ef_construction": 150,
        "ef_search": -1,
        "random_seed": 42,
        "output_csv_filepath": expected_output_csv_filepath,
        "cluster_attr": "cluster",
        "n_threads": 16,  # assigned
    }
    if mode == "linkage":
        expected_left_id_set = {
            id_ for id_, row in enumerate(UNLABELED_ROW_DICT_VALUES) if row["__source"] == "foo"
        }
        expected_right_id_set = {
            id_ for id_, row in enumerate(UNLABELED_ROW_DICT_VALUES) if row["__source"] == "bar"
        }

    # random asserts
    mock_cpu_count.assert_called_once()
    mock_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_np_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_torch_random_seed.assert_called_once_with(expected_args_dict["random_seed"])

    # predict_clusters asserts
    expected_row_dict = dict(enumerate(UNLABELED_ROW_DICT_VALUES))
    model_load.return_value.predict_clusters.assert_called_once_with(
        **{
            "row_dict": expected_row_dict,
            **(
                {
                    "left_id_set": expected_left_id_set,
                    "right_id_set": expected_right_id_set,
                }
                if mode == "linkage"
                else {}
            ),
            "batch_size": expected_args_dict["eval_batch_size"],
            "ann_k": expected_args_dict["ann_k"],
            "sim_threshold": expected_args_dict["sim_threshold"],
            "loader_kwargs": {
                "num_workers": expected_args_dict["num_workers"],
                "multiprocessing_context": expected_args_dict["multiprocessing_context"],
            },
            "index_build_kwargs": {
                k: expected_args_dict[k] for k in ["m", "max_m0", "ef_construction", "n_threads"]
            },
            "index_search_kwargs": {k: expected_args_dict[k] for k in ["ef_search", "n_threads"]},
        }
    )

    # assign_clusters assert
    mock_assign_clusters.assert_called_once_with(
        row_dict=expected_row_dict,
        cluster_attr=expected_args_dict["cluster_attr"],
        cluster_mapping=expected_cluster_mapping,
    )

    # assert outputs
    assert (
        "Cluster size quantiles: [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]"
        in caplog.records[-4].message
    )
    assert "Top 5 cluster sizes: [1, 2]" in caplog.records[-3].message
    assert "File is now labeled at column cluster:" in caplog.records[-2].message
    assert str(expected_args_dict["output_csv_filepath"]) in caplog.records[-1].message

    # assert output file
    expected_out_rows = [
        # cluster is empty because assign_clusters is mocked
        {"cluster": "", **row}
        for row in UNLABELED_ROW_DICT_VALUES
    ]
    with open(expected_output_csv_filepath, newline="") as f:
        out_rows = [row for row in csv.DictReader(f)]
    assert out_rows == expected_out_rows


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
def test_build_datamodule(mode):
    expected_row_dict = dict(enumerate(LABELED_ROW_DICT_VALUES))
    expected_row_numericalizer = mock.Mock()
    if mode == "linkage":
        expected_left_id_set = {
            id_ for id_, row in enumerate(LABELED_ROW_DICT_VALUES) if row["__source"] == "foo"
        }
        expected_right_id_set = {
            id_ for id_, row in enumerate(LABELED_ROW_DICT_VALUES) if row["__source"] == "bar"
        }
    expected_kwargs = {
        "row_dict": expected_row_dict,
        "cluster_attr": "cluster",
        "row_numericalizer": expected_row_numericalizer,
        "batch_size": 16,
        "eval_batch_size": 64,
        "train_cluster_len": 20,
        "valid_cluster_len": 10,
        "test_cluster_len": 5,
        **(
            {
                "left_id_set": expected_left_id_set,
                "right_id_set": expected_right_id_set,
            }
            if mode == "linkage"
            else {}
        ),
        "train_loader_kwargs": {
            "num_workers": 16,
            "multiprocessing_context": "fork",
        },
        "eval_loader_kwargs": {
            "num_workers": 16,
            "multiprocessing_context": "fork",
        },
        "random_seed": 42,
    }
    if mode == "linkage":
        expected_dm_cls = LinkageDataModule
    else:
        expected_dm_cls = DeduplicationDataModule

    with mock.patch(f"entity_embed.cli.{expected_dm_cls.__name__}") as mock_datamodule:
        cli._build_datamodule(
            row_dict=expected_row_dict,
            row_numericalizer=expected_row_numericalizer,
            kwargs={
                "cluster_attr": "cluster",
                "batch_size": 16,
                "eval_batch_size": 64,
                "train_len": 20,
                "valid_len": 10,
                "test_len": 5,
                **({"source_attr": "__source", "left_source": "foo"} if mode == "linkage" else {}),
                "num_workers": 16,
                "multiprocessing_context": "fork",
                "random_seed": 42,
            },
        )

    mock_datamodule.assert_called_once_with(**expected_kwargs)


@pytest.mark.parametrize("missing_kwarg", ["source_attr", "left_source"])
def test_build_linkage_datamodule_without_source_attr_or_left_source_raises(missing_kwarg):
    with pytest.raises(KeyError) as exc:
        kwargs = {
            "cluster_attr": "cluster",
            "batch_size": 16,
            "eval_batch_size": 64,
            "train_len": 20,
            "valid_len": 10,
            "test_len": 5,
            "source_attr": "__source",
            "left_source": "foo",
            "num_workers": 16,
            "multiprocessing_context": "fork",
            "random_seed": 42,
        }
        del kwargs[missing_kwarg]
        cli._build_datamodule(
            row_dict=dict(enumerate(LABELED_ROW_DICT_VALUES)),
            row_numericalizer=mock.Mock(),
            kwargs=kwargs,
        )
        assert 'must provide BOTH "source_attr" and "left_source"' in str(exc)


def test_build_linkage_datamodule_without_source_in_row_raises():
    wrong_row_dict_values = copy.deepcopy(LABELED_ROW_DICT_VALUES)
    for row in wrong_row_dict_values:
        del row["__source"]

    with pytest.raises(KeyError) as exc:
        cli._build_datamodule(
            row_dict=dict(enumerate(wrong_row_dict_values)),
            row_numericalizer=mock.Mock(),
            kwargs={
                "cluster_attr": "cluster",
                "batch_size": 16,
                "eval_batch_size": 64,
                "train_len": 20,
                "valid_len": 10,
                "test_len": 5,
                "source_attr": "__source",
                "left_source": "foo",
                "num_workers": 16,
                "multiprocessing_context": "fork",
                "random_seed": 42,
            },
        )
        assert "KeyError: '__source'" in str(exc)


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
            "max_epochs": 20,
            "check_val_every_n_epoch": 2,
            "tb_name": "foo",
            "tb_save_dir": "bar",
            "model_save_dirpath": "trained-models",
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
        save_top_k=1,
        mode="min",
        verbose=True,
        dirpath="trained-models",
    )

    mock_logger.assert_called_once_with("bar", name="foo")

    mock_trainer.assert_called_once_with(
        gpus=1,
        max_epochs=20,
        check_val_every_n_epoch=2,
        callbacks=[mock_early_stopping.return_value, mock_checkpoint.return_value],
        reload_dataloaders_every_epoch=True,
        logger=mock_logger.return_value,
    )

    assert trainer == mock_trainer.return_value


@mock.patch("entity_embed.cli.ModelCheckpoint")
@mock.patch("entity_embed.cli.EarlyStopping")
def test_build_trainer_with_only_tb_name_raises(
    mock_early_stopping,
    mock_checkpoint,
):
    with pytest.raises(KeyError) as exc:
        cli._build_trainer(
            {
                "early_stopping_monitor": "pair_entity_ratio_at_f0",
                "early_stopping_min_delta": 0.1,
                "early_stopping_patience": 20,
                "early_stopping_mode": None,
                "max_epochs": 20,
                "check_val_every_n_epoch": 2,
                "tb_name": "foo",
                "tb_save_dir": None,
                "model_save_dirpath": "trained-models",
            }
        )
        assert 'Please provide both "tb_name" and "tb_save_dir"' in str(exc)


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
def test_build_model(mode):
    if mode == "linkage":
        expected_model_cls = LinkageEmbed
    else:
        expected_model_cls = EntityEmbed

    mock_row_numericalizer = mock.MagicMock()

    with mock.patch(f"entity_embed.cli.{expected_model_cls.__name__}") as mock_model:
        cli._build_model(
            mock_row_numericalizer,
            {
                "embedding_size": 100,
                "lr": 0.2,
                "ann_k": 10,
                "sim_threshold": (0.2, 0.4, 0.6, 0.8),
                "m": None,
                "max_m0": None,
                "ef_construction": 128,
                "n_threads": 4,
                "ef_search": -1,
                **({"source_attr": "__source", "left_source": "foo"} if mode == "linkage" else {}),
            },
        )

    mock_model.assert_called_once_with(
        row_numericalizer=mock_row_numericalizer,
        eval_with_clusters=True,
        embedding_size=100,
        learning_rate=0.2,
        ann_k=10,
        sim_threshold_list=(0.2, 0.4, 0.6, 0.8),
        index_build_kwargs={"ef_construction": 128, "n_threads": 4},
        index_search_kwargs={"ef_search": -1, "n_threads": 4},
    )
