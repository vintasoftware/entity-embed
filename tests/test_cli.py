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
def field_config_json(tmp_path):
    filepath = tmp_path / "field_config.json"
    field_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
    }

    with open(filepath, "w") as f:
        json.dump(field_config_dict, f)

    yield filepath


LABELED_RECORD_DICT_VALUES = [
    {
        "id": 0,
        "cluster": 1,
        "name": "foo product",
        "price": "1.00",
        "__source": "foo",
    },
    {
        "id": 1,
        "cluster": 1,
        "name": "the foo product",
        "price": "1.20",
        "__source": "bar",
    },
    {
        "id": 2,
        "cluster": 1,
        "name": "foo product",
        "price": "1.00",
        "__source": "foo",
    },
    {
        "id": 3,
        "cluster": 2,
        "name": "bar product",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "id": 4,
        "cluster": 2,
        "name": "bar pr",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "id": 5,
        "cluster": 2,
        "name": "bar pr",
        "price": "1.30",
        "__source": "foo",
    },
    {
        "id": 6,
        "cluster": 3,
        "name": "dee",
        "price": "1.30",
        "__source": "bar",
    },
    {
        "id": 7,
        "cluster": 3,
        "name": "dee pr",
        "price": "1.30",
        "__source": "foo",
    },
    {
        "id": 8,
        "cluster": 4,
        "name": "999",
        "price": "10.00",
        "__source": "foo",
    },
    {
        "id": 9,
        "cluster": 4,
        "name": "9999",
        "price": "10.00",
        "__source": "foo",
    },
]
UNLABELED_RECORD_DICT_VALUES = [
    {
        "id": 10,
        "name": "good product",
        "price": "1.50",
        "__source": "foo",
    },
    {
        "id": 11,
        "name": "bad product",
        "price": "1.90",
        "__source": "bar",
    },
    {
        "id": 12,
        "name": "badd product",
        "price": "1.90",
        "__source": "foo",
    },
]


def yield_temporary_csv_filepath(record_dict_values, tmp_path, filename):
    filepath = tmp_path / filename

    with open(filepath, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(record_dict_values[0].keys())
        for record in record_dict_values:
            csv_writer.writerow(record.values())

    yield filepath


@pytest.fixture
def train_csv(tmp_path):
    yield from yield_temporary_csv_filepath(LABELED_RECORD_DICT_VALUES[:5], tmp_path, "train.csv")


@pytest.fixture
def valid_csv(tmp_path):
    yield from yield_temporary_csv_filepath(LABELED_RECORD_DICT_VALUES[5:8], tmp_path, "valid.csv")


@pytest.fixture
def test_csv(tmp_path):
    yield from yield_temporary_csv_filepath(LABELED_RECORD_DICT_VALUES[8:], tmp_path, "test.csv")


@pytest.fixture
def unlabeled_csv(tmp_path):
    yield from yield_temporary_csv_filepath(UNLABELED_RECORD_DICT_VALUES, tmp_path, "unlabeled.csv")


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
@mock.patch("os.cpu_count", return_value=16)
@mock.patch("torch.manual_seed")
@mock.patch("numpy.random.seed")
@mock.patch("random.seed")
def test_cli_train(
    mock_random_seed,
    mock_np_random_seed,
    mock_torch_random_seed,
    mock_cpu_count,
    mode,
    field_config_json,
    train_csv,
    valid_csv,
    test_csv,
    unlabeled_csv,
    caplog,
):
    if mode == "linkage":
        expected_model_cls = LinkageEmbed
    else:
        expected_model_cls = EntityEmbed

    with mock.patch(
        f"entity_embed.cli.{expected_model_cls.__name__}"
    ) as mock_model, caplog.at_level(logging.INFO):
        runner = CliRunner()
        result = runner.invoke(
            cli.train,
            [
                "--field_config_json",
                field_config_json,
                "--train_csv",
                train_csv,
                "--valid_csv",
                valid_csv,
                "--test_csv",
                test_csv,
                "--unlabeled_csv",
                unlabeled_csv,
                "--csv_encoding",
                "utf-8",
                "--cluster_field",
                "cluster",
                *(
                    [
                        "--source_field",
                        "__source",
                        "--left_source",
                        "foo",
                    ]
                    if mode == "linkage"
                    else []
                ),
                "--embedding_size",
                500,
                "--lr",
                0.005,
                "--min_epochs",
                1,
                "--max_epochs",
                50,
                "--early_stop_monitor",
                "valid_recall_at_0.5",
                "--early_stop_min_delta",
                0.01,
                "--early_stop_patience",
                10,
                "--early_stop_mode",
                "max",
                "--tb_save_dir",
                "tb_logs",
                "--tb_name",
                "test_experiment",
                "--check_val_every_n_epoch",
                2,
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
                "--model_save_dir",
                "trained-models",
                "--use_gpu",
                False,
            ],
        )

    assert result.exit_code == 0, result.stdout_bytes.decode("utf-8")

    expected_args_dict = {
        "field_config_json": field_config_json,
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "test_csv": test_csv,
        "unlabeled_csv": unlabeled_csv,
        "csv_encoding": "utf-8",
        "cluster_field": "cluster",
        **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
        "embedding_size": 500,
        "lr": 0.005,
        "min_epochs": 1,
        "max_epochs": 50,
        "early_stop_monitor": "valid_recall_at_0.5",
        "early_stop_min_delta": 0.01,
        "early_stop_patience": 10,
        "early_stop_mode": "max",
        "tb_save_dir": "tb_logs",
        "tb_name": "test_experiment",
        "check_val_every_n_epoch": 2,
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
        "model_save_dir": "trained-models",
        "n_threads": 16,  # assigned
        "use_gpu": False,
    }
    expected_field_config_name_dict = {
        "key": "name",
        "field_type": FieldType.MULTITOKEN,
        "tokenizer": entity_embed.data_utils.numericalizer.default_tokenizer,
        "alphabet": entity_embed.data_utils.numericalizer.DEFAULT_ALPHABET,
        "max_str_len": 8,
        "vocab": None,
        "n_channels": 8,
        "embed_dropout_p": 0.2,
        "use_attention": True,
    }

    # random asserts
    mock_cpu_count.assert_called_once()
    mock_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_np_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_torch_random_seed.assert_called_once_with(expected_args_dict["random_seed"])

    # model asserts
    mock_model.assert_called_once_with(
        **{
            "record_numericalizer": mock.ANY,  # record_numericalizer, will get below and assert
            **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
            "embedding_size": expected_args_dict["embedding_size"],
            "learning_rate": expected_args_dict["lr"],
            "ann_k": expected_args_dict["ann_k"],
            "sim_threshold_list": expected_args_dict["sim_threshold"],
            "index_build_kwargs": {
                k: expected_args_dict[k] for k in ["m", "max_m0", "ef_construction", "n_threads"]
            },
            "index_search_kwargs": {k: expected_args_dict[k] for k in ["ef_search", "n_threads"]},
        }
    )
    record_numericalizer = mock_model.call_args[1]["record_numericalizer"]

    # record_numericalizer asserts
    assert all(
        getattr(record_numericalizer.field_config_dict["name"], k) == expected
        for k, expected in expected_field_config_name_dict.items()
    )

    # fit asserts
    mock_model.return_value.fit.assert_called_once_with(
        mock.ANY,  # datamodule, will get below and assert
        min_epochs=expected_args_dict["min_epochs"],
        max_epochs=expected_args_dict["max_epochs"],
        check_val_every_n_epoch=expected_args_dict["check_val_every_n_epoch"],
        early_stop_monitor=expected_args_dict["early_stop_monitor"],
        early_stop_min_delta=expected_args_dict["early_stop_min_delta"],
        early_stop_patience=expected_args_dict["early_stop_patience"],
        early_stop_mode=expected_args_dict["early_stop_mode"],
        early_stop_verbose=True,
        model_save_top_k=1,
        model_save_dir=expected_args_dict["model_save_dir"],
        model_save_verbose=True,
        tb_save_dir=expected_args_dict["tb_save_dir"],
        tb_name=expected_args_dict["tb_name"],
        use_gpu=expected_args_dict["use_gpu"],
    )
    datamodule = mock_model.return_value.fit.call_args[0][0]

    # datamodule asserts
    assert datamodule.train_record_dict == {
        record["id"]: record for record in LABELED_RECORD_DICT_VALUES[:5]
    }
    assert datamodule.valid_record_dict == {
        record["id"]: record for record in LABELED_RECORD_DICT_VALUES[5:8]
    }
    assert datamodule.test_record_dict == {
        record["id"]: record for record in LABELED_RECORD_DICT_VALUES[8:]
    }
    assert datamodule.record_numericalizer == record_numericalizer
    assert datamodule.batch_size == expected_args_dict["batch_size"]
    assert datamodule.eval_batch_size == expected_args_dict["eval_batch_size"]
    assert datamodule.train_loader_kwargs == {
        k: expected_args_dict[k] for k in ["num_workers", "multiprocessing_context"]
    }
    assert datamodule.eval_loader_kwargs == {
        k: expected_args_dict[k] for k in ["num_workers", "multiprocessing_context"]
    }
    assert datamodule.random_seed == expected_args_dict["random_seed"]

    # validate and test asserts
    mock_model.return_value.validate.assert_called_once_with(datamodule)
    mock_model.return_value.test.assert_called_once_with(datamodule)

    # assert outputs
    assert "Validating best model:" in caplog.records[-6].message
    assert str(mock_model.return_value.validate.return_value) in caplog.records[-5].message
    assert "Testing best model:" in caplog.records[-4].message
    assert str(mock_model.return_value.test.return_value) in caplog.records[-3].message
    assert "Saved best model at path:" in caplog.records[-2].message
    assert (
        str(mock_model.return_value.fit.return_value.checkpoint_callback.best_model_path)
        in caplog.records[-1].message
    )


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
@mock.patch("os.cpu_count", return_value=16)
@mock.patch("torch.manual_seed")
@mock.patch("numpy.random.seed")
@mock.patch("random.seed")
@mock.patch("torch.device")
def test_cli_predict(
    mock_torch_device,
    mock_random_seed,
    mock_np_random_seed,
    mock_torch_random_seed,
    mock_cpu_count,
    mode,
    unlabeled_csv,
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
        predict_pairs_mock = model_load.return_value.to.return_value.predict_pairs
        predict_pairs_mock.return_value = [(11, 12)]
        expected_output_json = tmp_path / f"labeled-{mode}.json"

        runner = CliRunner()
        result = runner.invoke(
            cli.predict,
            [
                "--model_save_filepath",
                "trained-model.ckpt",
                "--unlabeled_csv",
                unlabeled_csv,
                "--csv_encoding",
                "utf-8",
                *(
                    [
                        "--source_field",
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
                "--output_json",
                expected_output_json,
                "--use_gpu",
                False,
            ],
        )

    assert result.exit_code == 0, result.stdout_bytes.decode("utf-8")

    expected_args_dict = {
        "model_save_filepath": "trained-model.ckpt",
        "field_config_json": field_config_json,
        "unlabeled_csv": unlabeled_csv,
        "csv_encoding": "utf-8",
        **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
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
        "output_json": expected_output_json,
        "cluster_field": "cluster",
        "n_threads": 16,  # assigned
    }

    # random asserts
    mock_cpu_count.assert_called_once()
    mock_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_np_random_seed.assert_called_once_with(expected_args_dict["random_seed"])
    mock_torch_random_seed.assert_called_once_with(expected_args_dict["random_seed"])

    # cuda asserts
    mock_torch_device.assert_called_once_with("cpu")

    # predict_pairs asserts
    expected_record_dict = {record["id"]: record for record in UNLABELED_RECORD_DICT_VALUES}
    predict_pairs_mock.assert_called_once_with(
        **{
            "record_dict": expected_record_dict,
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

    # assert outputs
    assert "Found 1 candidate pairs, writing to JSON file at:" in caplog.records[-2].message
    assert str(expected_args_dict["output_json"]) in caplog.records[-1].message

    # assert output file
    expected_pairs = [(11, 12)]
    with open(expected_output_json, newline="") as f:
        json_pairs = json.load(f)
        json_pairs = [tuple(pair) for pair in json_pairs]
    assert json_pairs == expected_pairs


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
def test_build_datamodule(mode):
    expected_train_record_dict = {record["id"]: record for record in LABELED_RECORD_DICT_VALUES[:5]}
    expected_valid_record_dict = {
        record["id"]: record for record in LABELED_RECORD_DICT_VALUES[5:8]
    }
    expected_test_record_dict = {record["id"]: record for record in LABELED_RECORD_DICT_VALUES[8:]}
    expected_record_numericalizer = mock.Mock()
    expected_kwargs = {
        "train_record_dict": expected_train_record_dict,
        "valid_record_dict": expected_valid_record_dict,
        "test_record_dict": expected_test_record_dict,
        "cluster_field": "cluster",
        "record_numericalizer": expected_record_numericalizer,
        "batch_size": 16,
        "eval_batch_size": 64,
        **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
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
            train_record_dict=expected_train_record_dict,
            valid_record_dict=expected_valid_record_dict,
            test_record_dict=expected_test_record_dict,
            record_numericalizer=expected_record_numericalizer,
            kwargs={
                "cluster_field": "cluster",
                "batch_size": 16,
                "eval_batch_size": 64,
                **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
                "num_workers": 16,
                "multiprocessing_context": "fork",
                "random_seed": 42,
            },
        )

    mock_datamodule.assert_called_once_with(**expected_kwargs)


@pytest.mark.parametrize("missing_kwarg", ["source_field", "left_source"])
def test_build_linkage_datamodule_without_source_field_or_left_source_raises(missing_kwarg):
    with pytest.raises(KeyError) as exc:
        kwargs = {
            "cluster_field": "cluster",
            "batch_size": 16,
            "eval_batch_size": 64,
            "source_field": "__source",
            "left_source": "foo",
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
        del kwargs[missing_kwarg]
        cli._build_datamodule(
            train_record_dict=mock.Mock(),
            valid_record_dict=mock.Mock(),
            test_record_dict=mock.Mock(),
            record_numericalizer=mock.Mock(),
            kwargs=kwargs,
        )
    assert 'must provide BOTH "source_field" and "left_source"' in str(exc)


@pytest.mark.parametrize("mode", ["linkage", "deduplication"])
def test_build_model(mode):
    if mode == "linkage":
        expected_model_cls = LinkageEmbed
    else:
        expected_model_cls = EntityEmbed

    mock_record_numericalizer = mock.MagicMock()

    with mock.patch(f"entity_embed.cli.{expected_model_cls.__name__}") as mock_model:
        cli._build_model(
            mock_record_numericalizer,
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
                **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
            },
        )

    mock_model.assert_called_once_with(
        **{
            "record_numericalizer": mock_record_numericalizer,
            "embedding_size": 100,
            "learning_rate": 0.2,
            "ann_k": 10,
            "sim_threshold_list": (0.2, 0.4, 0.6, 0.8),
            "index_build_kwargs": {"ef_construction": 128, "n_threads": 4},
            "index_search_kwargs": {"ef_search": -1, "n_threads": 4},
            **({"source_field": "__source", "left_source": "foo"} if mode == "linkage" else {}),
        }
    )
