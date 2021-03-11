import csv
import json
import os
import tempfile

import mock
import pytest
from click.testing import CliRunner
from entity_embed.cli import main


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
        },
        {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
        },
    ]

    with tempfile.NamedTemporaryFile("w", delete=False) as attr_info_json_file:
        json.dump(attr_info_dict, attr_info_json_file)

    with tempfile.NamedTemporaryFile("w", delete=False) as row_dict_csv_file:
        csv_writer = csv.writer(row_dict_csv_file)
        csv_writer.writerow(row_dict_values[0].keys())
        csv_writer.writerows(row_dict_values)

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
    attr_info_json_filepath, row_dict_csv_filepath = datamodule_files

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-attr_info_json_filepath",
            attr_info_json_filepath,
            "-row_dict_csv_filepath",
            row_dict_csv_filepath,
            "-cluster_attr",
            "name",
            "-batch_size",
            10,
            "-row_batch_size",
            10,
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
        ],
    )

    assert result.exit_code == 0

    mock_trainer = mock_build_trainer.return_value
    mock_build_trainer.assert_called_once_with()
    mock_trainer.fit.assert_called_once()
    mock_validate_best.assert_called_once_with(mock_trainer)
    mock_trainer.test.assert_called_once_with(ckpt_path="best", verbose=False)


def test_build_trainer():
    pass
