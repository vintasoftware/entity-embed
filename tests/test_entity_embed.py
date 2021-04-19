import mock
import pytest
import torch
from entity_embed import EntityEmbed
from entity_embed.data_utils.field_config_parser import FieldConfigDictParser

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


@pytest.fixture
def field_config_dict():
    return {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
        "name_semantic": {
            "key": "name",
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
    }


@pytest.fixture()
def record_list():
    return LABELED_RECORD_DICT_VALUES + UNLABELED_RECORD_DICT_VALUES


@pytest.fixture()
@mock.patch("entity_embed.data_utils.field_config_parser.Vocab.load_vectors")
def record_numericalizer(mock_load_vectors, field_config_dict, record_list):
    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict, record_list=record_list
    )
    record_numericalizer.field_config_dict["name_semantic"].vocab.vectors = torch.empty((1, 300))
    return record_numericalizer


def test_set_embedding_size_when_using_semantic_fields(record_numericalizer):
    with pytest.raises(ValueError) as excinfo:
        EntityEmbed(record_numericalizer=record_numericalizer, embedding_size=500)

    assert "Invalid embedding_size=500. Expected 300, due to semantic fields." in str(excinfo)


@mock.patch("pytorch_lightning.Trainer")
@mock.patch("entity_embed.EntityEmbed.load_from_checkpoint")
@mock.patch("entity_embed.entity_embed.TensorBoardLogger")
@mock.patch("entity_embed.entity_embed.ModelCheckpointMinEpochs")
@mock.patch("entity_embed.entity_embed.EarlyStoppingMinEpochs")
def test_fit(
    mock_early_stopping,
    mock_checkpoint,
    mock_tb_logger,
    mock_load,
    mock_trainer,
    record_numericalizer,
):
    model = EntityEmbed(record_numericalizer=record_numericalizer)
    datamodule = mock.Mock()
    best_model = EntityEmbed(record_numericalizer=record_numericalizer)  # other model, best model
    mock_load.return_value.to.return_value.blocker_net = best_model

    trainer = model.fit(
        datamodule=datamodule,
        min_epochs=3,
        max_epochs=50,
        check_val_every_n_epoch=2,
        early_stop_monitor="pair_entity_ratio_at_0.3",
        early_stop_min_delta=0.1,
        early_stop_patience=10,
        early_stop_mode="min",
        early_stop_verbose=False,
        model_save_top_k=2,
        model_save_dir="trained-models",
        model_save_verbose=True,
        tb_save_dir="tb-logs",
        tb_name="test-model",
    )
    mock_early_stopping.assert_called_once_with(
        min_epochs=3,
        monitor="pair_entity_ratio_at_0.3",
        min_delta=0.1,
        patience=10,
        mode="min",
        verbose=False,
    )
    mock_checkpoint.assert_called_once_with(
        min_epochs=3,
        monitor="pair_entity_ratio_at_0.3",
        save_top_k=2,
        mode="min",
        dirpath="trained-models",
        verbose=True,
    )
    mock_tb_logger.assert_called_once_with("tb-logs", name="test-model")
    mock_trainer.assert_called_once_with(
        gpus=1,
        min_epochs=3,
        max_epochs=50,
        check_val_every_n_epoch=2,
        callbacks=[mock_early_stopping.return_value, mock_checkpoint.return_value],
        reload_dataloaders_every_epoch=True,
        logger=mock_tb_logger.return_value,
    )
    mock_trainer.return_value.fit.assert_called_once_with(model, datamodule)
    assert model.blocker_net == best_model
    assert trainer == mock_trainer.return_value


@mock.patch("pytorch_lightning.Trainer")
@mock.patch("entity_embed.EntityEmbed.load_from_checkpoint")
@mock.patch("entity_embed.entity_embed.TensorBoardLogger")
@mock.patch("entity_embed.entity_embed.ModelCheckpointMinEpochs")
@mock.patch("entity_embed.entity_embed.EarlyStoppingMinEpochs")
def test_fit_with_only_tb_name_raises(
    mock_early_stopping,
    mock_checkpoint,
    mock_tb_logger,
    mock_load,
    mock_trainer,
    record_numericalizer,
):
    model = EntityEmbed(record_numericalizer=record_numericalizer)
    datamodule = mock.Mock()

    with pytest.raises(ValueError) as exc:
        model.fit(
            datamodule=datamodule,
            min_epochs=3,
            max_epochs=50,
            check_val_every_n_epoch=2,
            early_stop_monitor="pair_entity_ratio_at_0.3",
            early_stop_min_delta=0.1,
            early_stop_patience=10,
            early_stop_mode="min",
            early_stop_verbose=False,
            model_save_top_k=2,
            model_save_dir="trained-models",
            model_save_verbose=True,
            tb_save_dir=None,
            tb_name="test-model",
        )
    assert 'Please provide both "tb_name" and "tb_save_dir"' in str(exc)
