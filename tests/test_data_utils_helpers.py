import collections
import json
import tempfile

import mock
import n2  # noqa: F401
import pytest
from entity_embed.data_utils.field_config_parser import FieldConfigDictParser
from entity_embed.data_utils.numericalizer import FieldConfig, FieldType, RecordNumericalizer
from torchtext.vocab import Vocab

EXPECTED_DEFAULT_ALPHABET = list(
    "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)


def _validate_base_record_numericalizer(record_numericalizer):
    assert isinstance(record_numericalizer, RecordNumericalizer)

    parsed_field_config_dict = record_numericalizer.field_config_dict
    assert list(parsed_field_config_dict.keys()) == ["name"]

    name_field_config = parsed_field_config_dict["name"]
    assert isinstance(name_field_config, FieldConfig)
    assert name_field_config.key == "name"

    # Assert values were converted from str into proper types
    assert name_field_config.field_type == FieldType.MULTITOKEN
    assert isinstance(name_field_config.tokenizer, collections.abc.Callable)

    # Assert max_str_len was computed
    assert isinstance(name_field_config.max_str_len, int)

    # Assert non-provided keys were added with the correct default values
    assert name_field_config.alphabet == EXPECTED_DEFAULT_ALPHABET
    assert name_field_config.vocab is None
    assert name_field_config.n_channels == 8
    assert name_field_config.embed_dropout_p == 0.2
    assert name_field_config.use_attention


def test_record_numericalizer_parse_from_dict():
    field_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict, record_list=record_dict.values()
    )
    _validate_base_record_numericalizer(record_numericalizer)


def test_record_numericalizer_parse_from_json_file():
    field_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    record_list = [
        {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    ]

    with tempfile.NamedTemporaryFile("w+") as f:
        json.dump(field_config_dict, f)
        f.seek(0)  # Must move the pointer back to beginning since we aren't re-opening the file
        record_numericalizer = FieldConfigDictParser.from_json(f, record_list=record_list)
        _validate_base_record_numericalizer(record_numericalizer)


def test_record_numericalizer_parse_raises_when_field_config_is_empty():
    field_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
        "foo": {},
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    with pytest.raises(ValueError):
        FieldConfigDictParser.from_dict(field_config_dict, record_list=record_dict.values())


def test_field_with_wrong_field_type_raises():
    field_config_dict = {
        "name": {
            "field_type": "FOO_TYPE",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
    }

    with pytest.raises(KeyError):
        FieldConfigDictParser.from_dict(field_config_dict, record_list=[{"name": "foo"}])


@mock.patch("entity_embed.data_utils.field_config_parser.Vocab.load_vectors")
def test_record_numericalizer_parse_with_field_with_semantic_field_type(mock_load_vectors):
    field_config_dict = {
        "name": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        }
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict, record_list=record_dict.values()
    )

    mock_load_vectors.assert_called_once_with("fasttext.en.300d")
    name_field_config = record_numericalizer.field_config_dict["name"]
    assert name_field_config.key == "name"
    assert name_field_config.max_str_len is None
    assert isinstance(name_field_config.vocab, Vocab)


@mock.patch("entity_embed.data_utils.field_config_parser.Vocab.load_vectors")
def test_field_config_dict_with_different_vocab_types_raises(mock_load_vectors):
    field_config_dict = {
        "title": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
        "artist": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "glove.6B.50d",
        },
    }

    record_dict = {
        "1": {
            "id": "1",
            "title": "foo product",
            "artist": "foo artist",
        },
        "2": {
            "id": "2",
            "title": "the foo product from world",
            "artist": "artist",
        },
    }

    with pytest.raises(ValueError):
        FieldConfigDictParser.from_dict(field_config_dict, record_list=record_dict.values())

    mock_load_vectors.assert_called_once_with("fasttext.en.300d")


def test_field_with_semantic_field_type_without_vocab_raises():
    field_config_dict = {
        "name": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": None,
        }
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    with pytest.raises(ValueError):
        FieldConfigDictParser.from_dict(field_config_dict, record_list=record_dict.values())


def test_record_numericalizer_parse_multiple_field_for_same_key():
    field_config_dict = {
        "name_multitoken": {
            "key": "name",
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
        },
        "name_string": {
            "key": "name",
            "field_type": "STRING",
            "tokenizer": "entity_embed.default_tokenizer",
        },
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict, record_list=record_dict.values()
    )

    assert isinstance(record_numericalizer, RecordNumericalizer)

    parsed_field_config_dict = record_numericalizer.field_config_dict
    assert list(parsed_field_config_dict.keys()) == ["name_multitoken", "name_string"]

    name_multitoken_field_config = parsed_field_config_dict["name_multitoken"]
    assert isinstance(name_multitoken_field_config, FieldConfig)
    assert name_multitoken_field_config.key == "name"
    assert name_multitoken_field_config.field_type == FieldType.MULTITOKEN

    name_string_field_config = parsed_field_config_dict["name_string"]
    assert isinstance(name_string_field_config, FieldConfig)
    assert name_string_field_config.key == "name"
    assert name_string_field_config.field_type == FieldType.STRING


def test_record_numericalizer_parse_multiple_field_with_key():
    field_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
        },
        "name_string": {
            "key": "name",
            "field_type": "STRING",
            "tokenizer": "entity_embed.default_tokenizer",
        },
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict, record_list=record_dict.values()
    )

    assert isinstance(record_numericalizer, RecordNumericalizer)

    parsed_field_config_dict = record_numericalizer.field_config_dict
    assert list(parsed_field_config_dict.keys()) == ["name", "name_string"]

    name_field_config = parsed_field_config_dict["name"]
    assert isinstance(name_field_config, FieldConfig)
    assert name_field_config.key == "name"
    assert name_field_config.field_type == FieldType.MULTITOKEN

    name_string_field_config = parsed_field_config_dict["name_string"]
    assert isinstance(name_string_field_config, FieldConfig)
    assert name_string_field_config.key == "name"
    assert name_string_field_config.field_type == FieldType.STRING


def test_record_numericalizer_parse_multiple_field_without_key_raises():
    field_config_dict = {
        "name_multitoken": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
        },
        "name_string": {
            "key": "name",
            "field_type": "STRING",
            "tokenizer": "entity_embed.default_tokenizer",
        },
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    with pytest.raises(ValueError):
        FieldConfigDictParser.from_dict(field_config_dict, record_list=record_dict.values())


@mock.patch("entity_embed.data_utils.field_config_parser.Vocab.load_vectors")
def test_record_numericalizer_parse_multiple_field_for_same_key_semantic_field_type(
    mock_load_vectors,
):
    field_config_dict = {
        "name_multitoken": {
            "key": "name",
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
        "name_string": {
            "key": "name",
            "field_type": "SEMANTIC_STRING",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict, record_list=record_dict.values()
    )
    assert isinstance(record_numericalizer, RecordNumericalizer)

    mock_load_vectors.assert_has_calls(
        [
            mock.call("fasttext.en.300d"),
            mock.call("fasttext.en.300d"),
        ]
    )

    parsed_field_config_dict = record_numericalizer.field_config_dict
    assert list(parsed_field_config_dict.keys()) == ["name_multitoken", "name_string"]

    name_multitoken_field_config = parsed_field_config_dict["name_multitoken"]
    assert isinstance(name_multitoken_field_config, FieldConfig)
    assert name_multitoken_field_config.key == "name"
    assert name_multitoken_field_config.field_type == FieldType.SEMANTIC_MULTITOKEN
    assert name_multitoken_field_config.max_str_len is None
    assert isinstance(name_multitoken_field_config.vocab, Vocab)

    name_string_field_config = parsed_field_config_dict["name_string"]
    assert isinstance(name_string_field_config, FieldConfig)
    assert name_string_field_config.key == "name"
    assert name_string_field_config.field_type == FieldType.SEMANTIC_STRING
    assert name_string_field_config.max_str_len is None
    assert isinstance(name_string_field_config.vocab, Vocab)


def test_record_numericalizer_parse_multiple_field_for_same_key_semantic_field_type_raises():
    field_config_dict = {
        "name_multitoken": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
        "name_string": {
            "key": "name",
            "field_type": "SEMANTIC_STRING",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
    }

    record_dict = {
        "1": {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        "2": {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }

    with pytest.raises(ValueError):
        FieldConfigDictParser.from_dict(field_config_dict, record_list=record_dict.values())
