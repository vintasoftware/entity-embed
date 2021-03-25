import collections
import json
import tempfile

import mock
import n2  # noqa: F401
import pytest
from entity_embed.data_utils.attr_config_parser import AttrConfigDictParser
from entity_embed.data_utils.numericalizer import AttrConfig, FieldType, RowNumericalizer
from torchtext.vocab import Vocab

EXPECTED_DEFAULT_ALPHABET = list(
    "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)


def _validate_base_row_numericalizer(row_numericalizer):
    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_config_dict = row_numericalizer.attr_config_dict
    assert list(parsed_attr_config_dict.keys()) == ["name"]

    name_attr_config = parsed_attr_config_dict["name"]
    assert isinstance(name_attr_config, AttrConfig)
    assert name_attr_config.source_attr == "name"

    # Assert values were converted from str into proper types
    assert name_attr_config.field_type == FieldType.MULTITOKEN
    assert isinstance(name_attr_config.tokenizer, collections.Callable)

    # Assert max_str_len was computed
    assert isinstance(name_attr_config.max_str_len, int)

    # Assert non-provided keys were added with the correct default values
    assert name_attr_config.alphabet == EXPECTED_DEFAULT_ALPHABET
    assert name_attr_config.vocab is None
    assert name_attr_config.n_channels == 8
    assert name_attr_config.embed_dropout_p == 0.2
    assert name_attr_config.use_attention


def test_row_numericalizer_parse_from_dict():
    attr_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    row_dict = {
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

    row_numericalizer = AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())
    _validate_base_row_numericalizer(row_numericalizer)


def test_row_numericalizer_parse_from_json_file():
    attr_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }

    row_list = [
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
        json.dump(attr_config_dict, f)
        f.seek(0)  # Must move the pointer back to beginning since we aren't re-opening the file
        row_numericalizer = AttrConfigDictParser.from_json(f, row_list=row_list)
        _validate_base_row_numericalizer(row_numericalizer)


def test_row_numericalizer_parse_raises_when_attr_config_is_empty():
    attr_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
        "foo": {},
    }

    row_dict = {
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
        AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())


def test_attr_with_wrong_field_type_raises():
    attr_config_dict = {
        "name": {
            "field_type": "FOO_TYPE",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
    }

    with pytest.raises(KeyError):
        AttrConfigDictParser.from_dict(attr_config_dict, row_list=[{"name": "foo"}])


@mock.patch("entity_embed.data_utils.attr_config_parser.Vocab.load_vectors")
def test_row_numericalizer_parse_with_attr_with_semantic_field_type(mock_load_vectors):
    attr_config_dict = {
        "name": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        }
    }

    row_dict = {
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

    row_numericalizer = AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())

    mock_load_vectors.assert_called_once_with("fasttext.en.300d")
    name_attr_config = row_numericalizer.attr_config_dict["name"]
    assert name_attr_config.source_attr == "name"
    assert name_attr_config.max_str_len is None
    assert isinstance(name_attr_config.vocab, Vocab)


@mock.patch("entity_embed.data_utils.attr_config_parser.Vocab.load_vectors")
def test_attr_config_dict_with_different_vocab_types_raises(mock_load_vectors):
    attr_config_dict = {
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

    row_dict = {
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
        AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())

    mock_load_vectors.assert_called_once_with("fasttext.en.300d")


def test_attr_with_semantic_field_type_without_vocab_raises():
    attr_config_dict = {
        "name": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": None,
        }
    }

    row_dict = {
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
        AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())


def test_row_numericalizer_parse_multiple_attr_for_same_source_attr():
    attr_config_dict = {
        "name_multitoken": {
            "source_attr": "name",
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
        },
        "name_string": {
            "source_attr": "name",
            "field_type": "STRING",
            "tokenizer": "entity_embed.default_tokenizer",
        },
    }

    row_dict = {
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

    row_numericalizer = AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())

    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_config_dict = row_numericalizer.attr_config_dict
    assert list(parsed_attr_config_dict.keys()) == ["name_multitoken", "name_string"]

    name_multitoken_attr_config = parsed_attr_config_dict["name_multitoken"]
    assert isinstance(name_multitoken_attr_config, AttrConfig)
    assert name_multitoken_attr_config.source_attr == "name"
    assert name_multitoken_attr_config.field_type == FieldType.MULTITOKEN

    name_string_attr_config = parsed_attr_config_dict["name_string"]
    assert isinstance(name_string_attr_config, AttrConfig)
    assert name_string_attr_config.source_attr == "name"
    assert name_string_attr_config.field_type == FieldType.STRING


def test_row_numericalizer_parse_multiple_attr_with_source_attr_as_key():
    attr_config_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
        },
        "name_string": {
            "source_attr": "name",
            "field_type": "STRING",
            "tokenizer": "entity_embed.default_tokenizer",
        },
    }

    row_dict = {
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

    row_numericalizer = AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())

    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_config_dict = row_numericalizer.attr_config_dict
    assert list(parsed_attr_config_dict.keys()) == ["name", "name_string"]

    name_attr_config = parsed_attr_config_dict["name"]
    assert isinstance(name_attr_config, AttrConfig)
    assert name_attr_config.source_attr == "name"
    assert name_attr_config.field_type == FieldType.MULTITOKEN

    name_string_attr_config = parsed_attr_config_dict["name_string"]
    assert isinstance(name_string_attr_config, AttrConfig)
    assert name_string_attr_config.source_attr == "name"
    assert name_string_attr_config.field_type == FieldType.STRING


def test_row_numericalizer_parse_multiple_attr_without_source_attr_raises():
    attr_config_dict = {
        "name_multitoken": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
        },
        "name_string": {
            "source_attr": "name",
            "field_type": "STRING",
            "tokenizer": "entity_embed.default_tokenizer",
        },
    }

    row_dict = {
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
        AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())


@mock.patch("entity_embed.data_utils.attr_config_parser.Vocab.load_vectors")
def test_row_numericalizer_parse_multiple_attr_for_same_source_attr_semantic_field_type(
    mock_load_vectors,
):
    attr_config_dict = {
        "name_multitoken": {
            "source_attr": "name",
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
        "name_string": {
            "source_attr": "name",
            "field_type": "SEMANTIC_STRING",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
    }

    row_dict = {
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

    row_numericalizer = AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())
    assert isinstance(row_numericalizer, RowNumericalizer)

    mock_load_vectors.assert_has_calls(
        [
            mock.call("fasttext.en.300d"),
            mock.call("fasttext.en.300d"),
        ]
    )

    parsed_attr_config_dict = row_numericalizer.attr_config_dict
    assert list(parsed_attr_config_dict.keys()) == ["name_multitoken", "name_string"]

    name_multitoken_attr_config = parsed_attr_config_dict["name_multitoken"]
    assert isinstance(name_multitoken_attr_config, AttrConfig)
    assert name_multitoken_attr_config.source_attr == "name"
    assert name_multitoken_attr_config.field_type == FieldType.SEMANTIC_MULTITOKEN
    assert name_multitoken_attr_config.max_str_len is None
    assert isinstance(name_multitoken_attr_config.vocab, Vocab)

    name_string_attr_config = parsed_attr_config_dict["name_string"]
    assert isinstance(name_string_attr_config, AttrConfig)
    assert name_string_attr_config.source_attr == "name"
    assert name_string_attr_config.field_type == FieldType.SEMANTIC_STRING
    assert name_string_attr_config.max_str_len is None
    assert isinstance(name_string_attr_config.vocab, Vocab)


def test_row_numericalizer_parse_multiple_attr_for_same_source_attr_semantic_field_type_raises():
    attr_config_dict = {
        "name_multitoken": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
        "name_string": {
            "source_attr": "name",
            "field_type": "SEMANTIC_STRING",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "fasttext.en.300d",
        },
    }

    row_dict = {
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
        AttrConfigDictParser.from_dict(attr_config_dict, row_list=row_dict.values())
