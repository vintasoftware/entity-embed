import collections
import json
import tempfile

import mock
import n2  # noqa: F401
import pytest
from entity_embed.data_utils.helpers import AttrInfoDictParser
from entity_embed.data_utils.numericalizer import FieldType, NumericalizeInfo, RowNumericalizer
from torchtext.vocab import Vocab

EXPECTED_DEFAULT_ALPHABET = list(
    "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)


def _validate_base_row_numericalizer(row_numericalizer):
    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_info_dict = row_numericalizer.attr_info_dict
    assert list(parsed_attr_info_dict.keys()) == ["name"]

    name_attr_info = parsed_attr_info_dict["name"]
    assert isinstance(name_attr_info, NumericalizeInfo)
    assert name_attr_info.attr == "name"

    # Assert values were converted from str into proper types
    assert name_attr_info.field_type == FieldType.MULTITOKEN
    assert isinstance(name_attr_info.tokenizer, collections.Callable)

    # Assert max_str_len was computed
    assert isinstance(name_attr_info.max_str_len, int)

    # Assert non-provided keys were added with the correct default values
    assert name_attr_info.alphabet == EXPECTED_DEFAULT_ALPHABET
    assert name_attr_info.vocab is None
    assert name_attr_info.n_channels == 8
    assert name_attr_info.embed_dropout_p == 0.2
    assert name_attr_info.use_attention
    assert not name_attr_info.use_mask


def test_row_numericalizer_parse_from_dict():
    attr_info_dict = {
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

    row_numericalizer = AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)
    _validate_base_row_numericalizer(row_numericalizer)


def test_row_numericalizer_parse_from_json_file():
    attr_info_dict = {
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

    with tempfile.NamedTemporaryFile("w+") as f:
        json.dump(attr_info_dict, f)
        f.seek(0)  # Must move the pointer back to beginning since we aren't re-opening the file
        row_numericalizer = AttrInfoDictParser.from_json(f, row_dict=row_dict)
        _validate_base_row_numericalizer(row_numericalizer)


def test_row_numericalizer_parse_raises_when_attr_info_is_empty():
    attr_info_dict = {
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
        AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)


def test_row_numericalizer_parse_raises_when_row_dict_and_max_str_len_are_none():
    attr_info_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
    }

    with pytest.raises(ValueError):
        AttrInfoDictParser.from_dict(attr_info_dict)


def test_attr_with_wrong_field_type_raises():
    attr_info_dict = {
        "name": {
            "field_type": "FOO_TYPE",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
    }

    with pytest.raises(KeyError):
        AttrInfoDictParser.from_dict(attr_info_dict)


@mock.patch("entity_embed.data_utils.helpers.Vocab.load_vectors")
def test_row_numericalizer_parse_with_attr_with_semantic_field_type(mock_load_vectors):
    attr_info_dict = {
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

    row_numericalizer = AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)

    mock_load_vectors.assert_called_once_with("fasttext.en.300d")
    name_attr_info = row_numericalizer.attr_info_dict["name"]
    assert name_attr_info.attr == "name"
    assert name_attr_info.max_str_len is None
    assert isinstance(name_attr_info.vocab, Vocab)


def test_attr_with_semantic_field_type_without_vocab_raises():
    attr_info_dict = {
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
        AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)


def test_row_numericalizer_parse_multiple_attr_for_same_source_attr():
    attr_info_dict = {
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

    row_numericalizer = AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)

    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_info_dict = row_numericalizer.attr_info_dict
    assert list(parsed_attr_info_dict.keys()) == ["name_multitoken", "name_string"]

    name_multitoken_attr_info = parsed_attr_info_dict["name_multitoken"]
    assert isinstance(name_multitoken_attr_info, NumericalizeInfo)
    assert name_multitoken_attr_info.attr == "name"
    assert name_multitoken_attr_info.field_type == FieldType.MULTITOKEN

    name_string_attr_info = parsed_attr_info_dict["name_string"]
    assert isinstance(name_string_attr_info, NumericalizeInfo)
    assert name_string_attr_info.attr == "name"
    assert name_string_attr_info.field_type == FieldType.STRING


def test_row_numericalizer_parse_multiple_attr_without_source_attr_raises():
    attr_info_dict = {
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
        AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)


@mock.patch("entity_embed.data_utils.helpers.Vocab.load_vectors")
def test_row_numericalizer_parse_multiple_attr_for_same_source_attr_semantic_field_type(
    mock_load_vectors,
):
    attr_info_dict = {
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

    row_numericalizer = AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)
    assert isinstance(row_numericalizer, RowNumericalizer)

    mock_load_vectors.assert_has_calls(
        [
            mock.call("fasttext.en.300d"),
            mock.call("fasttext.en.300d"),
        ]
    )

    parsed_attr_info_dict = row_numericalizer.attr_info_dict
    assert list(parsed_attr_info_dict.keys()) == ["name_multitoken", "name_string"]

    name_multitoken_attr_info = parsed_attr_info_dict["name_multitoken"]
    assert isinstance(name_multitoken_attr_info, NumericalizeInfo)
    assert name_multitoken_attr_info.attr == "name"
    assert name_multitoken_attr_info.field_type == FieldType.SEMANTIC_MULTITOKEN
    assert name_multitoken_attr_info.max_str_len is None
    assert isinstance(name_multitoken_attr_info.vocab, Vocab)

    name_string_attr_info = parsed_attr_info_dict["name_string"]
    assert isinstance(name_string_attr_info, NumericalizeInfo)
    assert name_string_attr_info.attr == "name"
    assert name_string_attr_info.field_type == FieldType.SEMANTIC_STRING
    assert name_string_attr_info.max_str_len is None
    assert isinstance(name_string_attr_info.vocab, Vocab)


def test_row_numericalizer_parse_multiple_attr_for_same_source_attr_semantic_field_type_raises():
    attr_info_dict = {
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
        AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)
