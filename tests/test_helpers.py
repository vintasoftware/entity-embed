import collections
import json
import tempfile

import pytest
from entity_embed.data_utils.helpers import AttrInfoDictParser
from entity_embed.data_utils.numericalizer import FieldType, NumericalizeInfo, RowNumericalizer

DEFAULT_ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")


def _validate_row_numericalizer(row_numericalizer):
    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_info_dict = row_numericalizer.attr_info_dict
    assert list(parsed_attr_info_dict.keys()) == ["name"]

    name_attr_info = parsed_attr_info_dict["name"]
    assert isinstance(name_attr_info, NumericalizeInfo)

    # Assert values were converted from str into proper types
    assert name_attr_info.field_type == FieldType.MULTITOKEN
    assert isinstance(name_attr_info.tokenizer, collections.Callable)

    # Assert max_str_len was computed
    assert isinstance(name_attr_info.max_str_len, int)

    # Assert non-provided keys were added with the correct default values
    assert name_attr_info.alphabet == DEFAULT_ALPHABET
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
    _validate_row_numericalizer(row_numericalizer)


def test_row_numericalizer_parse_from_dict_raises():
    attr_info_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        },
        "foo": {},
    }
    with pytest.raises(ValueError):
        AttrInfoDictParser.from_dict(attr_info_dict)


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
        _validate_row_numericalizer(row_numericalizer)
