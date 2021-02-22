import collections

import pytest
from entity_embed.data_utils.helpers import RowNumericalizerParser
from entity_embed.data_utils.numericalizer import FieldType, NumericalizeInfo, RowNumericalizer

DEFAULT_ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")


def test_row_numericalizer_parse_from_dict():
    attr_info_dict = {
        "name": {
            "field_type": "MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "max_str_len": None,
        }
    }
    row_numericalizer = RowNumericalizerParser.from_dict(attr_info_dict)
    assert isinstance(row_numericalizer, RowNumericalizer)

    parsed_attr_info_dict = row_numericalizer.attr_info_dict
    assert parsed_attr_info_dict.keys() == ["name"]
    assert isinstance(parsed_attr_info_dict["name"], NumericalizeInfo)

    # Assert values were converted from str into proper types
    assert parsed_attr_info_dict["name"]["field_type"] == FieldType.MULTITOKEN
    assert isinstance(parsed_attr_info_dict["name"]["tokenizer"], collections.Callable)

    # Assert max_str_len was computed
    assert parsed_attr_info_dict["name"]["max_str_len"] is not None

    # Assert non-provided keys were added with the correct default values
    assert parsed_attr_info_dict["name"]["alphabet"] == DEFAULT_ALPHABET
    assert parsed_attr_info_dict["name"]["vocab"] is None


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
        RowNumericalizerParser.from_dict(attr_info_dict)


def test_row_numericalizer_parse_from_json_file():
    pass
