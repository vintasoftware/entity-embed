import json

from .numericalizer import NumericalizeInfo, RowNumericalizer


class RowNumericalizerParser:
    @classmethod
    def from_json(cls, attr_info_json_filepath, row_dict=None):
        with open(attr_info_json_filepath) as f:
            attr_info_dict = json.load(f)
        return cls.from_dict(attr_info_dict, row_dict=row_dict)

    @classmethod
    def from_dict(cls, attr_info_dict, row_dict=None):
        for attr, numericalize_info in list(attr_info_dict.items()):
            if not numericalize_info:
                raise ValueError(
                    f'Please set the value of "{attr}" in attr_info_dict, {numericalize_info}'
                )
            attr_info_dict[attr] = NumericalizeInfo(**numericalize_info)
        # For now on, one must use row_numericalizer instead of attr_info_dict,
        # because RowNumericalizer fills None values of alphabet and max_str_len.
        return RowNumericalizer(attr_info_dict, row_dict=row_dict)
