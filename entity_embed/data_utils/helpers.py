import json

from .numericalizer import NumericalizeInfo, RowNumericalizer


class AttrInfoDictParser:
    @classmethod
    def from_json(cls, attr_info_json_file_obj, row_dict=None):
        attr_info_dict = json.load(attr_info_json_file_obj)
        return cls.from_dict(attr_info_dict, row_dict=row_dict)

    @classmethod
    def from_dict(cls, attr_info_dict, row_dict=None):
        for attr, numericalize_info in list(attr_info_dict.items()):
            if not numericalize_info:
                raise ValueError(
                    f'Please set the value of "{attr}" in attr_info_dict, {numericalize_info}'
                )
            attr_info_dict[attr] = NumericalizeInfo(**numericalize_info)
        return RowNumericalizer(attr_info_dict, row_dict=row_dict)
