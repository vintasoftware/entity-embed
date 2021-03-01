import json
import logging

from torchtext.vocab import Vocab

from .numericalizer import (
    FieldType,
    MultitokenNumericalizer,
    NumericalizeInfo,
    RowNumericalizer,
    SemanticMultitokenNumericalizer,
    SemanticStringNumericalizer,
    StringNumericalizer,
)
from .utils import compute_max_str_len, compute_vocab_counter, import_function

logger = logging.getLogger(__name__)


DEFAULT_ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")


AVAILABLE_VOCABS = [
    "charngram.100d",
    "fasttext.en.300d",
    "fasttext.simple.300d",
    "glove.42B.300d",
    "glove.840B.300d",
    "glove.twitter.27B.25d",
    "glove.twitter.27B.50d",
    "glove.twitter.27B.100d",
    "glove.twitter.27B.200d",
    "glove.6B.50d",
    "glove.6B.100d",
    "glove.6B.200d",
    "glove.6B.300d",
]


class AttrInfoDictParser:
    @classmethod
    def from_json(cls, attr_info_json_file_obj, row_dict=None):
        attr_info_dict = json.load(attr_info_json_file_obj)
        return cls.from_dict(attr_info_dict, row_dict=row_dict)

    @classmethod
    def from_dict(cls, attr_info_dict, row_dict=None):
        attr_info_dict = dict(attr_info_dict)  # make a copy
        attr_to_numericalizer = {}
        for attr, numericalize_info_dict in list(attr_info_dict.items()):
            if not numericalize_info_dict:
                raise ValueError(
                    f'Please set the value of "{attr}" in attr_info_dict, {numericalize_info_dict}'
                )
            numericalize_info = cls._parse_numericalize_info_dict(
                attr, numericalize_info_dict, row_dict=row_dict
            )
            attr_info_dict[attr] = numericalize_info
            attr_to_numericalizer[attr] = cls._build_attr_numericalizer(attr, numericalize_info)
        return RowNumericalizer(attr_info_dict, attr_to_numericalizer)

    @classmethod
    def _parse_numericalize_info_dict(cls, attr, numericalize_info_dict, row_dict=None):
        field_type = FieldType[numericalize_info_dict["field_type"]]
        tokenizer = import_function(
            numericalize_info_dict.get("tokenizer", "entity_embed.default_tokenizer")
        )
        alphabet = numericalize_info_dict.get("alphabet", DEFAULT_ALPHABET)
        max_str_len = numericalize_info_dict.get("max_str_len")
        vocab = None

        # Check if there's a source_attr defined on the numericalize_info_dict,
        # useful when we want to have multiple NumericalizeInfo for the same attr
        source_attr = numericalize_info_dict.get("source_attr", attr)

        # Compute vocab if necessary
        if field_type in (FieldType.SEMANTIC_STRING, FieldType.SEMANTIC_MULTITOKEN):
            if numericalize_info_dict.get("vocab") is None:
                raise ValueError(
                    "Please set a torchtext pretrained vocab to use. "
                    f"Available ones are: {AVAILABLE_VOCABS}"
                )
            try:
                vocab_counter = compute_vocab_counter(
                    attr_val_gen=(row[source_attr] for row in row_dict.values()),
                    tokenizer=tokenizer,
                )
            except KeyError:
                raise ValueError(
                    f"Cannot compute vocab_counter for attr={source_attr}. "
                    f"Please make sure that attr={attr} is a key in every "
                    "row of row_dict.values() or define source_attr in "
                    "numericalize_info_dict if you wish to use a override "
                    "an attr name."
                )
            vocab = Vocab(vocab_counter)
            vocab.load_vectors(numericalize_info_dict.get("vocab"))

        # Compute max_str_len if necessary
        if field_type in (FieldType.STRING, FieldType.MULTITOKEN) and (max_str_len is None):
            if row_dict is None:
                raise ValueError(
                    f"Cannot compute max_str_len for attr={attr}. "
                    "row_dict cannot be None if max_str_len is None. "
                    "Please set row_dict, a dictionary of id -> row with ALL your data "
                    "(train, test, valid). "
                    "Or call entity_embed.data_utils.compute_max_str_len "
                    "over ALL your data (train, test, valid) to compute max_str_len."
                )
            else:
                logger.info(f"For attr={attr}, computing actual max_str_len")
                is_multitoken = field_type in (FieldType.MULTITOKEN, FieldType.SEMANTIC_MULTITOKEN)
                try:
                    actual_max_str_len = compute_max_str_len(
                        attr_val_gen=(row[source_attr] for row in row_dict.values()),
                        is_multitoken=is_multitoken,
                        tokenizer=tokenizer,
                    )
                except KeyError:
                    raise ValueError(
                        f"Cannot compute max_str_len for attr={source_attr}. "
                        f"Please make sure that attr={attr} is a key in every "
                        "row of row_dict.values() or define source_attr in "
                        "numericalize_info_dict if you wish to use a override "
                        "an attr name."
                    )
                if max_str_len is None:
                    logger.info(f"For attr={attr}, using actual_max_str_len={actual_max_str_len}")
                    max_str_len = actual_max_str_len

        n_channels = numericalize_info_dict.get("n_channels", 8)
        embed_dropout_p = numericalize_info_dict.get("embed_dropout_p", 0.2)
        use_attention = numericalize_info_dict.get("use_attention", True)
        use_mask = numericalize_info_dict.get("use_mask", False)

        return NumericalizeInfo(
            source_attr=source_attr,
            field_type=field_type,
            tokenizer=tokenizer,
            alphabet=alphabet,
            max_str_len=max_str_len,
            vocab=vocab,
            n_channels=n_channels,
            embed_dropout_p=embed_dropout_p,
            use_attention=use_attention,
            use_mask=use_mask,
        )

    @classmethod
    def _build_attr_numericalizer(cls, attr, numericalize_info: NumericalizeInfo):
        field_type = numericalize_info.field_type

        field_type_to_numericalizer_cls = {
            FieldType.STRING: StringNumericalizer,
            FieldType.MULTITOKEN: MultitokenNumericalizer,
            FieldType.SEMANTIC_STRING: SemanticStringNumericalizer,
            FieldType.SEMANTIC_MULTITOKEN: SemanticMultitokenNumericalizer,
        }
        numericalizer_cls = field_type_to_numericalizer_cls.get(field_type)
        if numericalizer_cls is None:
            raise ValueError(f"Unexpected field_type={field_type}")  # pragma: no cover

        return numericalizer_cls(attr=attr, numericalize_info=numericalize_info)
