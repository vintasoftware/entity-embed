import json
import logging
from importlib import import_module

from torchtext.vocab import Vocab

from .numericalizer import (
    AVAILABLE_VOCABS,
    DEFAULT_ALPHABET,
    AttrConfig,
    FieldType,
    MultitokenNumericalizer,
    RowNumericalizer,
    SemanticMultitokenNumericalizer,
    SemanticStringNumericalizer,
    StringNumericalizer,
)
from .utils import compute_max_str_len, compute_vocab_counter

logger = logging.getLogger(__name__)


def _import_function(function_dotted_path):
    module_dotted_path, function_name = function_dotted_path.rsplit(".", 1)
    module = import_module(module_dotted_path)
    return getattr(module, function_name)


class AttrConfigDictParser:
    @classmethod
    def from_json(cls, attr_config_json_file_obj, row_list):
        attr_config_dict = json.load(attr_config_json_file_obj)
        return cls.from_dict(attr_config_dict, row_list=row_list)

    @classmethod
    def from_dict(cls, attr_config_dict, row_list):
        parsed_attr_config_dict = {}
        attr_to_numericalizer = {}
        found_vocab = None
        for attr, attr_config in list(attr_config_dict.items()):
            if not attr_config:
                raise ValueError(f'Please set the value of "{attr}" in attr_config_dict')
            # Validate if all vocabs used are the same,
            # to ensure embedding sizes match
            current_vocab = attr_config.get("vocab")
            if current_vocab:
                if not found_vocab:
                    found_vocab = current_vocab
                elif found_vocab != current_vocab:
                    raise ValueError(
                        "Found more than one vocab on attr_config_dict, please "
                        "use a single vocab for the whole attr_config_dict, "
                        f'"{found_vocab}" != "{current_vocab}"'
                    )
            attr_config = cls._parse_attr_config(attr, attr_config, row_list=row_list)
            parsed_attr_config_dict[attr] = attr_config
            attr_to_numericalizer[attr] = cls._build_attr_numericalizer(attr, attr_config)
        return RowNumericalizer(parsed_attr_config_dict, attr_to_numericalizer)

    @classmethod
    def _parse_attr_config(cls, attr, attr_config, row_list):
        field_type = FieldType[attr_config["field_type"]]
        tokenizer = _import_function(attr_config.get("tokenizer", "entity_embed.default_tokenizer"))
        alphabet = attr_config.get("alphabet", DEFAULT_ALPHABET)
        max_str_len = attr_config.get("max_str_len")
        vocab = None

        # Check if there's a source_attr defined on the attr_config,
        # useful when we want to have multiple AttrConfig for the same attr
        source_attr = attr_config.get("source_attr", attr)

        # Compute vocab if necessary
        if field_type in (FieldType.SEMANTIC_STRING, FieldType.SEMANTIC_MULTITOKEN):
            vocab_type = attr_config.get("vocab")
            if vocab_type is None:
                raise ValueError(
                    "Please set a torchtext pretrained vocab to use. "
                    f"Available ones are: {AVAILABLE_VOCABS}"
                )
            try:
                vocab_counter = compute_vocab_counter(
                    attr_val_gen=(row[source_attr] for row in row_list),
                    tokenizer=tokenizer,
                )
            except KeyError:
                raise ValueError(
                    f"Cannot compute vocab_counter for attr={source_attr}. "
                    f"Please make sure that attr={attr} is a key in every "
                    "row of row_list or define source_attr in "
                    "attr_config if you wish to use a override "
                    "an attr name."
                )
            vocab = Vocab(vocab_counter)
            vocab.load_vectors(vocab_type)

        # Compute max_str_len if necessary
        if field_type in (FieldType.STRING, FieldType.MULTITOKEN) and (max_str_len is None):
            logger.info(f"For attr={attr}, computing actual max_str_len")
            is_multitoken = field_type in (FieldType.MULTITOKEN, FieldType.SEMANTIC_MULTITOKEN)
            try:
                actual_max_str_len = compute_max_str_len(
                    attr_val_gen=(row[source_attr] for row in row_list),
                    is_multitoken=is_multitoken,
                    tokenizer=tokenizer,
                )
            except KeyError:
                raise ValueError(
                    f"Cannot compute max_str_len for attr={source_attr}. "
                    f"Please make sure that attr={attr} is a key in every "
                    "row of row_list or define source_attr in "
                    "attr_config if you wish to use a override "
                    "an attr name."
                )
            if max_str_len is None:
                logger.info(f"For attr={attr}, using actual_max_str_len={actual_max_str_len}")
                max_str_len = actual_max_str_len

        n_channels = attr_config.get("n_channels", 8)
        embed_dropout_p = attr_config.get("embed_dropout_p", 0.2)
        use_attention = attr_config.get("use_attention", True)

        return AttrConfig(
            source_attr=source_attr,
            field_type=field_type,
            tokenizer=tokenizer,
            alphabet=alphabet,
            max_str_len=max_str_len,
            vocab=vocab,
            n_channels=n_channels,
            embed_dropout_p=embed_dropout_p,
            use_attention=use_attention,
        )

    @classmethod
    def _build_attr_numericalizer(cls, attr, attr_config: AttrConfig):
        field_type = attr_config.field_type

        field_type_to_numericalizer_cls = {
            FieldType.STRING: StringNumericalizer,
            FieldType.MULTITOKEN: MultitokenNumericalizer,
            FieldType.SEMANTIC_STRING: SemanticStringNumericalizer,
            FieldType.SEMANTIC_MULTITOKEN: SemanticMultitokenNumericalizer,
        }
        numericalizer_cls = field_type_to_numericalizer_cls.get(field_type)
        if numericalizer_cls is None:
            raise ValueError(f"Unexpected field_type={field_type}")  # pragma: no cover

        return numericalizer_cls(attr=attr, attr_config=attr_config)
