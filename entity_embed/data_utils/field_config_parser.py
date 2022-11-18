import json
import logging
from importlib import import_module

import torch
from torch import Tensor, nn
from torchtext.vocab import Vocab, Vectors, FastText
from torchtext.vocab import vocab as factory_vocab

from .numericalizer import (
    AVAILABLE_VOCABS,
    DEFAULT_ALPHABET,
    FieldConfig,
    FieldType,
    MultitokenNumericalizer,
    RecordNumericalizer,
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


class FieldConfigDictParser:
    @classmethod
    def from_json(cls, field_config_json_file_obj, record_list):
        field_config_dict = json.load(field_config_json_file_obj)
        return cls.from_dict(field_config_dict, record_list=record_list)

    @classmethod
    def from_dict(cls, field_config_dict, record_list):
        parsed_field_config_dict = {}
        field_to_numericalizer = {}
        found_vocab = None
        for field, field_config in list(field_config_dict.items()):
            if not field_config:
                raise ValueError(f'Please set the value of "{field}" in field_config_dict')
            # Validate if all vocabs used are the same,
            # to ensure embedding sizes match
            current_vocab = field_config.get("vocab")
            if current_vocab:
                if not found_vocab:
                    found_vocab = current_vocab
                elif found_vocab != current_vocab:
                    raise ValueError(
                        "Found more than one vocab on field_config_dict, please "
                        "use a single vocab for the whole field_config_dict, "
                        f'"{found_vocab}" != "{current_vocab}"'
                    )
            field_config = cls._parse_field_config(field, field_config, record_list=record_list)
            parsed_field_config_dict[field] = field_config
            field_to_numericalizer[field] = cls._build_field_numericalizer(field, field_config)
        return RecordNumericalizer(parsed_field_config_dict, field_to_numericalizer)

    @classmethod
    def _parse_field_config(cls, field, field_config, record_list):
        field_type = FieldType[field_config["field_type"]]
        tokenizer = _import_function(
            field_config.get("tokenizer", "entity_embed.default_tokenizer")
        )
        pre_processor = _import_function(
            field_config.get("pre_processor", "entity_embed.default_pre_processor")
        )
        multi_pre_processor = _import_function(
            field_config.get("multi_pre_processor", "entity_embed.default_pre_processor")
        )
        alphabet = field_config.get("alphabet", DEFAULT_ALPHABET)
        max_str_len = field_config.get("max_str_len")
        vocab = None
        vector_tensor = None

        # Check if there's a key defined on the field_config,
        # useful when we want to have multiple FieldConfig for the same field
        key = field_config.get("key", field)

        # Compute vocab if necessary
        if field_type in (FieldType.SEMANTIC_STRING, FieldType.SEMANTIC_MULTITOKEN):
            vocab_type = field_config.get("vocab")
            if vocab_type is None:
                raise ValueError(
                    "Please set a torchtext pretrained vocab to use. "
                    f"Available ones are: {AVAILABLE_VOCABS}"
                )
            try:
                vocab_counter = compute_vocab_counter(
                    field_val_gen=(record[key] for record in record_list),
                    tokenizer=tokenizer,
                )
            except KeyError:
                raise ValueError(
                    f"Cannot compute vocab_counter for field={key}. "
                    f"Please make sure that field={field} is a key in every "
                    "record of record_list or define key in "
                    "field_config if you wish to use a override "
                    "an field name."
                )

            if vocab_type in {"tx_embeddings_large.vec", "tx_embeddings.vec"}:
                vectors = Vectors(vocab_type, cache=".vector_cache")
            elif vocab_type == "fasttext":
                vectors = FastText("en")  # might need to add standard fasttext
            else:
                vocab.load_vectors(vocab_type)  # won't work

            # adding <unk> token
            unk_token = "<unk>"
            vocab = factory_vocab(vocab_counter, specials=[unk_token])
            # print(vocab["<unk>"])  # prints 0
            # make default index same as index of unk_token
            vocab.set_default_index(vocab[unk_token])
            # print(vocab["probably out of vocab"])  # prints 0

            # create vector tensor using tokens in vocab, order important
            vectors = [vectors]
            # device = torch.device("mps")
            tot_dim = sum(v.dim for v in vectors)
            # vector_tensor = torch.zeros(len(vocab), tot_dim)
            vector_tensor = torch.Tensor(len(vocab), tot_dim)

            for i, token in enumerate(vocab.get_itos()):
                start_dim = 0
                for v in vectors:
                    end_dim = start_dim + v.dim
                    vector_tensor[i][start_dim:end_dim] = v[token.strip()]
                    start_dim = end_dim
                assert start_dim == tot_dim

            logger.info(f"Vector tensor shape: {vector_tensor.shape}")
            assert len(vector_tensor) == len(vocab)

        # Compute max_str_len if necessary
        if field_type in (FieldType.STRING, FieldType.MULTITOKEN) and (max_str_len is None):
            logger.info(f"For field={field}, computing actual max_str_len")
            is_multitoken = field_type in (FieldType.MULTITOKEN, FieldType.SEMANTIC_MULTITOKEN)
            try:
                actual_max_str_len = compute_max_str_len(
                    field_val_gen=(record[key] for record in record_list),
                    is_multitoken=is_multitoken,
                    tokenizer=tokenizer,
                )
            except KeyError:
                raise ValueError(
                    f"Cannot compute max_str_len for field={key}. "
                    f"Please make sure that field={field} is a key in every "
                    "record of record_list or define key in "
                    "field_config if you wish to use a override "
                    "an field name."
                )
            if max_str_len is None:
                logger.info(f"For field={field}, using actual_max_str_len={actual_max_str_len}")
                max_str_len = actual_max_str_len

        n_channels = field_config.get("n_channels", 8)
        embed_dropout_p = field_config.get("embed_dropout_p", 0.2)
        use_attention = field_config.get("use_attention", True)

        return FieldConfig(
            key=key,
            field_type=field_type,
            tokenizer=tokenizer,
            pre_processor=pre_processor,
            multi_pre_processor=multi_pre_processor,
            alphabet=alphabet,
            max_str_len=max_str_len,
            vocab=vocab,
            vector_tensor=vector_tensor,
            n_channels=n_channels,
            embed_dropout_p=embed_dropout_p,
            use_attention=use_attention,
        )

    @classmethod
    def _build_field_numericalizer(cls, field, field_config: FieldConfig):
        field_type = field_config.field_type

        field_type_to_numericalizer_cls = {
            FieldType.STRING: StringNumericalizer,
            FieldType.MULTITOKEN: MultitokenNumericalizer,
            FieldType.SEMANTIC_STRING: SemanticStringNumericalizer,
            FieldType.SEMANTIC_MULTITOKEN: SemanticMultitokenNumericalizer,
        }
        numericalizer_cls = field_type_to_numericalizer_cls.get(field_type)
        if numericalizer_cls is None:
            raise ValueError(f"Unexpected field_type={field_type}")  # pragma: no cover

        return numericalizer_cls(field=field, field_config=field_config)
