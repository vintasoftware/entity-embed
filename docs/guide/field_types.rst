.. _field_types:

===========
Field Types
===========

Entity Embed supports different field types that define how record fields are numericalized and encoded by Entity Embed's :ref:`Neural Network Architecture <nn_architecture>`. Field types enable you to choose the best architecture to solve your problem.

Currently only fields of strings are supported.

Field Config and Record Numericalizer
-------------------------------------

The ``field_config_dict`` holds the definition of the field types your network will use. For example::

    from entity_embed.data_utils.field_config_parser import DEFAULT_ALPHABET

    field_config_dict = {
        'title': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
            'max_str_len': None,  # compute
        },
        'title_semantic': {
            'key': 'title',
            'field_type': "SEMANTIC",
            'tokenizer': "entity_embed.default_tokenizer",
        },
        'artist': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
            'max_str_len': None,  # compute
        },
        'album': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
            'max_str_len': None,  # compute
        },
        'album_semantic': {
            'key': 'album',
            'field_type': "SEMANTIC",
            'tokenizer': "entity_embed.default_tokenizer",
        }
    }

With the ``field_config_dict``, we can get a ``record_numericalizer``. This object can convert the strings from your records into tensors for the neural network and it's necessary for building the datamodule and model instances::

    from entity_embed import FieldConfigDictParser

    record_numericalizer = FieldConfigDictParser.from_dict(
        field_config_dict,
        record_list=record_dict.values(),
    )

You can also read the field config from a JSON file object::

    with open("path-to-field-config.json", "r") as field_config_json_file_obj:
        record_numericalizer = FieldConfigDictParser.from_json(
            field_config_json_file_obj,
            record_list=record_dict.values(),
        )

Below you can learn more details about the available field types.

Syntactic Fields
----------------

Syntactic Fields capture best the **syntactic** features of the strings.

Syntactic Fields need an ``alphabet``. The default alphabet is a list with the ASCII numbers, letters, symbols and space. You can use any other alphabet if you need::

    >> from entity_embed.data_utils.field_config_parser import DEFAULT_ALPHABET

    >> DEFAULT_ALPHABET

    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
     'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
     '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<',
     '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']

You can also set a ``max_str_len`` to Syntactic Fields or leave it ``None``. When ``None``, the value is calculated by iterating over all data in ``record_list`` when you call ``FieldConfigDictParser.from_dict``. ``max_str_len`` is necessary for the neural network.

STRING
~~~~~~

Use this field type when the string is made of a single token or when the expected similar values will probably have very similar tokens. This field type produces embeddings that approximate strings with short edit-distances.

Use this in fields like: first/last name, number, other short strings, etc.

MULTITOKEN
~~~~~~~~~~

Use this field type when the string is made of multiple tokens. This field type produces embeddings that approximate strings with short edit-distances.

Use this in fields like: full name, company name, product name, etc.

Semantic Fields
---------------

Semantic Fields capture best the **semantic** features of the strings.

Semantic Fields need a ``vocab`` that defines which pre-trained embeddings to use. Available ones are the same of `torchtext <https://pytorch.org/text/stable/index.html>`_ library::

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


SEMANTIC_STRING
~~~~~~~~~~~~~~~

Use this field type when the string is made of a single token. This field type uses the pre-trained embeddings of ``vocab``.

Use this in fields like: product category, song genre, etc.

SEMANTIC
~~~~~~~~~~~~~~~~~~~

Use this field type when the string is made of multiple tokens. This field type uses the pre-trained embeddings of ``vocab``.

Use this in fields like: company name, product name, product description, etc.

Tokenizer
---------

MULTITOKEN and SEMANTIC fields need a tokenizer function that receives a string a returns a list of strings. The default tokenizer function is ``entity_embed.default_tokenizer``, which simply splits the string on all symbols::

    >> entity_embed.default_tokenizer("vinta-software_ltda 2021")

    ['vinta', '-', 'software', '_', 'ltda', '2021']

Multiple field types for same field
-----------------------------------

Use ``key`` to derive multiple field types from the same record field. When ``key`` is omitted, it's inferred from the field key on ``field_config_dict``::


    field_config_dict = {
        'title': {
            'field_type': "MULTITOKEN",
            'tokenizer': "entity_embed.default_tokenizer",
            'alphabet': DEFAULT_ALPHABET,
        },
        'title_semantic': {
            'key': 'title',
            'field_type': "SEMANTIC",
            'tokenizer': "entity_embed.default_tokenizer",
        }
    }

How the Neural Network processes the fields
-------------------------------------------

Check :ref:`Neural Network Architecture <nn_architecture>`.
