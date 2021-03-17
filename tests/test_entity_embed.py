import logging

import mock
from entity_embed.data_utils.helpers import AttrInfoDictParser
from entity_embed.entity_embed import DeduplicationDataModule, EntityEmbed
from torch import tensor


@mock.patch("entity_embed.BlockerNet.__init__", return_value=None)
@mock.patch("entity_embed.data_utils.helpers.Vocab.load_vectors")
def test_set_embedding_size_when_using_semantic_attrs(
    mock_load_vectors,
    mock_blocker_net_init,
    caplog,
):
    attr_info_dict = {
        "name": {
            "field_type": "SEMANTIC_MULTITOKEN",
            "tokenizer": "entity_embed.default_tokenizer",
            "vocab": "charngram.100d",
            "use_mask": True,
        },
    }

    row_dict = {x: {"id": x, "name": f"foo product {x}"} for x in range(50)}

    row_numericalizer = AttrInfoDictParser.from_dict(attr_info_dict, row_dict=row_dict)

    mock_load_vectors.assert_called_once_with("charngram.100d")

    EXPECTED_EMBEDDING_SIZE = 100
    row_numericalizer.attr_info_dict["name"].vocab.vectors = tensor([[0] * EXPECTED_EMBEDDING_SIZE])

    datamodule = DeduplicationDataModule(
        row_dict=row_dict,
        cluster_attr="name",
        row_numericalizer=row_numericalizer,
        batch_size=10,
        eval_batch_size=32,
        train_cluster_len=10,
        valid_cluster_len=10,
        test_cluster_len=10,
    )

    caplog.set_level(logging.WARNING)
    model = EntityEmbed(datamodule=datamodule, embedding_size=500)
    assert (
        "Overriding embedding_size=500 with embedding_size=100 "
        "since you're using semantic fields" in caplog.text
    )

    mock_blocker_net_init.assert_called_once_with(
        row_numericalizer.attr_info_dict, embedding_size=EXPECTED_EMBEDDING_SIZE
    )

    assert model.embedding_size == EXPECTED_EMBEDDING_SIZE
