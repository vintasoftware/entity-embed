import mock
import pytest
from entity_embed import EntityEmbed
from entity_embed.data_utils.helpers import AttrInfoDictParser
from torch import tensor


@mock.patch("entity_embed.BlockerNet.__init__", return_value=None)
@mock.patch("entity_embed.data_utils.helpers.Vocab.load_vectors")
def test_set_embedding_size_when_using_semantic_attrs(
    mock_load_vectors,
    mock_blocker_net_init,
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

    row_numericalizer = AttrInfoDictParser.from_dict(attr_info_dict, row_list=row_dict.values())

    mock_load_vectors.assert_called_once_with("charngram.100d")

    EXPECTED_EMBEDDING_SIZE = 100
    row_numericalizer.attr_info_dict["name"].vocab.vectors = tensor([[0] * EXPECTED_EMBEDDING_SIZE])

    with pytest.raises(ValueError) as excinfo:
        EntityEmbed(row_numericalizer=row_numericalizer, embedding_size=500)

        assert "Invalid embedding_size=500. Expected 100, due to semantic fields." == excinfo.value
