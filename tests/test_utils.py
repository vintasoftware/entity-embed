import pytest
from entity_embed.data_utils.utils import compute_max_str_len


@pytest.fixture
def attr_val_gen():
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
    return (row["name"] for row in row_dict.values())


def test_compute_max_str_len_is_multitoken_false(attr_val_gen):
    max_str_len = compute_max_str_len(
        attr_val_gen=attr_val_gen,
        is_multitoken=False,
        # We don't need a tokenizer here
        tokenizer=None,
    )

    # Since this isn't multitoken, we simply get the length of the
    # biggest attr_val_gen value ("the foo product from world")
    assert max_str_len == 26


def test_compute_max_str_len_is_multitoken_true(attr_val_gen):
    max_str_len = compute_max_str_len(
        attr_val_gen=attr_val_gen,
        is_multitoken=True,
        tokenizer=lambda x: x.split(),
    )

    # We get the length of the largest token we obtained from attr_val_gen,
    # since our tokenizer simply split "name" into smaller strings, we're
    # getting the length for "product", which should be "7". However, since
    # we always want even values, we get the next available integer, "8".
    assert max_str_len == 8


def test_compute_max_str_len_is_multitoken_true_without_callable_tokenizer_raises(attr_val_gen):
    with pytest.raises(TypeError):
        compute_max_str_len(attr_val_gen=attr_val_gen, is_multitoken=True, tokenizer=None)
