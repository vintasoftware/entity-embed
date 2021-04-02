import pytest
from entity_embed.data_utils.utils import (
    Enumerator,
    cluster_dict_to_id_pairs,
    compute_max_str_len,
    compute_vocab_counter,
    id_pairs_to_cluster_mapping_and_dict,
    record_dict_to_cluster_dict,
    split_clusters,
)


def test_enumerator():
    enumerator = Enumerator()
    for x in range(100):
        enumerator[f"test-{x}"]

    for x in range(100):
        assert enumerator[f"test-{x}"] == x


def test_enumerator_with_start():
    start = 5
    enumerator = Enumerator(start=start)
    for x in range(100):
        enumerator[f"test-{x}"]

    for x in range(100):
        assert enumerator[f"test-{x}"] == x + start


def test_record_dict_to_cluster_dict():
    record_dict = {
        1: {"id": 1, "cluster": 0},
        2: {"id": 2, "cluster": 1},
        3: {"id": 3, "cluster": 0},
        4: {"id": 4, "cluster": 0},
        5: {"id": 5, "cluster": 1},
        6: {"id": 6, "cluster": 2},
    }

    cluster_dict = record_dict_to_cluster_dict(record_dict=record_dict, cluster_field="cluster")

    assert cluster_dict == {
        0: [1, 3, 4],
        1: [2, 5],
        2: [6],
    }


@pytest.fixture
def field_val_gen():
    record_dict = {
        1: {
            "id": "1",
            "name": "foo product",
            "price": 1.00,
            "source": "bar",
        },
        2: {
            "id": "2",
            "name": "the foo product from world",
            "price": 1.20,
            "source": "baz",
        },
    }
    return (record["name"] for record in record_dict.values())


def test_compute_max_str_len_is_multitoken_false(field_val_gen):
    max_str_len = compute_max_str_len(
        field_val_gen=field_val_gen,
        is_multitoken=False,
        # We don't need a tokenizer here
        tokenizer=None,
    )

    # Since this isn't multitoken, we simply get the length of the
    # biggest field_val_gen value ("the foo product from world")
    assert max_str_len == 26


def test_compute_max_str_len_is_multitoken_true(field_val_gen):
    max_str_len = compute_max_str_len(
        field_val_gen=field_val_gen,
        is_multitoken=True,
        tokenizer=lambda x: x.split(),
    )

    # We get the length of the largest token we obtained from field_val_gen,
    # since our tokenizer simply split "name" into smaller strings, we're
    # getting the length for "product", which should be "7". However, since
    # we always want even values, we get the next available integer, "8".
    assert max_str_len == 8


def test_compute_max_str_len_is_multitoken_true_without_callable_tokenizer_raises(field_val_gen):
    with pytest.raises(TypeError):
        compute_max_str_len(field_val_gen=field_val_gen, is_multitoken=True, tokenizer=None)


def test_compute_max_str_len_is_multitoken_with_tokenizer_that_doesnt_return_tokens(field_val_gen):
    max_str_len = compute_max_str_len(
        field_val_gen=field_val_gen,
        is_multitoken=True,
        tokenizer=lambda x: [],
    )

    assert max_str_len == 0


def test_compute_vocab_counter(field_val_gen):
    vocab_counter = compute_vocab_counter(
        field_val_gen=field_val_gen,
        tokenizer=lambda x: x.split(),
    )
    assert dict(vocab_counter) == {"foo": 2, "product": 2, "the": 1, "from": 1, "world": 1}


def test_id_pairs_to_cluster_mapping_and_dict():
    id_pairs = {
        (1, 2),
        (2, 3),
        (4, 5),
        (6, 7),
        (7, 8),
        (7, 9),
        (9, 10),
    }
    record_dict = {id_: {"name": str(id_)} for id_ in range(1, 13)}
    cluster_mapping, cluster_dict = id_pairs_to_cluster_mapping_and_dict(id_pairs, record_dict)

    # 1, 2, 3 are part of the same cluster
    assert len(set(v for k, v in cluster_mapping.items() if k in [1, 2, 3])) == 1

    # 4, 5 are part of the same cluster
    assert len(set(v for k, v in cluster_mapping.items() if k in [4, 5])) == 1

    # 6, 7, 8, 9, 10 are part of the same cluster
    assert len(set(v for k, v in cluster_mapping.items() if k in [6, 7, 8, 9, 10])) == 1

    # check clusters and singletons
    clusters_list = sorted(c for c in cluster_dict.values())
    assert clusters_list == [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11], [12]]


def fake_rnd_sample(population, sample_len):
    return list(population)[:sample_len]


def test_split_clusters():
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11, 12],
        13: [13, 14, 15],
        16: [16, 17],
        18: [18, 19, 20],
        21: [21, 22],
        23: [23],
        24: [24],
        25: [25],
        26: [26],
    }

    train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
        cluster_dict=cluster_dict,
        train_proportion=0.5,
        valid_proportion=0.25,
        random_seed=40,
    )

    assert train_cluster_dict == {
        21: [21, 22],
        13: [13, 14, 15],
        18: [18, 19, 20],
        1: [1, 2, 3],
        25: [25],
        26: [26],
    }

    assert valid_cluster_dict == {6: [6, 7, 8, 9, 10], 16: [16, 17], 23: [23]}

    assert test_cluster_dict == {
        4: [4, 5],
        11: [11, 12],
        24: [24],
    }


def test_cluster_dict_to_id_pairs():
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11, 18],
        12: [12, 13, 15],
        14: [14, 16],
    }
    id_pairs = cluster_dict_to_id_pairs(cluster_dict)
    assert id_pairs == {
        (1, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 10),
        (7, 8),
        (7, 9),
        (7, 10),
        (8, 9),
        (8, 10),
        (9, 10),
        (11, 18),
        (12, 13),
        (12, 15),
        (13, 15),
        (14, 16),
    }
