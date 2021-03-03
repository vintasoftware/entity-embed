import logging

import mock
import pytest
from entity_embed.data_utils.utils import (
    cluster_dict_to_id_pairs,
    compute_max_str_len,
    compute_vocab_counter,
    count_cluster_dict_pairs,
    id_pairs_to_cluster_mapping_and_dict,
    split_clusters,
)


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


def test_compute_max_str_len_is_multitoken_tokenizer_returns_none(attr_val_gen):
    max_str_len = compute_max_str_len(
        attr_val_gen=attr_val_gen,
        is_multitoken=True,
        tokenizer=lambda x: None,
    )

    assert max_str_len == 0


def test_compute_vocab_counter(attr_val_gen):
    vocab_counter = compute_vocab_counter(
        attr_val_gen=attr_val_gen,
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
    cluster_mapping, cluster_dict = id_pairs_to_cluster_mapping_and_dict(id_pairs)

    # 1, 2, 3 are part of the same cluster
    assert len(set(v for k, v in cluster_mapping.items() if k in [1, 2, 3])) == 1

    # 4, 5 are part of the same cluster
    assert len(set(v for k, v in cluster_mapping.items() if k in [4, 5])) == 1

    # 6, 7, 8, 9, 10 are part of the same cluster
    assert len(set(v for k, v in cluster_mapping.items() if k in [6, 7, 8, 9, 10])) == 1

    clusters_list = sorted(c for c in cluster_dict.values())
    assert clusters_list == [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]


def fake_rnd_sample(population, sample_len):
    return list(population)[:sample_len]


@mock.patch("entity_embed.data_utils.utils.random.Random.sample", wraps=fake_rnd_sample)
def test_split_clusters(mock_rnd_sample):
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11, 18],
        12: [12, 13, 15],
        14: [14, 16],
    }

    train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
        cluster_dict=cluster_dict,
        train_len=2,
        valid_len=2,
        test_len=2,
        random_seed=40,
    )

    assert mock_rnd_sample.call_count == 2

    assert train_cluster_dict == {
        1: [1, 2, 3],
        4: [4, 5],
    }

    assert valid_cluster_dict == {
        6: [6, 7, 8, 9, 10],
        11: [11, 18],
    }

    assert test_cluster_dict == {
        12: [12, 13, 15],
        14: [14, 16],
    }


@mock.patch("entity_embed.data_utils.utils.random.Random.sample", wraps=fake_rnd_sample)
def test_split_clusters_only_plural_clusters_false(mock_rnd_sample):
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11],
        12: [12, 13, 15],
        14: [14, 16],
    }

    train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
        cluster_dict=cluster_dict,
        train_len=2,
        valid_len=2,
        test_len=2,
        random_seed=40,
        only_plural_clusters=False,
    )

    assert mock_rnd_sample.call_count == 2

    assert train_cluster_dict == {
        1: [1, 2, 3],
        4: [4, 5],
    }

    # [11] stays since we allow non-plural clusters
    assert valid_cluster_dict == {
        6: [6, 7, 8, 9, 10],
        11: [11],
    }

    assert test_cluster_dict == {
        12: [12, 13, 15],
        14: [14, 16],
    }


@mock.patch("entity_embed.data_utils.utils.random.Random.sample", wraps=fake_rnd_sample)
def test_split_clusters_not_all_clusters_used(mock_rnd_sample, caplog):
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11, 18],
        12: [12, 13, 15],
        14: [14, 16],
        20: [20, 21, 22],
    }

    caplog.set_level(logging.WARNING)
    train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
        cluster_dict=cluster_dict,
        train_len=2,
        valid_len=2,
        test_len=2,
        random_seed=40,
    )
    assert (
        "(train_len + valid_len + test_len)=6 is less than len(all_cluster_id_set)=7" in caplog.text
    )

    assert mock_rnd_sample.call_count == 3

    # [20, 21, 22] should be left out of the returned dicts
    assert train_cluster_dict == {
        1: [1, 2, 3],
        4: [4, 5],
    }

    assert valid_cluster_dict == {
        6: [6, 7, 8, 9, 10],
        11: [11, 18],
    }

    assert test_cluster_dict == {
        12: [12, 13, 15],
        14: [14, 16],
    }


@mock.patch("entity_embed.data_utils.utils.random.Random.sample", wraps=fake_rnd_sample)
def test_split_clusters_only_plural_clusters(mock_rnd_sample):
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11],
        12: [12, 13, 15],
        14: [14, 16],
    }

    train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
        cluster_dict=cluster_dict,
        train_len=2,
        valid_len=2,
        test_len=2,
        random_seed=40,
    )

    assert mock_rnd_sample.call_count == 2

    # [11] should be removed since it's a cluster with only one id
    assert train_cluster_dict == {
        1: [1, 2, 3],
        4: [4, 5],
    }

    assert valid_cluster_dict == {
        6: [6, 7, 8, 9, 10],
        12: [12, 13, 15],
    }

    assert test_cluster_dict == {
        14: [14, 16],
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


def test_count_cluster_dict_pairs():
    cluster_dict = {
        1: [1, 2, 3],
        4: [4, 5],
        6: [6, 7, 8, 9, 10],
        11: [11, 18],
        12: [12, 13, 15],
        14: [14, 16],
    }

    count = count_cluster_dict_pairs(cluster_dict)
    # sum((n*(n - 1))//2) => 3 + 1 + 10 + 1 + 3 + 1
    assert count == 19
