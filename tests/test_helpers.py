import mock
from entity_embed.helpers import (
    build_index_build_kwargs,
    build_index_search_kwargs,
    build_loader_kwargs,
)


@mock.patch("entity_embed.helpers.os.cpu_count", return_value=8)
def test_build_loader_kwargs(mock_cpu_count):
    loader_kwargs = build_loader_kwargs()
    assert loader_kwargs == {"num_workers": 8, "multiprocessing_context": "fork"}

    loader_kwargs = build_loader_kwargs({"num_workers": 4})
    assert loader_kwargs == {"num_workers": 4, "multiprocessing_context": "fork"}

    loader_kwargs = build_loader_kwargs({"multiprocessing_context": "foo"})
    assert loader_kwargs == {"num_workers": 8, "multiprocessing_context": "foo"}

    loader_kwargs = build_loader_kwargs({"num_workers": 16, "multiprocessing_context": "bar"})
    assert loader_kwargs == {"num_workers": 16, "multiprocessing_context": "bar"}

    loader_kwargs = build_loader_kwargs({"num_workers": 32, "multiprocessing_context": None})
    assert loader_kwargs == {"num_workers": 32, "multiprocessing_context": "fork"}


@mock.patch("entity_embed.helpers.os.cpu_count", return_value=8)
def test_build_index_build_kwargs(mock_cpu_count):
    index_build_kwargs = build_index_build_kwargs()
    assert index_build_kwargs == {"m": 64, "max_m0": 64, "ef_construction": 150, "n_threads": 8}

    index_build_kwargs = build_index_build_kwargs({"m": 128, "n_threads": None})
    assert index_build_kwargs == {"m": 128, "max_m0": 64, "ef_construction": 150, "n_threads": 8}

    index_build_kwargs = build_index_build_kwargs(
        {"max_m0": 32, "ef_construction": 200, "n_threads": 16}
    )
    assert index_build_kwargs == {"m": 64, "max_m0": 32, "ef_construction": 200, "n_threads": 16}


@mock.patch("entity_embed.helpers.os.cpu_count", return_value=8)
def test_build_index_search_kwargs(mock_cpu_count):
    index_search_kwargs = build_index_search_kwargs()
    assert index_search_kwargs == {"ef_search": -1, "num_threads": 8}

    index_search_kwargs = build_index_search_kwargs({"ef_search": 2, "n_threads": None})
    assert index_search_kwargs == {"ef_search": 2, "num_threads": 8}

    index_search_kwargs = build_index_search_kwargs({"n_threads": 16})
    assert index_search_kwargs == {"ef_search": -1, "num_threads": 16}
