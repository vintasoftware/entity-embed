import os


def build_loader_kwargs(kwargs_dict=None):
    if kwargs_dict:
        actual_kwargs_dict = dict(kwargs_dict)
    else:
        actual_kwargs_dict = {}
    num_workers = actual_kwargs_dict.get("num_workers") or os.cpu_count()
    multiprocessing_context = actual_kwargs_dict.get("multiprocessing_context") or "fork"
    return {
        "num_workers": num_workers,
        "multiprocessing_context": multiprocessing_context,
    }


def build_index_build_kwargs(kwargs_dict=None):
    if kwargs_dict:
        actual_kwargs_dict = dict(kwargs_dict)
    else:
        actual_kwargs_dict = {}
    m = actual_kwargs_dict.get("m") or 64
    max_m0 = actual_kwargs_dict.get("max_m0") or 64
    ef_construction = actual_kwargs_dict.get("ef_construction") or 150
    n_threads = actual_kwargs_dict.get("n_threads") or os.cpu_count()
    return {
        "m": m,
        "max_m0": max_m0,
        "ef_construction": ef_construction,
        "n_threads": n_threads,
    }


def build_index_search_kwargs(kwargs_dict=None):
    if kwargs_dict:
        actual_kwargs_dict = dict(kwargs_dict)
    else:
        actual_kwargs_dict = {}
    ef_search = actual_kwargs_dict.get("ef_search") or -1
    num_threads = actual_kwargs_dict.get("num_threads") or os.cpu_count()
    return {
        "ef_search": ef_search,
        "num_threads": num_threads,
    }
