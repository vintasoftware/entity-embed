import os


def build_loader_kwargs(kwargs_dict=None):
    if not kwargs_dict:
        kwargs_dict = {}
    num_workers = kwargs_dict.get("num_workers") or os.cpu_count()
    multiprocessing_context = kwargs_dict.get("multiprocessing_context") or "fork"
    return {
        "num_workers": num_workers,
        "multiprocessing_context": multiprocessing_context,
    }


def build_index_build_kwargs(kwargs_dict=None):
    if not kwargs_dict:
        kwargs_dict = {}
    m = kwargs_dict.get("m") or 64
    max_m0 = kwargs_dict.get("max_m0") or 64
    ef_construction = kwargs_dict.get("ef_construction") or 150
    n_threads = kwargs_dict.get("n_threads") or os.cpu_count()
    return {
        "m": m,
        "max_m0": max_m0,
        "ef_construction": ef_construction,
        "n_threads": n_threads,
    }


def build_index_search_kwargs(kwargs_dict=None):
    if not kwargs_dict:
        kwargs_dict = {}
    ef_search = kwargs_dict.get("ef_search") or -1
    # Note the N2 API is inconsistent! Here it's num_threads, on build it's n_threads
    num_threads = kwargs_dict.get("n_threads") or os.cpu_count()
    return {
        "ef_search": ef_search,
        "num_threads": num_threads,
    }
