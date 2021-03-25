import logging

from n2 import HnswIndex

from .helpers import build_index_build_kwargs, build_index_search_kwargs

logger = logging.getLogger(__name__)


class ANNEntityIndex:
    def __init__(self, embedding_size):
        self.approx_knn_index = HnswIndex(dimension=embedding_size, metric="angular")
        self.vector_idx_to_id = None
        self.is_built = False

    def insert_vector_dict(self, vector_dict):
        for vector in vector_dict.values():
            self.approx_knn_index.add_data(vector)
        self.vector_idx_to_id = dict(enumerate(vector_dict.keys()))

    def build(
        self,
        index_build_kwargs=None,
    ):
        if self.vector_idx_to_id is None:
            raise ValueError("Please call insert_vector_dict first")

        actual_index_build_kwargs = build_index_build_kwargs(index_build_kwargs)
        self.approx_knn_index.build(**actual_index_build_kwargs)
        self.is_built = True

    def search_pairs(self, k, sim_threshold, index_search_kwargs=None):
        if not self.is_built:
            raise ValueError("Please call build first")
        if sim_threshold > 1 or sim_threshold < 0:
            raise ValueError(f"sim_threshold={sim_threshold} must be <= 1 and >= 0")

        logger.debug("Searching on approx_knn_index...")

        distance_threshold = 1 - sim_threshold

        index_search_kwargs = build_index_search_kwargs(index_search_kwargs)
        neighbor_and_distance_list_of_list = self.approx_knn_index.batch_search_by_ids(
            item_ids=self.vector_idx_to_id.keys(),
            k=k,
            include_distances=True,
            **index_search_kwargs,
        )

        logger.debug("Search on approx_knn_index done, building found_pair_set now...")

        found_pair_set = set()
        for i, neighbor_distance_list in enumerate(neighbor_and_distance_list_of_list):
            left_id = self.vector_idx_to_id[i]
            for j, distance in neighbor_distance_list:
                if i != j and distance <= distance_threshold:
                    right_id = self.vector_idx_to_id[j]
                    # must use sorted to always have smaller id on left of pair tuple
                    pair = tuple(sorted([left_id, right_id]))
                    found_pair_set.add(pair)

        logger.debug(
            f"Building found_pair_set done. Found len(found_pair_set)={len(found_pair_set)} pairs."
        )

        return found_pair_set


class ANNLinkageIndex:
    def __init__(self, embedding_size):
        self.left_index = ANNEntityIndex(embedding_size)
        self.right_index = ANNEntityIndex(embedding_size)

    def insert_vector_dict(self, left_vector_dict, right_vector_dict):
        self.left_index.insert_vector_dict(vector_dict=left_vector_dict)
        self.right_index.insert_vector_dict(vector_dict=right_vector_dict)

    def build(
        self,
        index_build_kwargs=None,
    ):
        self.left_index.build(index_build_kwargs=index_build_kwargs)
        self.right_index.build(index_build_kwargs=index_build_kwargs)

    def search_pairs(
        self,
        k,
        sim_threshold,
        left_vector_dict,
        right_vector_dict,
        left_source,
        index_search_kwargs=None,
    ):
        if not self.left_index.is_built or not self.right_index.is_built:
            raise ValueError("Please call build first")
        if sim_threshold > 1 or sim_threshold < 0:
            raise ValueError(f"sim_threshold={sim_threshold} must be <= 1 and >= 0")

        index_search_kwargs = build_index_search_kwargs(index_search_kwargs)
        distance_threshold = 1 - sim_threshold
        all_pair_set = set()

        for dataset_name, index, vector_dict, other_index in [
            (left_source, self.left_index, right_vector_dict, self.right_index),
            (None, self.right_index, left_vector_dict, self.left_index),
        ]:
            logger.debug(f"Searching on approx_knn_index of dataset_name={dataset_name}...")

            neighbor_and_distance_list_of_list = index.approx_knn_index.batch_search_by_vectors(
                vs=vector_dict.values(), k=k, include_distances=True, **index_search_kwargs
            )

            logger.debug(
                f"Search on approx_knn_index of dataset_name={dataset_name}... done, "
                "filling all_pair_set now..."
            )

            for i, neighbor_distance_list in enumerate(neighbor_and_distance_list_of_list):
                other_id = other_index.vector_idx_to_id[i]
                for j, distance in neighbor_distance_list:
                    if distance <= distance_threshold:  # do NOT check for i != j here
                        id_ = index.vector_idx_to_id[j]
                        if dataset_name and dataset_name == left_source:
                            left_id, right_id = (id_, other_id)
                        else:
                            left_id, right_id = (other_id, id_)
                        pair = (
                            left_id,
                            right_id,
                        )  # do NOT use sorted here, figure out from datasets
                        all_pair_set.add(pair)

            logger.debug(f"Filling all_pair_set with dataset_name={dataset_name} done.")

        logger.debug(
            "All searches done, all_pair_set filled. "
            f"Found len(all_pair_set)={len(all_pair_set)} pairs."
        )

        return all_pair_set
