import itertools
import logging
import os

import pytorch_lightning as pl
import torch
from n2 import HnswIndex
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.miners import BatchHardMiner
from tqdm.auto import tqdm

from .data_utils.datasets import ClusterDataset, RowDataset
from .data_utils.numericalizer import NumericalizeInfo, RowNumericalizer
from .data_utils.utils import (
    Enumerator,
    cluster_dict_to_id_pairs,
    count_cluster_dict_pairs,
    id_pairs_to_cluster_mapping_and_dict,
    row_dict_to_cluster_dict,
    separate_dict_left_right,
    split_clusters,
    split_clusters_to_row_dicts,
)
from .evaluation import f1_score, pair_entity_ratio, precision_and_recall
from .models import BlockerNet

logger = logging.getLogger(__name__)


def build_row_numericalizer(attr_info_dict, row_dict=None):
    # Fix NumericalizeInfo from dicts and initialize RowNumericalizer.
    for attr, numericalize_info in list(attr_info_dict.items()):
        if not numericalize_info:
            raise ValueError(
                f'Please set the value of "{attr}" in attr_info_dict, {numericalize_info}'
            )
        if not isinstance(numericalize_info, NumericalizeInfo):
            numericalize_info["tokenizer"] = numericalize_info.get("tokenizer")
            numericalize_info["alphabet"] = numericalize_info.get("alphabet")
            numericalize_info["max_str_len"] = numericalize_info.get("max_str_len")
            numericalize_info["vocab"] = numericalize_info.get("vocab")
            attr_info_dict[attr] = NumericalizeInfo(**numericalize_info)

    # For now on, one must use row_numericalizer instead of attr_info_dict,
    # because RowNumericalizer fills None values of alphabet and max_str_len.
    return RowNumericalizer(attr_info_dict=attr_info_dict, row_dict=row_dict)


class DeduplicationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        row_dict,
        cluster_attr,
        row_numericalizer,
        pos_pair_batch_size,
        neg_pair_batch_size,
        row_batch_size,
        train_cluster_len,
        valid_cluster_len,
        test_cluster_len,
        only_plural_clusters,
        pair_loader_kwargs=None,
        row_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__()
        self.row_dict = row_dict
        self.cluster_attr = cluster_attr
        self.row_numericalizer = row_numericalizer
        self.pos_pair_batch_size = pos_pair_batch_size
        self.neg_pair_batch_size = neg_pair_batch_size
        self.row_batch_size = row_batch_size
        self.train_cluster_len = train_cluster_len
        self.valid_cluster_len = valid_cluster_len
        self.test_cluster_len = test_cluster_len
        self.only_plural_clusters = only_plural_clusters
        self.pair_loader_kwargs = pair_loader_kwargs or {
            "num_workers": os.cpu_count(),
            "multiprocessing_context": "fork",
        }
        self.row_loader_kwargs = row_loader_kwargs or {
            "num_workers": os.cpu_count(),
            "multiprocessing_context": "fork",
        }
        self.random_seed = random_seed

        self.valid_true_pair_set = None
        self.test_true_pair_set = None
        self.train_row_dict = None
        self.valid_row_dict = None
        self.test_row_dict = None

    def setup(self, stage=None):
        cluster_dict = row_dict_to_cluster_dict(self.row_dict, self.cluster_attr)

        train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
            cluster_dict,
            train_len=self.train_cluster_len,
            valid_len=self.valid_cluster_len,
            test_len=self.test_cluster_len,
            random_seed=self.random_seed,
            only_plural_clusters=self.only_plural_clusters,
        )
        self.valid_true_pair_set = cluster_dict_to_id_pairs(valid_cluster_dict)
        self.test_true_pair_set = cluster_dict_to_id_pairs(test_cluster_dict)
        logger.info("Train pair count: %s", count_cluster_dict_pairs(train_cluster_dict))
        logger.info("Valid pair count: %s", len(self.valid_true_pair_set))
        logger.info("Test pair count: %s", len(self.test_true_pair_set))

        self.train_row_dict, self.valid_row_dict, self.test_row_dict = split_clusters_to_row_dicts(
            row_dict=self.row_dict,
            train_cluster_dict=train_cluster_dict,
            valid_cluster_dict=valid_cluster_dict,
            test_cluster_dict=test_cluster_dict,
        )

        # If not test, drop test values
        if stage == "fit":
            self.test_true_pair_set = None
            self.test_row_dict = None
        elif stage == "test":
            self.valid_true_pair_set = None
            self.train_row_dict = None
            self.valid_row_dict = None

    def train_dataloader(self):
        train_cluster_dataset = ClusterDataset.from_cluster_dict(
            row_dict=self.train_row_dict,
            cluster_attr=self.cluster_attr,
            row_numericalizer=self.row_numericalizer,
            pos_pair_batch_size=self.pos_pair_batch_size,
            neg_pair_batch_size=self.neg_pair_batch_size,
            random_seed=self.random_seed,
        )
        train_cluster_loader = torch.utils.data.DataLoader(
            train_cluster_dataset,
            batch_size=None,  # batch size is in ClusterDataset
            shuffle=True,
            **self.pair_loader_kwargs,
        )
        return train_cluster_loader

    def val_dataloader(self):
        valid_row_dataset = RowDataset(
            row_dict=self.valid_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.row_batch_size,
        )
        valid_row_loader = torch.utils.data.DataLoader(
            valid_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.row_loader_kwargs,
        )
        return valid_row_loader

    def test_dataloader(self):
        test_row_dataset = RowDataset(
            row_dict=self.test_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.row_batch_size,
        )
        test_row_loader = torch.utils.data.DataLoader(
            test_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.row_loader_kwargs,
        )
        return test_row_loader


class LinkageDataModule(DeduplicationDataModule):
    def __init__(
        self,
        row_dict,
        cluster_attr,
        row_numericalizer,
        pos_pair_batch_size,
        neg_pair_batch_size,
        row_batch_size,
        train_cluster_len,
        valid_cluster_len,
        test_cluster_len,
        only_plural_clusters,
        left_id_set,
        right_id_set,
        pair_loader_kwargs=None,
        row_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__(
            row_dict=row_dict,
            cluster_attr=cluster_attr,
            row_numericalizer=row_numericalizer,
            pos_pair_batch_size=pos_pair_batch_size,
            neg_pair_batch_size=neg_pair_batch_size,
            row_batch_size=row_batch_size,
            train_cluster_len=train_cluster_len,
            valid_cluster_len=valid_cluster_len,
            test_cluster_len=test_cluster_len,
            only_plural_clusters=only_plural_clusters,
            pair_loader_kwargs=pair_loader_kwargs,
            row_loader_kwargs=row_loader_kwargs,
            random_seed=random_seed,
        )
        self.left_id_set = left_id_set
        self.right_id_set = right_id_set

    def _set_filtered_from_id_sets(self, s):
        return {
            (id_1, id_2)
            for (id_1, id_2) in s
            if (id_1 in self.left_id_set and id_2 in self.right_id_set)
            or (id_1 in self.right_id_set and id_2 in self.left_id_set)
        }

    def setup(self, stage=None):
        super().setup(stage=stage)

        # Ensure pair sets only have ids with datset sources like (left, right) or (right, left),
        # i.e., no ids from the same dataset (left, left) or (right, right)
        if self.valid_true_pair_set is not None:
            self.valid_true_pair_set = self._set_filtered_from_id_sets(self.valid_true_pair_set)
        if self.test_true_pair_set is not None:
            self.test_true_pair_set = self._set_filtered_from_id_sets(self.test_true_pair_set)

    def separate_dict_left_right(self, d):
        return separate_dict_left_right(
            d, left_id_set=self.left_id_set, right_id_set=self.right_id_set
        )


class PairwiseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        row_dict,
        row_numericalizer,
        pos_pair_batch_size,
        neg_pair_batch_size,
        row_batch_size,
        train_true_pair_set,
        valid_true_pair_set,
        test_true_pair_set,
        pair_loader_kwargs=None,
        row_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__()

        self.row_dict = row_dict
        self.row_numericalizer = row_numericalizer
        self.pos_pair_batch_size = pos_pair_batch_size
        self.neg_pair_batch_size = neg_pair_batch_size
        self.row_batch_size = row_batch_size
        self.pair_loader_kwargs = pair_loader_kwargs or {
            "num_workers": os.cpu_count(),
            "multiprocessing_context": "fork",
        }
        self.row_loader_kwargs = row_loader_kwargs or {
            "num_workers": os.cpu_count(),
            "multiprocessing_context": "fork",
        }
        self.random_seed = random_seed

        self.train_true_pair_set = train_true_pair_set
        self.valid_true_pair_set = valid_true_pair_set
        self.test_true_pair_set = test_true_pair_set

        self.train_row_dict = None
        self.valid_row_dict = None
        self.test_row_dict = None

    def setup(self, stage=None):
        if stage == "fit":
            all_pair_sets = [
                self.train_true_pair_set,
                self.valid_true_pair_set,
                self.test_true_pair_set,
            ]
        elif stage == "test":
            all_pair_sets = [
                self.test_true_pair_set,
            ]
        else:
            all_pair_sets = []
        self.left_id_set = {pair[0] for pair_set in all_pair_sets for pair in pair_set}
        self.right_id_set = {pair[1] for pair_set in all_pair_sets for pair in pair_set}

        if stage == "fit":
            self.train_row_dict = {
                id_: self.row_dict[id_] for pair in self.train_true_pair_set for id_ in pair
            }
            self.valid_row_dict = {
                id_: self.row_dict[id_] for pair in self.valid_true_pair_set for id_ in pair
            }
        elif stage == "test":
            self.test_row_dict = {
                id_: self.row_dict[id_] for pair in self.test_true_pair_set for id_ in pair
            }

    def train_dataloader(self):
        train_cluster_dataset = ClusterDataset.from_pairs(
            row_dict=self.train_row_dict,
            true_pair_set=self.train_true_pair_set,
            row_numericalizer=self.row_numericalizer,
            pos_pair_batch_size=self.pos_pair_batch_size,
            neg_pair_batch_size=self.neg_pair_batch_size,
            random_seed=self.random_seed,
        )
        train_cluster_loader = torch.utils.data.DataLoader(
            train_cluster_dataset,
            batch_size=None,  # batch size is set on ClusterDataset
            shuffle=True,
            **self.pair_loader_kwargs,
        )
        return train_cluster_loader

    def val_dataloader(self):
        valid_row_dataset = RowDataset(
            row_dict=self.valid_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.row_batch_size,
        )
        valid_row_loader = torch.utils.data.DataLoader(
            valid_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.row_loader_kwargs,
        )
        return valid_row_loader

    def test_dataloader(self):
        test_row_dataset = RowDataset(
            row_dict=self.test_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.row_batch_size,
        )
        test_row_loader = torch.utils.data.DataLoader(
            test_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.row_loader_kwargs,
        )
        return test_row_loader

    def separate_dict_left_right(self, d):
        return separate_dict_left_right(
            d, left_id_set=self.left_id_set, right_id_set=self.right_id_set
        )


class EntityEmbed(pl.LightningModule):
    def __init__(
        self,
        datamodule,
        n_channels=8,
        embedding_size=128,
        embed_dropout_p=0.2,
        use_attention=True,
        use_mask=False,
        loss_cls=NTXentLoss,
        loss_kwargs=None,
        miner_cls=BatchHardMiner,
        miner_kwargs=None,
        optimizer_cls=torch.optim.Adam,
        learning_rate=0.001,
        optimizer_kwargs=None,
        ann_k=10,
        sim_threshold_list=[0.3, 0.5, 0.7, 0.9],
        index_build_kwargs=None,
        index_search_kwargs=None,
    ):
        super().__init__()
        self.row_numericalizer = datamodule.row_numericalizer
        self.attr_info_dict = self.row_numericalizer.attr_info_dict
        self.n_channels = n_channels
        self.embedding_size = embedding_size
        self.embed_dropout_p = embed_dropout_p
        self.use_attention = use_attention
        self.use_mask = use_mask
        self.blocker_net = BlockerNet(
            self.attr_info_dict,
            n_channels=n_channels,
            embedding_size=embedding_size,
            embed_dropout_p=embed_dropout_p,
            use_attention=use_attention,
            use_mask=use_mask,
        )
        self.losser = loss_cls(**loss_kwargs if loss_kwargs else {"temperature": 0.1})
        if miner_cls:
            self.miner = miner_cls(
                **miner_kwargs if miner_kwargs else {"distance": CosineSimilarity()}
            )
        else:
            self.miner = None
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.ann_k = ann_k
        self.sim_threshold_list = sim_threshold_list
        self.index_build_kwargs = index_build_kwargs
        self.index_search_kwargs = index_search_kwargs

        self.save_hyperparameters(
            "n_channels",
            "embedding_size",
            "embed_dropout_p",
            "use_attention",
            "use_mask",
            "loss_cls",
            "miner_cls",
            "optimizer_cls",
            "learning_rate",
            "optimizer_kwargs",
            "ann_k",
            "sim_threshold_list",
            "index_build_kwargs",
            "index_search_kwargs",
        )

        # set self._datamodule to access valid_row_dict and valid_true_pair_set
        # in validation_epoch_end
        self._datamodule = datamodule

    def forward(self, tensor_dict, sequence_length_dict):
        return self.blocker_net(tensor_dict, sequence_length_dict)

    def _warn_if_empty_indices_tuple(self, indices_tuple, batch_idx):
        with torch.no_grad():
            if all(t.nelement() == 0 for t in indices_tuple):
                logger.warning(f"Found empty indices_tuple at {self.current_epoch=}, {batch_idx=}")

    def training_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict, labels = batch
        embeddings = self.blocker_net(tensor_dict, sequence_length_dict)
        if self.miner:
            indices_tuple = self.miner(embeddings, labels)
            self._warn_if_empty_indices_tuple(indices_tuple, batch_idx)
        else:
            indices_tuple = None
        loss = self.losser(embeddings, labels, indices_tuple=indices_tuple)

        self.log("train_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.blocker_net.fix_signature_weights()
        self.log_dict(
            {
                f"signature_{attr}": weight
                for attr, weight in self.blocker_net.get_signature_weights().items()
            }
        )

    def validation_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict = batch
        embedding_batch = self.blocker_net(tensor_dict, sequence_length_dict)
        return embedding_batch

    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, true_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        vector_dict = dict(zip(row_dict.keys(), vector_list))

        ann_index = ANNEntityIndex(embedding_size=self.blocker_net.embedding_size)
        ann_index.insert_vector_dict(vector_dict)
        ann_index.build(index_build_kwargs=self.index_build_kwargs)

        for sim_threshold in self.sim_threshold_list:
            found_pair_set = ann_index.search_pairs(
                k=self.ann_k,
                sim_threshold=sim_threshold,
                index_search_kwargs=self.index_search_kwargs,
            )

            precision, recall = precision_and_recall(found_pair_set, true_pair_set)
            self.log_dict(
                {
                    f"{set_name}_precision_at_{sim_threshold}": precision,
                    f"{set_name}_recall_at_{sim_threshold}": recall,
                    f"{set_name}_f1_at_{sim_threshold}": f1_score(precision, recall),
                    f"{set_name}_pair_entity_ratio_at_{sim_threshold}": pair_entity_ratio(
                        len(found_pair_set), len(vector_list)
                    ),
                }
            )

    def validation_epoch_end(self, outputs):
        self._evaluate_with_ann(
            set_name="valid",
            row_dict=self._datamodule.valid_row_dict,
            embedding_batch_list=outputs,
            true_pair_set=self._datamodule.valid_true_pair_set,
        )

    def test_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict = batch
        return self.blocker_net(tensor_dict, sequence_length_dict)

    def test_epoch_end(self, outputs):
        self._evaluate_with_ann(
            set_name="test",
            row_dict=self._datamodule.test_row_dict,
            embedding_batch_list=outputs,
            true_pair_set=self._datamodule.test_true_pair_set,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        return optimizer

    def get_signature_weights(self):
        return self.blocker_net.get_signature_weights()

    def predict(
        self,
        row_dict,
        batch_size,
        loader_kwargs=None,
        device=None,
        show_progress=True,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        row_dataset = RowDataset(
            row_numericalizer=self.row_numericalizer, row_dict=row_dict, batch_size=batch_size
        )
        row_loader = torch.utils.data.DataLoader(
            row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **loader_kwargs
            if loader_kwargs
            else {"num_workers": os.cpu_count(), "multiprocessing_context": "fork"},
        )

        blocker_net = self.blocker_net.to(device)
        blocker_net.eval()
        with torch.no_grad():
            with tqdm(
                total=len(row_loader), desc="# batch embedding", disable=not show_progress
            ) as p_bar:
                vector_list = []
                for i, (tensor_dict, sequence_length_dict) in enumerate(row_loader):
                    tensor_dict = {attr: t.to(device) for attr, t in tensor_dict.items()}
                    embeddings = blocker_net(tensor_dict, sequence_length_dict)
                    vector_list.extend(v.data.numpy() for v in embeddings.cpu().unbind())
                    p_bar.update(1)

        vector_dict = dict(zip(row_dict.keys(), vector_list))
        return vector_dict


class LinkageEmbed(EntityEmbed):
    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, true_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        vector_dict = dict(zip(row_dict.keys(), vector_list))
        left_vector_dict, right_vector_dict = self.datamodule.separate_dict_left_right(vector_dict)

        ann_index = ANNLinkageIndex(embedding_size=self.blocker_net.embedding_size)
        ann_index.insert_vector_dict(
            left_vector_dict=left_vector_dict, right_vector_dict=right_vector_dict
        )
        ann_index.build(
            index_build_kwargs=self.index_build_kwargs,
        )

        for sim_threshold in self.sim_threshold_list:
            found_pair_set = ann_index.search_pairs(
                k=self.ann_k,
                sim_threshold=sim_threshold,
                left_vector_dict=left_vector_dict,
                right_vector_dict=right_vector_dict,
                index_search_kwargs=self.index_search_kwargs,
            )

            precision, recall = precision_and_recall(found_pair_set, true_pair_set)
            self.log_dict(
                {
                    f"{set_name}_precision_at_{sim_threshold}": precision,
                    f"{set_name}_recall_at_{sim_threshold}": recall,
                    f"{set_name}_f1_at_{sim_threshold}": f1_score(precision, recall),
                    f"{set_name}_pair_entity_ratio_at_{sim_threshold}": pair_entity_ratio(
                        len(found_pair_set), len(vector_list)
                    ),
                }
            )

    def predict(
        self,
        row_dict,
        left_id_set,
        right_id_set,
        batch_size,
        loader_kwargs=None,
        device=None,
        show_progress=True,
    ):
        vector_dict = super().predict(
            row_dict=row_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            device=device,
            show_progress=show_progress,
        )
        left_vector_dict, right_vector_dict = separate_dict_left_right(
            vector_dict, left_id_set=left_id_set, right_id_set=right_id_set
        )
        return left_vector_dict, right_vector_dict


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

        self.approx_knn_index.build(
            **index_build_kwargs
            if index_build_kwargs
            else {
                "m": 64,
                "max_m0": 64,
                "ef_construction": 150,
                "n_threads": os.cpu_count(),
            }
        )
        self.is_built = True

    def search_pairs(self, k, sim_threshold, index_search_kwargs=None):
        if not self.is_built:
            raise ValueError("Please call build first")
        if sim_threshold > 1 or sim_threshold < 0:
            raise ValueError(f"{sim_threshold=} must be <= 1 and >= 0")

        logger.debug("Searching on approx_knn_index...")

        distance_threshold = 1 - sim_threshold
        neighbor_and_distance_list_of_list = self.approx_knn_index.batch_search_by_ids(
            item_ids=self.vector_idx_to_id.keys(),
            k=k,
            include_distances=True,
            **index_search_kwargs
            if index_search_kwargs
            else {"ef_search": -1, "num_threads": os.cpu_count()},
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

        logger.debug(f"Building found_pair_set done. Found {len(found_pair_set)=} pairs.")

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
        index_search_kwargs=None,
        left_dataset_name="left",
        right_dataset_name="right",
    ):
        if not self.left_index.is_built or not self.right_index.is_built:
            raise ValueError("Please call build first")
        if sim_threshold > 1 or sim_threshold < 0:
            raise ValueError(f"{sim_threshold=} must be <= 1 and >= 0")

        distance_threshold = 1 - sim_threshold
        all_pair_set = set()

        for dataset_name, index, vector_dict, other_index in [
            (left_dataset_name, self.left_index, right_vector_dict, self.right_index),
            (right_dataset_name, self.right_index, left_vector_dict, self.left_index),
        ]:
            logger.debug(f"Searching on approx_knn_index of {dataset_name=}...")

            neighbor_and_distance_list_of_list = index.approx_knn_index.batch_search_by_vectors(
                vs=vector_dict.values(),
                k=k,
                include_distances=True,
                **index_search_kwargs
                if index_search_kwargs
                else {"ef_search": -1, "num_threads": os.cpu_count()},
            )

            logger.debug(
                f"Search on approx_knn_index of {dataset_name=}... done, filling all_pair_set now..."
            )

            for i, neighbor_distance_list in enumerate(neighbor_and_distance_list_of_list):
                left_id = other_index.vector_idx_to_id[i]
                for j, distance in neighbor_distance_list:
                    if distance <= distance_threshold:  # do NOT check for i != j here
                        right_id = index.vector_idx_to_id[j]
                        # must use sorted to always have smaller id on left of pair tuple
                        pair = tuple(sorted([left_id, right_id]))
                        all_pair_set.add(pair)

            logger.debug(f"Filling all_pair_set with {dataset_name=} done.")

        logger.debug(f"All searches done, all_pair_set filled. Found {len(all_pair_set)=} pairs.")

        return all_pair_set
