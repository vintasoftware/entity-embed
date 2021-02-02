import logging
import os
import time

import torch
from n2 import HnswIndex
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.miners import BatchHardMiner
from torch._C import Value
from tqdm.auto import tqdm

from .data_utils.datasets import PairDataset, RowDataset
from .data_utils.one_hot_encoders import OneHotEncodingInfo, RowOneHotEncoder
from .data_utils.utils import row_dict_to_id_pairs
from .evaluation import precision_and_recall
from .models import BlockerNet, get_current_signature_weights
from .trainer import train_epoch

logger = logging.getLogger(__name__)


class EntityEmbed:
    def __init__(
        self,
        attr_info_dict,
        device=None,
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
        optimizer_kwargs=None,
        row_dict=None,
    ):
        # Fix OneHotEncodingInfo from dicts and initialize RowOneHotEncoder.
        for attr, one_hot_encoding_info in list(attr_info_dict.items()):
            if not one_hot_encoding_info:
                raise ValueError(
                    f'Please set the value of "{attr}" in attr_info_dict, '
                    f"found {one_hot_encoding_info}"
                )
            if not isinstance(one_hot_encoding_info, OneHotEncodingInfo):
                attr_info_dict[attr] = OneHotEncodingInfo(**one_hot_encoding_info)
        self.row_encoder = RowOneHotEncoder(attr_info_dict=attr_info_dict, row_dict=row_dict)

        # get updated attr_info_dict,
        # because RowOneHotEncoder fills None values of alphabet and max_str_len
        # when row_dict is not None.
        self.attr_info_dict = self.row_encoder.attr_info_dict

        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.n_channels = n_channels
        self.embedding_size = embedding_size
        self.embed_dropout_p = embed_dropout_p
        self.use_attention = use_attention
        self.use_mask = use_mask
        self.model = BlockerNet(
            attr_info_dict=self.attr_info_dict,
            n_channels=self.n_channels,
            embedding_size=self.embedding_size,
            embed_dropout_p=self.embed_dropout_p,
            use_attention=self.use_attention,
            use_mask=self.use_mask,
        ).to(self.device)
        self.losser = loss_cls(**loss_kwargs if loss_kwargs else {"temperature": 0.1})
        self.miner = miner_cls(**miner_kwargs if miner_kwargs else {"distance": CosineSimilarity()})
        self.optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_kwargs if optimizer_kwargs else {"lr": 0.001}
        )

    def train(
        self,
        epochs,
        train_row_dict,
        cluster_attr,
        train_pos_pair_batch_size,
        train_neg_pair_batch_size,
        train_loader_kwargs=None,
        valid_row_dict=None,
        valid_batch_size=64,
        valid_k=10,
        valid_sim_threshold=0.5,
        valid_loader_kwargs=None,
        valid_index_build_kwargs=None,
        valid_index_search_kwargs=None,
        random_seed=42,
        log_empty_vals=False,
        show_progress=True,
        valid_show_progress=False,
    ):
        if valid_row_dict:
            valid_true_pair_set = row_dict_to_id_pairs(
                row_dict=valid_row_dict, cluster_attr=cluster_attr
            )
        else:
            valid_true_pair_set = None
        train_pair_dataset = PairDataset(
            row_dict=train_row_dict,
            cluster_attr=cluster_attr,
            row_encoder=self.row_encoder,
            pos_pair_batch_size=train_pos_pair_batch_size,
            neg_pair_batch_size=train_neg_pair_batch_size,
            random_seed=random_seed,
            log_empty_vals=log_empty_vals,
        )
        train_pair_loader = torch.utils.data.DataLoader(
            train_pair_dataset,
            batch_size=None,
            shuffle=True,
            **train_loader_kwargs
            if train_loader_kwargs
            else {"num_workers": os.cpu_count(), "multiprocessing_context": "fork"},
        )

        with tqdm(
            total=epochs * len(train_pair_loader), desc="# training", disable=not show_progress
        ) as p_bar:
            for epoch in range(epochs):
                self.model.train()
                loss_agg = 0.0
                start_time = time.time()

                idx = 0
                for idx, loss_item in enumerate(
                    train_epoch(
                        epoch=epoch,
                        device=self.device,
                        train_pair_loader=train_pair_loader,
                        model=self.model,
                        losser=self.losser,
                        miner=self.miner,
                        optimizer=self.optimizer,
                    )
                ):
                    loss_agg += loss_item
                    p_bar.update(1)
                    p_bar.set_description(
                        "# Train Epoch: %3d Time: %.3f Loss: %.3f"
                        % (
                            epoch,
                            time.time() - start_time,
                            loss_agg / (idx + 1),
                        )
                    )
                if show_progress:
                    logger.info(get_current_signature_weights(self.model))
                if valid_true_pair_set:
                    precision, recall = self.validate(
                        valid_row_dict=valid_row_dict,
                        batch_size=valid_batch_size,
                        k=valid_k,
                        sim_threshold=valid_sim_threshold,
                        valid_true_pair_set=valid_true_pair_set,
                        loader_kwargs=valid_loader_kwargs,
                        index_build_kwargs=valid_index_build_kwargs,
                        index_search_kwargs=valid_index_search_kwargs,
                        show_progress=valid_show_progress,
                    )
                    logger.info(
                        "# Train Epoch: %3d Time: %.3f Loss: %.3f, Precision: %.3f Recall: %.3f"
                        % (epoch, time.time() - start_time, loss_agg / (idx + 1), precision, recall)
                    )

    def validate(
        self,
        valid_row_dict,
        batch_size,
        k,
        sim_threshold,
        valid_true_pair_set,
        loader_kwargs=None,
        index_build_kwargs=None,
        index_search_kwargs=None,
        show_progress=True,
    ):
        valid_vector_dict = self.predict(
            row_dict=valid_row_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
        )
        ann_index = ANNEntityIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(valid_vector_dict)
        ann_index.build(index_build_kwargs=index_build_kwargs)
        valid_found_pair_set = ann_index.search_pairs(
            k=k, sim_threshold=sim_threshold, index_search_kwargs=index_search_kwargs
        )
        return precision_and_recall(valid_found_pair_set, valid_true_pair_set)

    def predict(
        self,
        row_dict,
        batch_size,
        loader_kwargs=None,
        show_progress=True,
    ):
        row_dataset = RowDataset(
            row_encoder=self.row_encoder, row_dict=row_dict, batch_size=batch_size
        )
        row_loader = torch.utils.data.DataLoader(
            row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **loader_kwargs
            if loader_kwargs
            else {"num_workers": os.cpu_count(), "multiprocessing_context": "fork"},
        )

        self.model.eval()
        with torch.no_grad():
            with tqdm(
                total=len(row_loader), desc="# batch embedding", disable=not show_progress
            ) as p_bar:
                vector_list = []
                for i, (tensor_dict, tensor_lengths_dict) in enumerate(row_loader):
                    tensor_dict = {attr: t.to(self.device) for attr, t in tensor_dict.items()}
                    vector_list.extend(
                        v.data.numpy()
                        for v in self.model(tensor_dict, tensor_lengths_dict).cpu().unbind()
                    )
                    p_bar.update(1)

        vector_dict = dict(zip(row_dict.keys(), vector_list))
        return vector_dict


class ANNEntityIndex:
    def __init__(self, embedding_size):
        self.approx_knn_index = HnswIndex(dimension=embedding_size, metric="angular")
        self.vector_idx_to_id = None

    def insert_vector_dict(self, vector_dict):
        for vector in vector_dict.values():
            self.approx_knn_index.add_data(vector)
        self.vector_idx_to_id = dict(enumerate(vector_dict.keys()))

    def build(
        self,
        index_build_kwargs=None,
    ):
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

    def search_pairs(self, k, sim_threshold, index_search_kwargs=None):
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
            for j, distance in neighbor_distance_list:
                if i != j and distance <= distance_threshold:
                    # must use sorted to always have smaller id on left of pair tuple
                    pair = tuple(sorted([self.vector_idx_to_id[i], self.vector_idx_to_id[j]]))
                    found_pair_set.add(pair)

        logger.debug(f"Building found_pair_set done. Found {len(found_pair_set)=} pairs.")

        return found_pair_set
