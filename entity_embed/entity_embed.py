import logging
import os

import pytorch_lightning as pl
import torch
from n2 import HnswIndex
from pytorch_metric_learning.distances import DotProductSimilarity
from tqdm.auto import tqdm

from .data_utils import utils
from .data_utils.datasets import ClusterDataset, RowDataset
from .evaluation import f1_score, pair_entity_ratio, precision_and_recall
from .helpers import build_index_build_kwargs, build_index_search_kwargs, build_loader_kwargs
from .losses import SupConLoss
from .models import BlockerNet

logger = logging.getLogger(__name__)


class DeduplicationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        row_dict,
        cluster_attr,
        row_numericalizer,
        batch_size,
        eval_batch_size,
        train_cluster_len,
        valid_cluster_len,
        test_cluster_len,
        pair_loader_kwargs=None,
        row_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__()
        self.row_dict = row_dict
        self.cluster_attr = cluster_attr
        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_cluster_len = train_cluster_len
        self.valid_cluster_len = valid_cluster_len
        self.test_cluster_len = test_cluster_len
        self.pair_loader_kwargs = build_loader_kwargs(pair_loader_kwargs)
        self.row_loader_kwargs = build_loader_kwargs(row_loader_kwargs)
        self.random_seed = random_seed

        self.valid_true_pair_set = None
        self.test_true_pair_set = None
        self.train_row_dict = None
        self.valid_row_dict = None
        self.test_row_dict = None

    def setup(self, stage=None):
        cluster_dict = utils.row_dict_to_cluster_dict(self.row_dict, self.cluster_attr)

        train_cluster_dict, valid_cluster_dict, test_cluster_dict = utils.split_clusters(
            cluster_dict,
            train_len=self.train_cluster_len,
            valid_len=self.valid_cluster_len,
            test_len=self.test_cluster_len,
            random_seed=self.random_seed,
        )
        self.valid_true_pair_set = utils.cluster_dict_to_id_pairs(valid_cluster_dict)
        self.test_true_pair_set = utils.cluster_dict_to_id_pairs(test_cluster_dict)
        logger.info("Train pair count: %s", utils.count_cluster_dict_pairs(train_cluster_dict))
        logger.info("Valid pair count: %s", len(self.valid_true_pair_set))
        logger.info("Test pair count: %s", len(self.test_true_pair_set))

        (
            self.train_row_dict,
            self.valid_row_dict,
            self.test_row_dict,
        ) = utils.cluster_dicts_to_row_dicts(
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
            batch_size=self.batch_size,
            max_cluster_size_in_batch=self.batch_size // 3,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving contrastive learning:
            random_seed=self.random_seed + self.trainer.current_epoch,
        )
        train_cluster_loader = torch.utils.data.DataLoader(
            train_cluster_dataset,
            batch_size=None,  # batch size is set on ClusterDataset
            shuffle=False,  # shuffling is implemented on ClusterDataset
            **self.pair_loader_kwargs,
        )
        return train_cluster_loader

    def val_dataloader(self):
        valid_row_dataset = RowDataset(
            row_dict=self.valid_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.eval_batch_size,
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
            batch_size=self.eval_batch_size,
        )
        test_row_loader = torch.utils.data.DataLoader(
            test_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.row_loader_kwargs,
        )
        return test_row_loader


class LinkageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        row_dict,
        left_id_set,
        right_id_set,
        row_numericalizer,
        batch_size,
        eval_batch_size,
        cluster_attr=None,
        true_pair_set=None,
        train_cluster_len=None,
        valid_cluster_len=None,
        test_cluster_len=None,
        train_true_pair_set=None,
        valid_true_pair_set=None,
        test_true_pair_set=None,
        pair_loader_kwargs=None,
        row_loader_kwargs=None,
        random_seed=42,
    ):
        super().__init__()

        self.row_dict = row_dict
        self.left_id_set = left_id_set
        self.right_id_set = right_id_set
        self.row_numericalizer = row_numericalizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.pair_loader_kwargs = build_loader_kwargs(pair_loader_kwargs)
        self.row_loader_kwargs = build_loader_kwargs(row_loader_kwargs)
        self.random_seed = random_seed
        if train_true_pair_set:
            if valid_true_pair_set is None:
                raise ValueError(
                    "valid_true_pair_set can't be None when train_true_pair_set is provided"
                )
            if test_true_pair_set is None:
                raise ValueError(
                    "test_true_pair_set can't be None when train_true_pair_set is provided"
                )
            self.train_true_pair_set = train_true_pair_set
            self.valid_true_pair_set = valid_true_pair_set
            self.test_true_pair_set = test_true_pair_set
        elif true_pair_set:
            if train_cluster_len is None:
                raise ValueError("train_cluster_len can't be None")
            if valid_cluster_len is None:
                raise ValueError("valid_cluster_len can't be None")
            __, cluster_dict = utils.id_pairs_to_cluster_mapping_and_dict(true_pair_set)
            self._split_clusters(
                cluster_dict=cluster_dict,
                train_cluster_len=train_cluster_len,
                valid_cluster_len=valid_cluster_len,
                test_cluster_len=test_cluster_len,
            )
        elif cluster_attr:
            if train_cluster_len is None:
                raise ValueError("train_cluster_len can't be None")
            if valid_cluster_len is None:
                raise ValueError("valid_cluster_len can't be None")
            cluster_dict = utils.row_dict_to_cluster_dict(row_dict, cluster_attr)
            self._split_clusters(
                cluster_dict=cluster_dict,
                train_cluster_len=train_cluster_len,
                valid_cluster_len=valid_cluster_len,
                test_cluster_len=test_cluster_len,
            )
        else:
            raise Exception("Please set one of train_true_pair_set, true_pair_set or cluster_attr")
        self.train_row_dict = None
        self.valid_row_dict = None
        self.test_row_dict = None

    def _split_clusters(self, cluster_dict, train_cluster_len, valid_cluster_len, test_cluster_len):
        train_cluster_dict, valid_cluster_dict, test_cluster_dict = utils.split_clusters(
            cluster_dict,
            train_len=train_cluster_len,
            valid_len=valid_cluster_len,
            test_len=test_cluster_len,
            random_seed=self.random_seed,
        )
        self.train_true_pair_set = utils.cluster_dict_to_id_pairs(
            cluster_dict=train_cluster_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
        )
        self.valid_true_pair_set = utils.cluster_dict_to_id_pairs(
            cluster_dict=valid_cluster_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
        )
        self.test_true_pair_set = utils.cluster_dict_to_id_pairs(
            cluster_dict=test_cluster_dict,
            left_id_set=self.left_id_set,
            right_id_set=self.right_id_set,
        )

    def setup(self, stage=None):
        logger.info("Train pair count: %s", len(self.train_true_pair_set))
        logger.info("Valid pair count: %s", len(self.valid_true_pair_set))
        logger.info("Test pair count: %s", len(self.test_true_pair_set))

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
            batch_size=self.batch_size,
            max_cluster_size_in_batch=self.batch_size // 3,
            # Combined with reload_dataloaders_every_epoch on Trainer,
            # this re-shuffles training batches every epoch,
            # therefore improving contrastive learning:
            random_seed=self.random_seed + self.trainer.current_epoch,
        )
        train_cluster_loader = torch.utils.data.DataLoader(
            train_cluster_dataset,
            batch_size=None,  # batch size is set on ClusterDataset
            shuffle=False,  # shuffling is implemented on ClusterDataset
            **self.pair_loader_kwargs,
        )
        return train_cluster_loader

    def val_dataloader(self):
        valid_row_dataset = RowDataset(
            row_dict=self.valid_row_dict,
            row_numericalizer=self.row_numericalizer,
            batch_size=self.eval_batch_size,
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
            batch_size=self.eval_batch_size,
        )
        test_row_loader = torch.utils.data.DataLoader(
            test_row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **self.row_loader_kwargs,
        )
        return test_row_loader

    def separate_dict_left_right(self, d):
        return utils.separate_dict_left_right(
            d, left_id_set=self.left_id_set, right_id_set=self.right_id_set
        )


class EntityEmbed(pl.LightningModule):
    def __init__(
        self,
        row_numericalizer,
        embedding_size=300,
        loss_cls=SupConLoss,
        loss_kwargs=None,
        miner_cls=None,
        miner_kwargs=None,
        optimizer_cls=torch.optim.Adam,
        learning_rate=0.001,
        optimizer_kwargs=None,
        ann_k=10,
        sim_threshold_list=[0.5, 0.7, 0.9],
        eval_with_clusters=True,
        index_build_kwargs=None,
        index_search_kwargs=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.row_numericalizer = row_numericalizer
        for numericalize_info in self.row_numericalizer.attr_info_dict.values():
            vocab = numericalize_info.vocab
            if vocab:
                # We can assume that there's only one vocab type across the
                # whole attr_info_dict, so we can stop the loop once we've
                # found a numericalize_info with a vocab
                valid_embedding_size = vocab.vectors.size(1)
                if valid_embedding_size != embedding_size:
                    raise ValueError(
                        f"Invalid embedding_size={embedding_size}. "
                        f"Expected {valid_embedding_size}, due to semantic fields."
                    )
        self.embedding_size = embedding_size
        self.blocker_net = BlockerNet(
            attr_info_dict=self.row_numericalizer.attr_info_dict,
            embedding_size=self.embedding_size,
        )
        self.losser = loss_cls(**loss_kwargs if loss_kwargs else {"temperature": 0.1})
        if miner_cls:
            self.miner = miner_cls(
                **miner_kwargs if miner_kwargs else {"distance": DotProductSimilarity()}
            )
        else:
            self.miner = None
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.ann_k = ann_k
        self.sim_threshold_list = sim_threshold_list
        self.eval_with_clusters = eval_with_clusters
        self.index_build_kwargs = index_build_kwargs
        self.index_search_kwargs = index_search_kwargs

    def forward(self, tensor_dict, sequence_length_dict):
        return self.blocker_net(tensor_dict, sequence_length_dict)

    def _warn_if_empty_indices_tuple(self, indices_tuple, batch_idx):
        with torch.no_grad():
            if all(t.nelement() == 0 for t in indices_tuple):
                logger.warning(
                    f"Found empty indices_tuple at self.current_epoch={self.current_epoch}, "
                    f"batch_idx={batch_idx}"
                )

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

        ann_index = ANNEntityIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(vector_dict)
        ann_index.build(index_build_kwargs=self.index_build_kwargs)

        metric_dict = {}
        for sim_threshold in self.sim_threshold_list:
            if self.eval_with_clusters:
                __, cluster_dict = ann_index.search_clusters(
                    k=self.ann_k,
                    sim_threshold=sim_threshold,
                    index_search_kwargs=self.index_search_kwargs,
                )
                found_pair_set = utils.cluster_dict_to_id_pairs(cluster_dict)
            else:
                found_pair_set = ann_index.search_pairs(
                    k=self.ann_k,
                    sim_threshold=sim_threshold,
                    index_search_kwargs=self.index_search_kwargs,
                )

            precision, recall = precision_and_recall(found_pair_set, true_pair_set)
            metric_dict.update(
                {
                    f"{set_name}_precision_at_{sim_threshold}": precision,
                    f"{set_name}_recall_at_{sim_threshold}": recall,
                    f"{set_name}_f1_at_{sim_threshold}": f1_score(precision, recall),
                    f"{set_name}_pair_entity_ratio_at_{sim_threshold}": pair_entity_ratio(
                        len(found_pair_set), len(vector_list)
                    ),
                }
            )
        metric_dict = dict(sorted(metric_dict.items(), key=lambda kv: kv[0]))
        return metric_dict

    def validation_epoch_end(self, outputs):
        metric_dict = self._evaluate_with_ann(
            set_name="valid",
            row_dict=self.trainer.datamodule.valid_row_dict,
            embedding_batch_list=outputs,
            true_pair_set=self.trainer.datamodule.valid_true_pair_set,
        )
        self.log_dict(metric_dict)

    def test_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict = batch
        return self.blocker_net(tensor_dict, sequence_length_dict)

    def test_epoch_end(self, outputs):
        metric_dict = self._evaluate_with_ann(
            set_name="test",
            row_dict=self.trainer.datamodule.test_row_dict,
            embedding_batch_list=outputs,
            true_pair_set=self.trainer.datamodule.test_true_pair_set,
        )
        self.log_dict(metric_dict)

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
        show_progress=True,
    ):
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

        self.freeze()

        with tqdm(
            total=len(row_loader), desc="# batch embedding", disable=not show_progress
        ) as p_bar:
            vector_list = []
            for i, (tensor_dict, sequence_length_dict) in enumerate(row_loader):
                tensor_dict = {attr: t.to(self.device) for attr, t in tensor_dict.items()}
                embeddings = self.blocker_net(tensor_dict, sequence_length_dict)
                vector_list.extend(v.data.numpy() for v in embeddings.cpu().unbind())
                p_bar.update(1)

        self.unfreeze()

        vector_dict = dict(zip(row_dict.keys(), vector_list))
        return vector_dict

    def predict_clusters(
        self,
        row_dict,
        batch_size,
        ann_k,
        sim_threshold,
        loader_kwargs=None,
        index_build_kwargs=None,
        index_search_kwargs=None,
        show_progress=True,
    ):
        vector_dict = self.predict(
            row_dict=row_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
        )
        ann_index = ANNEntityIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(vector_dict)
        ann_index.build(index_build_kwargs=index_build_kwargs)
        cluster_mapping, cluster_dict = ann_index.search_clusters(
            k=ann_k, sim_threshold=sim_threshold, index_search_kwargs=index_search_kwargs
        )
        return cluster_mapping, cluster_dict


def validate_best(trainer):
    logger.info("Validating best model...")
    old_model = trainer.get_model()
    model = old_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    model.freeze()
    model.trainer = trainer

    embedding_batch_list = []
    for tensor_dict, sequence_length_dict in trainer.datamodule.val_dataloader():
        tensor_dict = {attr: t.to(model.device) for attr, t in tensor_dict.items()}
        embeddings = model.blocker_net(tensor_dict, sequence_length_dict)
        embedding_batch_list.append(embeddings)
    metric_dict = model._evaluate_with_ann(
        set_name="valid",
        row_dict=trainer.datamodule.valid_row_dict,
        embedding_batch_list=embedding_batch_list,
        true_pair_set=trainer.datamodule.valid_true_pair_set,
    )

    return metric_dict


class LinkageEmbed(EntityEmbed):
    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, true_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        vector_dict = dict(zip(row_dict.keys(), vector_list))
        left_vector_dict, right_vector_dict = self.trainer.datamodule.separate_dict_left_right(
            vector_dict
        )

        ann_index = ANNLinkageIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(
            left_vector_dict=left_vector_dict, right_vector_dict=right_vector_dict
        )
        ann_index.build(
            index_build_kwargs=self.index_build_kwargs,
        )

        metric_dict = {}
        for sim_threshold in self.sim_threshold_list:
            if self.eval_with_clusters:
                __, cluster_dict = ann_index.search_clusters(
                    k=self.ann_k,
                    sim_threshold=sim_threshold,
                    left_vector_dict=left_vector_dict,
                    right_vector_dict=right_vector_dict,
                    index_search_kwargs=self.index_search_kwargs,
                )
                found_pair_set = utils.cluster_dict_to_id_pairs(
                    cluster_dict,
                    left_id_set=self.trainer.datamodule.left_id_set,
                    right_id_set=self.trainer.datamodule.right_id_set,
                )
            else:
                found_pair_set = ann_index.search_pairs(
                    k=self.ann_k,
                    sim_threshold=sim_threshold,
                    left_vector_dict=left_vector_dict,
                    right_vector_dict=right_vector_dict,
                    index_search_kwargs=self.index_search_kwargs,
                )

            precision, recall = precision_and_recall(found_pair_set, true_pair_set)
            metric_dict.update(
                {
                    f"{set_name}_precision_at_{sim_threshold}": precision,
                    f"{set_name}_recall_at_{sim_threshold}": recall,
                    f"{set_name}_f1_at_{sim_threshold}": f1_score(precision, recall),
                    f"{set_name}_pair_entity_ratio_at_{sim_threshold}": pair_entity_ratio(
                        len(found_pair_set), len(vector_list)
                    ),
                }
            )
        metric_dict = dict(sorted(metric_dict.items(), key=lambda kv: kv[0]))
        return metric_dict

    def predict(
        self,
        row_dict,
        left_id_set,
        right_id_set,
        batch_size,
        loader_kwargs=None,
        show_progress=True,
    ):
        vector_dict = super().predict(
            row_dict=row_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
        )
        left_vector_dict, right_vector_dict = utils.separate_dict_left_right(
            vector_dict, left_id_set=left_id_set, right_id_set=right_id_set
        )
        return left_vector_dict, right_vector_dict

    def predict_clusters(
        self,
        row_dict,
        left_id_set,
        right_id_set,
        batch_size,
        ann_k,
        sim_threshold,
        loader_kwargs=None,
        index_build_kwargs=None,
        index_search_kwargs=None,
        show_progress=True,
    ):
        left_vector_dict, right_vector_dict = self.predict(
            row_dict=row_dict,
            left_id_set=left_id_set,
            right_id_set=right_id_set,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
        )
        ann_index = ANNLinkageIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(
            left_vector_dict=left_vector_dict, right_vector_dict=right_vector_dict
        )
        ann_index.build(index_build_kwargs=index_build_kwargs)
        cluster_mapping, cluster_dict = ann_index.search_clusters(
            k=ann_k,
            sim_threshold=sim_threshold,
            left_vector_dict=left_vector_dict,
            right_vector_dict=right_vector_dict,
            index_search_kwargs=index_search_kwargs,
        )
        return cluster_mapping, cluster_dict


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

    def search_clusters(self, k, sim_threshold, index_search_kwargs=None):
        found_pair_set = self.search_pairs(
            k=k, sim_threshold=sim_threshold, index_search_kwargs=index_search_kwargs
        )
        cluster_mapping, cluster_dict = utils.id_pairs_to_cluster_mapping_and_dict(found_pair_set)
        return cluster_mapping, cluster_dict


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
            raise ValueError(f"sim_threshold={sim_threshold} must be <= 1 and >= 0")

        index_search_kwargs = build_index_search_kwargs(index_search_kwargs)
        distance_threshold = 1 - sim_threshold
        all_pair_set = set()

        for dataset_name, index, vector_dict, other_index in [
            (left_dataset_name, self.left_index, right_vector_dict, self.right_index),
            (right_dataset_name, self.right_index, left_vector_dict, self.left_index),
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
                        if dataset_name == left_dataset_name:
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

    def search_clusters(
        self, k, sim_threshold, left_vector_dict, right_vector_dict, index_search_kwargs=None
    ):
        found_pair_set = self.search_pairs(
            k=k,
            sim_threshold=sim_threshold,
            left_vector_dict=left_vector_dict,
            right_vector_dict=right_vector_dict,
            index_search_kwargs=index_search_kwargs,
        )
        cluster_mapping, cluster_dict = utils.id_pairs_to_cluster_mapping_and_dict(found_pair_set)
        return cluster_mapping, cluster_dict
