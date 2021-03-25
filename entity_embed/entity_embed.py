import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import SupConLoss
from tqdm.auto import tqdm

from .data_utils import utils
from .data_utils.datasets import RowDataset
from .evaluation import f1_score, pair_entity_ratio, precision_and_recall
from .indexes import ANNEntityIndex, ANNLinkageIndex
from .models import BlockerNet

logger = logging.getLogger(__name__)


class _BaseEmbed(pl.LightningModule):
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

    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, true_pair_set):
        raise NotImplementedError

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
            for tensor_dict, sequence_length_dict in row_loader:
                tensor_dict = {attr: t.to(self.device) for attr, t in tensor_dict.items()}
                embeddings = self.blocker_net(tensor_dict, sequence_length_dict)
                vector_list.extend(v.data.numpy() for v in embeddings.cpu().unbind())
                p_bar.update(1)

        self.unfreeze()

        vector_dict = dict(zip(row_dict.keys(), vector_list))
        return vector_dict


class EntityEmbed(_BaseEmbed):
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


class LinkageEmbed(_BaseEmbed):
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
