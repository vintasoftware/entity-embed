import logging
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from tqdm.auto import tqdm

from .data_utils import utils
from .data_utils.datasets import RecordDataset
from .early_stopping import EarlyStoppingMinEpochs, ModelCheckpointMinEpochs
from .evaluation import f1_score, pair_entity_ratio, precision_and_recall
from .indexes import ANNEntityIndex, ANNLinkageIndex
from .models import BlockerNet

logger = logging.getLogger(__name__)


class _BaseEmbed(pl.LightningModule):
    def __init__(
        self,
        record_numericalizer,
        embedding_size=300,
        optimizer_cls=torch.optim.Adam,
        learning_rate=0.001,
        optimizer_kwargs=None,
        ann_k=10,
        sim_threshold_list=[0.4, 0.6, 0.8],
        index_build_kwargs=None,
        index_search_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.record_numericalizer = record_numericalizer
        for field_config in self.record_numericalizer.field_config_dict.values():
            vocab = field_config.vocab
            if vocab:
                # We can assume that there's only one vocab type across the
                # whole field_config_dict, so we can stop the loop once we've
                # found a field_config with a vocab
                valid_embedding_size = vocab.vectors.size(1)
                if valid_embedding_size != embedding_size:
                    raise ValueError(
                        f"Invalid embedding_size={embedding_size}. "
                        f"Expected {valid_embedding_size}, due to semantic fields."
                    )
        self.embedding_size = embedding_size
        self.blocker_net = BlockerNet(
            field_config_dict=self.record_numericalizer.field_config_dict,
            embedding_size=self.embedding_size,
        )
        self.lbda = 25
        self.mu = 25
        self.nu = 1
        self.small_val = 0.0001
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.ann_k = ann_k
        self.sim_threshold_list = sim_threshold_list
        self.index_build_kwargs = index_build_kwargs
        self.index_search_kwargs = index_search_kwargs

    def forward(self, tensor_dict, sequence_length_dict, return_field_embeddings=False):
        tensor_dict = utils.tensor_dict_to_device(tensor_dict, device=self.device)
        sequence_length_dict = utils.tensor_dict_to_device(sequence_length_dict, device=self.device)

        return F.normalize(
            self.blocker_net(tensor_dict, sequence_length_dict, return_field_embeddings), dim=-1
        )

    def training_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict, labels = batch
        embeddings = self.blocker_net(tensor_dict, sequence_length_dict)

        # invariance loss
        anchor_idx, pos_idx, __, __ = lmu.get_all_pairs_indices(labels)
        z_a = embeddings[anchor_idx]
        z_b = embeddings[pos_idx]
        inv_loss = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.small_val)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.small_val)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # covariance loss
        N, D = embeddings.size()
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = (
            cov_z_a.fill_diagonal_(0).pow_(2).sum() / D
            + cov_z_b.fill_diagonal_(0).pow_(2).sum() / D
        )

        loss = self.lbda * inv_loss + self.mu * std_loss + self.nu * cov_loss

        self.log("train_inv_loss", inv_loss)
        self.log("train_std_loss", std_loss)
        self.log("train_cov_loss", cov_loss)
        self.log("train_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.blocker_net.fix_pool_weights()
        self.log_dict(
            {
                f"pool_{field}": weight
                for field, weight in self.blocker_net.get_pool_weights().items()
            }
        )

    def validation_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict = batch
        embedding_batch = self.blocker_net(tensor_dict, sequence_length_dict)
        return embedding_batch

    def validation_epoch_end(self, outputs):
        metric_dict = self._evaluate_with_ann(
            set_name="valid",
            record_dict=self.trainer.datamodule.valid_record_dict,
            embedding_batch_list=outputs,
            pos_pair_set=self.trainer.datamodule.valid_pos_pair_set,
        )
        self.log_dict(metric_dict)

    def test_step(self, batch, batch_idx):
        tensor_dict, sequence_length_dict = batch
        return self.blocker_net(tensor_dict, sequence_length_dict)

    def test_epoch_end(self, outputs):
        metric_dict = self._evaluate_with_ann(
            set_name="test",
            record_dict=self.trainer.datamodule.test_record_dict,
            embedding_batch_list=outputs,
            pos_pair_set=self.trainer.datamodule.test_pos_pair_set,
        )
        self.log_dict(metric_dict)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        return optimizer

    def get_pool_weights(self):
        return self.blocker_net.get_pool_weights()

    def fit(
        self,
        datamodule,
        min_epochs=5,
        max_epochs=100,
        check_val_every_n_epoch=1,
        early_stop_monitor="valid_recall_at_0.3",
        early_stop_min_delta=0.0,
        early_stop_patience=20,
        early_stop_mode=None,
        early_stop_verbose=True,
        model_save_top_k=1,
        model_save_dir=None,
        model_save_verbose=False,
        tb_save_dir=None,
        tb_name=None,
        use_gpu=True,
    ):
        if early_stop_mode is None:
            if "pair_entity_ratio_at" in early_stop_monitor:
                early_stop_mode = "min"
            else:
                early_stop_mode = "max"

        early_stop_callback = EarlyStoppingMinEpochs(
            min_epochs=min_epochs,
            monitor=early_stop_monitor,
            min_delta=early_stop_min_delta,
            patience=early_stop_patience,
            mode=early_stop_mode,
            verbose=early_stop_verbose,
        )
        checkpoint_callback = ModelCheckpointMinEpochs(
            min_epochs=min_epochs,
            monitor=early_stop_monitor,
            save_top_k=model_save_top_k,
            mode=early_stop_mode,
            dirpath=model_save_dir,
            verbose=model_save_verbose,
        )
        trainer_args = {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "callbacks": [early_stop_callback, checkpoint_callback],
            "reload_dataloaders_every_epoch": True,  # for shuffling ClusterDataset every epoch
        }
        if use_gpu:
            trainer_args["gpus"] = 1

        if tb_name and tb_save_dir:
            trainer_args["logger"] = TensorBoardLogger(
                tb_save_dir,
                name=tb_name,
            )
        elif tb_name or tb_save_dir:
            raise ValueError(
                'Please provide both "tb_name" and "tb_save_dir" to enable '
                "TensorBoardLogger or omit both to disable it"
            )
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(self, datamodule)

        logger.info(
            "Loading the best validation model from "
            f"{trainer.checkpoint_callback.best_model_path}..."
        )
        self.blocker_net = None
        best_model = self.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model = best_model.to(self.device)
        self.blocker_net = best_model.blocker_net
        return trainer

    def _evaluate_with_ann(self, set_name, record_dict, embedding_batch_list, pos_pair_set):
        raise NotImplementedError

    def _evaluate_metrics(self, set_name, dataloader, record_dict, pos_pair_set):
        embedding_batch_list = []
        for tensor_dict, sequence_length_dict in dataloader:
            embeddings = self(tensor_dict, sequence_length_dict)
            embedding_batch_list.append(embeddings)

        metric_dict = self._evaluate_with_ann(
            set_name=set_name,
            record_dict=record_dict,
            embedding_batch_list=embedding_batch_list,
            pos_pair_set=pos_pair_set,
        )
        return metric_dict

    def validate(self, datamodule):
        self.freeze()

        metric_dict = self._evaluate_metrics(
            set_name="valid",
            dataloader=datamodule.val_dataloader(),
            record_dict=datamodule.valid_record_dict,
            pos_pair_set=datamodule.valid_pos_pair_set,
        )
        return metric_dict

    def test(self, datamodule):
        self.freeze()

        datamodule.setup(stage="test")

        metric_dict = self._evaluate_metrics(
            set_name="test",
            dataloader=datamodule.test_dataloader(),
            record_dict=datamodule.test_record_dict,
            pos_pair_set=datamodule.test_pos_pair_set,
        )
        return metric_dict

    def _get_record_loader(self, record_dict, batch_size, loader_kwargs):
        record_dataset = RecordDataset(
            record_numericalizer=self.record_numericalizer,
            record_dict=record_dict,
            batch_size=batch_size,
        )
        record_loader = torch.utils.data.DataLoader(
            record_dataset,
            batch_size=None,  # batch size is set on RecordDataset
            shuffle=False,
            **loader_kwargs
            if loader_kwargs
            else {"num_workers": os.cpu_count(), "multiprocessing_context": "fork"},
        )
        return record_loader

    def predict(
        self,
        record_dict,
        batch_size,
        loader_kwargs=None,
        show_progress=True,
        return_field_embeddings=False,
    ):
        self.freeze()
        record_loader = self._get_record_loader(record_dict, batch_size, loader_kwargs)

        with tqdm(
            total=len(record_loader), desc="# batch embedding", disable=not show_progress
        ) as p_bar:
            vector_list = []
            if return_field_embeddings:
                field_vector_dict = {
                    field: [] for field in self.blocker_net.field_config_dict.keys()
                }
            else:
                field_vector_dict = None

            for tensor_dict, sequence_length_dict in record_loader:
                result = self(tensor_dict, sequence_length_dict, return_field_embeddings)
                if return_field_embeddings:
                    field_embeddings_dict, embeddings = result
                    for field, field_embeddings in field_embeddings_dict.items():
                        field_vector_dict[field].extend(
                            v.data.numpy() for v in field_embeddings.cpu().unbind()
                        )
                else:
                    embeddings = result

                vector_list.extend(v.data.numpy() for v in embeddings.cpu().unbind())
                p_bar.update(1)

        vector_dict = dict(zip(record_dict.keys(), vector_list))
        if return_field_embeddings:
            field_vector_dict = {
                id_: dict(zip(field_vector_dict.keys(), field_values))
                for id_, *field_values in zip(record_dict.keys(), *field_vector_dict.values())
            }
            return vector_dict, field_vector_dict
        else:
            return vector_dict

    def interpret_attention(
        self,
        record_dict,
        batch_size,
        field,
        loader_kwargs=None,
        show_progress=True,
    ):
        self.freeze()
        record_loader = self._get_record_loader(record_dict, batch_size, loader_kwargs)

        with tqdm(
            total=len(record_loader), desc="# batch embedding", disable=not show_progress
        ) as p_bar:
            attn_scores_list = []

            for tensor_dict, sequence_length_dict in record_loader:
                field_tensor = tensor_dict[field].to(self.device)
                sequence_lengths = sequence_length_dict[field].to(self.device)

                (__, attn_scores) = self.blocker_net.field_embed_net.embed_net_dict[field]._forward(
                    field_tensor, sequence_lengths
                )

                attn_scores_list.extend(v.data.numpy() for v in attn_scores.cpu().unbind())
                p_bar.update(1)

        attn_scores_dict = dict(zip(record_dict.keys(), attn_scores_list))
        return attn_scores_dict


class EntityEmbed(_BaseEmbed):
    def _evaluate_with_ann(self, set_name, record_dict, embedding_batch_list, pos_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        vector_dict = dict(zip(record_dict.keys(), vector_list))

        ann_index = ANNEntityIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(vector_dict)
        ann_index.build(index_build_kwargs=self.index_build_kwargs)

        metric_dict = {}
        for sim_threshold in self.sim_threshold_list:
            found_pair_set = ann_index.search_pairs(
                k=self.ann_k,
                sim_threshold=sim_threshold,
                index_search_kwargs=self.index_search_kwargs,
            )

            precision, recall = precision_and_recall(found_pair_set, pos_pair_set)
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

    def predict_pairs(
        self,
        record_dict,
        batch_size,
        ann_k,
        sim_threshold,
        loader_kwargs=None,
        index_build_kwargs=None,
        index_search_kwargs=None,
        show_progress=True,
        return_field_embeddings=False,
    ):
        result = self.predict(
            record_dict=record_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
            return_field_embeddings=return_field_embeddings,
        )
        if return_field_embeddings:
            vector_dict, field_vector_dict = result
        else:
            vector_dict = result

        ann_index = ANNEntityIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(vector_dict)
        ann_index.build(index_build_kwargs=index_build_kwargs)
        found_pair_set = ann_index.search_pairs(
            k=ann_k, sim_threshold=sim_threshold, index_search_kwargs=index_search_kwargs
        )

        if return_field_embeddings:
            return found_pair_set, field_vector_dict
        else:
            return found_pair_set


class LinkageEmbed(_BaseEmbed):
    def __init__(
        self,
        record_numericalizer,
        source_field,
        left_source,
        **kwargs,
    ):
        self.source_field = source_field
        self.left_source = left_source

        super().__init__(
            record_numericalizer=record_numericalizer,
            source_field=source_field,
            left_source=left_source,
            **kwargs,
        )

    def _evaluate_with_ann(self, set_name, record_dict, embedding_batch_list, pos_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        left_vector_dict = {}
        right_vector_dict = {}
        for (id_, record), vector in zip(record_dict.items(), vector_list):
            if record[self.source_field] == self.left_source:
                left_vector_dict[id_] = vector
            else:
                right_vector_dict[id_] = vector

        ann_index = ANNLinkageIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(
            left_vector_dict=left_vector_dict, right_vector_dict=right_vector_dict
        )
        ann_index.build(
            index_build_kwargs=self.index_build_kwargs,
        )

        metric_dict = {}
        for sim_threshold in self.sim_threshold_list:
            found_pair_set = ann_index.search_pairs(
                k=self.ann_k,
                sim_threshold=sim_threshold,
                left_vector_dict=left_vector_dict,
                right_vector_dict=right_vector_dict,
                left_source=self.left_source,
                index_search_kwargs=self.index_search_kwargs,
            )

            precision, recall = precision_and_recall(found_pair_set, pos_pair_set)
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

    def _separate_left_right_vector_dict(self, record_dict, vector_dict):
        left_vector_dict = {}
        right_vector_dict = {}

        for (id_, record), vector in zip(record_dict.items(), vector_dict.values()):
            if record[self.source_field] == self.left_source:
                left_vector_dict[id_] = vector
            else:
                right_vector_dict[id_] = vector

        return left_vector_dict, right_vector_dict

    def predict(
        self,
        record_dict,
        batch_size,
        loader_kwargs=None,
        show_progress=True,
        return_field_embeddings=False,
    ):
        result = super().predict(
            record_dict=record_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
            return_field_embeddings=return_field_embeddings,
        )
        if return_field_embeddings:
            vector_dict, field_vector_dict = result
            left_field_vector_dict, right_field_vector_dict = self._separate_left_right_vector_dict(
                record_dict, field_vector_dict
            )
        else:
            vector_dict = result

        left_vector_dict, right_vector_dict = self._separate_left_right_vector_dict(
            record_dict, vector_dict
        )

        if return_field_embeddings:
            return (
                left_vector_dict,
                right_vector_dict,
                left_field_vector_dict,
                right_field_vector_dict,
            )
        else:
            return left_vector_dict, right_vector_dict

    def predict_pairs(
        self,
        record_dict,
        batch_size,
        ann_k,
        sim_threshold,
        loader_kwargs=None,
        index_build_kwargs=None,
        index_search_kwargs=None,
        show_progress=True,
        return_field_embeddings=False,
    ):
        result = self.predict(
            record_dict=record_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
            return_field_embeddings=return_field_embeddings,
        )
        if return_field_embeddings:
            (
                left_vector_dict,
                right_vector_dict,
                left_field_vector_dict,
                right_field_vector_dict,
            ) = result
        else:
            left_vector_dict, right_vector_dict = result

        ann_index = ANNLinkageIndex(embedding_size=self.embedding_size)
        ann_index.insert_vector_dict(
            left_vector_dict=left_vector_dict, right_vector_dict=right_vector_dict
        )
        ann_index.build(index_build_kwargs=index_build_kwargs)
        found_pair_set = ann_index.search_pairs(
            k=ann_k,
            sim_threshold=sim_threshold,
            left_vector_dict=left_vector_dict,
            right_vector_dict=right_vector_dict,
            left_source=self.left_source,
            index_search_kwargs=index_search_kwargs,
        )

        if return_field_embeddings:
            return found_pair_set, left_field_vector_dict, right_field_vector_dict
        else:
            return found_pair_set
