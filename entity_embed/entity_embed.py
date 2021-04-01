import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import SupConLoss
from tqdm.auto import tqdm

from .data_utils.datasets import RowDataset
from .early_stopping import EarlyStoppingMinEpochs, ModelCheckpointMinEpochs
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
        sim_threshold_list=[0.3, 0.5, 0.7],
        index_build_kwargs=None,
        index_search_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.row_numericalizer = row_numericalizer
        for field_config in self.row_numericalizer.field_config_dict.values():
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
            field_config_dict=self.row_numericalizer.field_config_dict,
            embedding_size=self.embedding_size,
        )
        self.loss_fn = loss_cls(**loss_kwargs if loss_kwargs else {"temperature": 0.1})
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
        loss = self.loss_fn(embeddings, labels, indices_tuple=indices_tuple)

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
            row_dict=self.trainer.datamodule.valid_row_dict,
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
            row_dict=self.trainer.datamodule.test_row_dict,
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
            "gpus": 1,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "callbacks": [early_stop_callback, checkpoint_callback],
            "reload_dataloaders_every_epoch": True,  # for shuffling ClusterDataset every epoch
        }

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
        self.blocker_net = best_model.blocker_net
        return trainer

    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, pos_pair_set):
        raise NotImplementedError

    def _evaluate_metrics(self, set_name, dataloader, row_dict, pos_pair_set):
        embedding_batch_list = []
        for tensor_dict, sequence_length_dict in dataloader:
            tensor_dict = {field: t.to(self.device) for field, t in tensor_dict.items()}
            embeddings = self(tensor_dict, sequence_length_dict)
            embedding_batch_list.append(embeddings)

        metric_dict = self._evaluate_with_ann(
            set_name=set_name,
            row_dict=row_dict,
            embedding_batch_list=embedding_batch_list,
            pos_pair_set=pos_pair_set,
        )
        return metric_dict

    def validate(self, datamodule):
        self.freeze()

        metric_dict = self._evaluate_metrics(
            set_name="valid",
            dataloader=datamodule.val_dataloader(),
            row_dict=datamodule.valid_row_dict,
            pos_pair_set=datamodule.valid_pos_pair_set,
        )
        return metric_dict

    def test(self, datamodule):
        self.freeze()

        datamodule.setup(stage="test")

        metric_dict = self._evaluate_metrics(
            set_name="test",
            dataloader=datamodule.test_dataloader(),
            row_dict=datamodule.test_row_dict,
            pos_pair_set=datamodule.test_pos_pair_set,
        )
        return metric_dict

    def predict(
        self,
        row_dict,
        batch_size,
        loader_kwargs=None,
        show_progress=True,
    ):
        self.freeze()

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

        with tqdm(
            total=len(row_loader), desc="# batch embedding", disable=not show_progress
        ) as p_bar:
            vector_list = []
            for tensor_dict, sequence_length_dict in row_loader:
                tensor_dict = {field: t.to(self.device) for field, t in tensor_dict.items()}
                embeddings = self(tensor_dict, sequence_length_dict)
                vector_list.extend(v.data.numpy() for v in embeddings.cpu().unbind())
                p_bar.update(1)

        vector_dict = dict(zip(row_dict.keys(), vector_list))
        return vector_dict


class EntityEmbed(_BaseEmbed):
    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, pos_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        vector_dict = dict(zip(row_dict.keys(), vector_list))

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
        found_pair_set = ann_index.search_pairs(
            k=ann_k, sim_threshold=sim_threshold, index_search_kwargs=index_search_kwargs
        )
        return found_pair_set


class LinkageEmbed(_BaseEmbed):
    def __init__(
        self,
        row_numericalizer,
        source_field,
        left_source,
        **kwargs,
    ):
        self.source_field = source_field
        self.left_source = left_source

        super().__init__(
            row_numericalizer=row_numericalizer,
            source_field=source_field,
            left_source=left_source,
            **kwargs,
        )

    def _evaluate_with_ann(self, set_name, row_dict, embedding_batch_list, pos_pair_set):
        vector_list = []
        for embedding_batch in embedding_batch_list:
            vector_list.extend(v.data.numpy() for v in embedding_batch.cpu().unbind())
        left_vector_dict = {}
        right_vector_dict = {}
        for (id_, row), vector in zip(row_dict.items(), vector_list):
            if row[self.source_field] == self.left_source:
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

    def predict(
        self,
        row_dict,
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
        left_vector_dict = {}
        right_vector_dict = {}
        for (id_, row), vector in zip(row_dict.items(), vector_dict.values()):
            if row[self.source_field] == self.left_source:
                left_vector_dict[id_] = vector
            else:
                right_vector_dict[id_] = vector
        return left_vector_dict, right_vector_dict

    def predict_pairs(
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
        left_vector_dict, right_vector_dict = self.predict(
            row_dict=row_dict,
            batch_size=batch_size,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
        )
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
        return found_pair_set
