import os
import time

import torch
import tqdm
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.miners import BatchHardMiner

from .data_utils.datasets import PairDataset, RowDataset
from .data_utils.one_hot_encoders import OneHotEncodingInfo, RowOneHotEncoder
from .models import BlockerNet, get_current_signature_weights
from .trainer import train_epoch


class EntityEmbed:
    def __init__(
        self,
        attr_info_dict,
        device=None,
        loss_cls=NTXentLoss,
        loss_kwargs={"temperature": 0.1},
        miner_cls=BatchHardMiner,
        miner_kwargs={"distance": CosineSimilarity()},
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
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

        self.model = BlockerNet(self.attr_info_dict).to(self.device)
        self.losser = loss_cls(**loss_kwargs)
        self.miner = miner_cls(**miner_kwargs)
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

    def train(
        self,
        epochs,
        train_row_dict,
        cluster_id_attr,
        pos_pair_batch_size,
        neg_pair_batch_size,
        loader_kwargs={"num_workers": os.cpu_count(), "multiprocessing_context": "fork"},
        random_seed=42,
        log_empty_vals=False,
        show_progress=True,
    ):
        train_pair_dataset = PairDataset(
            row_dict=train_row_dict,
            cluster_attr=cluster_id_attr,
            row_encoder=self.row_encoder,
            pos_pair_batch_size=pos_pair_batch_size,
            neg_pair_batch_size=neg_pair_batch_size,
            random_seed=random_seed,
            log_empty_vals=log_empty_vals,
        )
        train_pair_loader = torch.utils.data.DataLoader(
            train_pair_dataset, batch_size=None, shuffle=True, **loader_kwargs
        )

        self.model.train()
        with tqdm.tqdm(
            total=epochs * len(train_pair_loader), desc="# training", disable=not show_progress
        ) as p_bar:
            for epoch in range(epochs):
                loss_agg = 0.0
                start_time = time.time()

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
                    print(get_current_signature_weights(self.model))

    def evaluate(
        self,
        row_dict,
        batch_size,
        loader_kwargs={"num_workers": os.cpu_count(), "multiprocessing_context": "fork"},
        show_progress=True,
    ):
        row_dataset = RowDataset(
            row_encoder=self.row_encoder, row_dict=row_dict, batch_size=batch_size
        )
        row_loader = torch.utils.data.DataLoader(
            row_dataset,
            batch_size=None,  # batch size is set on RowDataset
            shuffle=False,
            **loader_kwargs,
        )

        self.model.eval()
        with tqdm.tqdm(
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

        return vector_list
