import logging

import torch

from .models import fix_signature_params

logger = logging.getLogger(__name__)


def _warn_empty_indices_tuple(epoch, idx, indices_tuple):
    if all(t.nelement() == 0 for t in indices_tuple):
        logger.warning(f"Empty indices_tuple at {epoch=} batch {idx=}")


def train_epoch(epoch, device, train_pair_loader, model, losser, miner, optimizer):
    model.train()

    for idx, (tensor_dict, tensor_lengths_dict, labels) in enumerate(train_pair_loader):
        tensor_dict, tensor_lengths_dict, labels = (
            {attr: t.to(device) for attr, t in tensor_dict.items()},
            tensor_lengths_dict,
            labels.to(device),
        )
        optimizer.zero_grad()
        embeddings = model(tensor_dict, tensor_lengths_dict)
        indices_tuple = miner(embeddings, labels)
        _warn_empty_indices_tuple(epoch, idx, indices_tuple)
        loss = losser(embeddings, labels, indices_tuple=indices_tuple)
        loss.backward()
        optimizer.step()
        fix_signature_params(model)
        yield loss.item()
