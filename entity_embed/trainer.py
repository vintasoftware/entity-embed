import logging

import torch

from .models import fix_signature_params

logger = logging.getLogger(__name__)


def valid_attr_epoch(triplet_net, loss_func, device, valid_loader):
    triplet_net.eval()

    for idx, batch in enumerate(valid_loader):
        (anchor, pos, neg, pos_dist, neg_dist) = (x.to(device) for x in batch)
        embeddings = triplet_net((anchor, pos, neg))
        loss = loss_func(embeddings, (pos_dist, neg_dist))
        yield loss.item()


def _warn_empty_indices_tuple(epoch, idx, indices_tuple):
    if all(t.nelement() == 0 for t in indices_tuple):
        logger.warning(f"Empty indices_tuple at {epoch=} batch {idx=}")


def train_epoch(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()

    for idx, (tensor_dict, tensor_lenghts_dict, labels) in enumerate(train_loader):
        tensor_dict, tensor_lenghts_dict, labels = (
            {attr: t.to(device) for attr, t in tensor_dict.items()},
            tensor_lenghts_dict,
            labels.to(device),
        )
        optimizer.zero_grad()
        embeddings = model(tensor_dict, tensor_lenghts_dict)
        indices_tuple = mining_func(embeddings, labels)
        _warn_empty_indices_tuple(epoch, idx, indices_tuple)
        loss = loss_func(embeddings, labels, indices_tuple=indices_tuple)
        loss.backward()
        optimizer.step()
        fix_signature_params(model)
        yield loss.item()


def valid_epoch(model, loss_func, device, valid_loader):
    model.eval()

    with torch.no_grad():
        for idx, (tensor_dict, labels) in enumerate(valid_loader):
            encoded_attr_tensor_list, labels = (
                {attr: t.to(device) for attr, t in tensor_dict.items()},
                labels.to(device),
            )
            embeddings = model(encoded_attr_tensor_list)
            loss = loss_func(embeddings, labels, indices_tuple=None)
            yield loss.item()
