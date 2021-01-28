import logging

import torch

from .models import fix_signature_params

logger = logging.getLogger(__name__)


def train_attr_epoch(triplet_net, loss_func, device, train_loader, optimizer):
    triplet_net.train()

    for idx, batch in enumerate(train_loader):
        (anchor, pos, neg, pos_dist, neg_dist) = (x.to(device) for x in batch)
        optimizer.zero_grad()
        embeddings = triplet_net((anchor, pos, neg))
        loss = loss_func(embeddings, (pos_dist, neg_dist))
        loss.backward()
        optimizer.step()
        yield loss.item()


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

    for idx, (encoded_attr_tensor_list, tensor_lengths_list, labels) in enumerate(train_loader):
        encoded_attr_tensor_list, tensor_lengths_list, labels = (
            [t.to(device) for t in encoded_attr_tensor_list],
            tensor_lengths_list,
            labels.to(device),
        )
        optimizer.zero_grad()
        embeddings = model(encoded_attr_tensor_list, tensor_lengths_list)
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
        for idx, (encoded_attr_tensor_list, labels) in enumerate(valid_loader):
            encoded_attr_tensor_list, labels = (
                [t.to(device) for t in encoded_attr_tensor_list],
                labels.to(device),
            )
            embeddings = model(encoded_attr_tensor_list)
            loss = loss_func(embeddings, labels, indices_tuple=None)
            yield loss.item()
