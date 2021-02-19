import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    # Based on: https://github.com/HobbitLong/SupContrast/blob/master/losses.py

    def __init__(self, temperature, base_temperature):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) > 2:
            raise ValueError(
                "`features` needs to be [batch_size, embedding_size], only 2 dimensions."
            )

        batch_size = features.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
