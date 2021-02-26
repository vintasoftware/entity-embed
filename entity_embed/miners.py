import torch
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class SameBlockMiner(BaseTupleMiner):
    def __init__(self, true_pair_set, false_pair_set, **kwargs):
        self.true_pair_set = true_pair_set
        self.false_pair_set = false_pair_set
        super().__init__(**kwargs)

    def mine(self, labels, ids):
        idx_to_id = dict(enumerate(ids.tolist()))
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, labels)
        pos_mask = [
            (tuple(sorted([idx_to_id[idx_l], idx_to_id[idx_r]])) in self.true_pair_set)
            for (idx_l, idx_r) in zip(a1.tolist(), p.tolist())
        ]
        neg_mask = [
            (tuple(sorted([idx_to_id[idx_l], idx_to_id[idx_r]])) in self.false_pair_set)
            for (idx_l, idx_r) in zip(a2.tolist(), n.tolist())
        ]
        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]

    def forward(self, labels, ids):
        self.reset_stats()
        with torch.no_grad():
            assert labels.size(0) == ids.size(0), "Number of labels must equal number of ids"
            mining_output = self.mine(labels, ids)
        self.output_assertion(mining_output)
        return mining_output
