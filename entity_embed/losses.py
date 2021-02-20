import torch
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import GenericPairLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f


class SupConLoss(GenericPairLoss):
    def __init__(self, temperature, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        mat = mat / self.temperature
        mat_max, _ = torch.max(mat, dim=1, keepdim=True)
        mat = mat - mat_max.detach()

        denominator = torch.logsumexp(mat * neg_mask, dim=1, keepdim=True)
        mean_log_prob_pos = (pos_mask * (mat - denominator)).sum(dim=1) / pos_mask.sum(dim=1)
        losses = self.temperature * mean_log_prob_pos

        return {
            "loss": {
                "losses": -losses,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return DotProductSimilarity()
