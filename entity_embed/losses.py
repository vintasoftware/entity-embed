from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import GenericPairLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class SupConLoss(GenericPairLoss):
    def __init__(self, temperature, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        # Based on: https://github.com/HobbitLong/SupContrast/blob/6d5a3de39070249a19c62a345eea4acb5f26c0bc/losses.py  # noqa: E501
        sim_mat = mat / self.temperature
        sim_mat_max, _ = sim_mat.max(dim=1, keepdim=True)
        sim_mat = sim_mat - sim_mat_max.detach()  # for numerical stability

        denominator = lmu.logsumexp(
            sim_mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
        )
        log_prob = sim_mat - denominator
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
            pos_mask.sum(dim=1) + c_f.small_val(sim_mat.dtype)
        )
        losses = self.temperature * mean_log_prob_pos

        return {
            "loss": {
                "losses": -losses,
                "indices": c_f.torch_arange_from_size(sim_mat),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return DotProductSimilarity()
