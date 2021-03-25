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
        if pos_mask.bool().any() and neg_mask.bool().any():
            mat = mat / self.temperature
            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
            )
            log_prob = mat - denominator
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )
            losses = mean_log_prob_pos

            return {
                "loss": {
                    "losses": -losses,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return DotProductSimilarity()
