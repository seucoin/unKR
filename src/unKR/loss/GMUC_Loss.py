import torch
import torch.nn as nn
import torch.nn.functional as F


class GMUC_Loss(nn.Module):
    """GMUC Loss

    Attributes:
        args: Some pre-set parameters, etc
        model: The UKG model for training.
    """
    def __init__(self, args, model):
        super(GMUC_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, query_scores, query_scores_var, query_ae_loss, false_scores, query_confidence):
        """
        Args:
            query_scores: The matching scores for link prediction.
            query_scores_var: The prediction value of confidence.
            query_ae_loss: The loss in the matching process,
            false_scores: The loss for false set.
            query_confidence: The true value for confidence in query set.
        Returns:
            loss: The training loss for back propagation.
        """
        if self.args.num_neg != 1:
            false_scores = false_scores.reshape((query_scores.shape[0], self.args.num_neg))  # resize
            false_scores = torch.mean(false_scores, dim=1)
            false_scores = false_scores.reshape((query_scores.shape[0]))  # resize

        zero_torch = torch.zeros(query_confidence.shape).cuda()
        query_conf_mask = torch.where(query_confidence < 0.5, zero_torch, query_confidence)
        # ------ MSE loss -------
        mae_loss = (query_scores_var - query_confidence) ** 2
        mae_loss = self.args.mae_weight * mae_loss.sum()
        # ------ rank loss ------
        rank_loss = self.args.margin - (query_scores - false_scores)
        if self.args.if_conf:
            rank_loss = torch.mean(F.relu(rank_loss) * query_conf_mask)  # rank loss
        else:
            rank_loss = torch.mean(F.relu(rank_loss))
        rank_loss = self.args.rank_weight * rank_loss
        # ------ lstem loss ------
        ae_loss = self.args.ae_weight * query_ae_loss  # lstm aggregation loss
        # ------ over all loss ------
        loss = rank_loss + mae_loss + ae_loss

        return loss
