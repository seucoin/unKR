import torch
import torch.nn as nn
import torch.nn.functional as F


class GMUCp_Loss(nn.Module):
    """GMUC+ Loss

    Attributes:
        args: Some pre-set parameters, etc
        model: The UKG model for training.
    """
    def __init__(self, args, model):
        super(GMUCp_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, query_scores, query_scores_var, false_scores, query_confidence, symbolid_ic):
        """
        Args:
            query_scores: The matching scores for link prediction.
            query_scores_var: The prediction value of confidence.
            false_scores: The loss for false set.
            query_confidence: The true value for confidence in query set.
            symbolid_ic: The loss for symbolid-ic value.
        Returns:
            loss: The training loss for back propagation.
        """
        if self.args.num_neg != 1:
            false_scores = false_scores.reshape((query_scores.shape[0], self.args.num_neg))  # resize
            false_scores = torch.mean(false_scores, dim=1)
            false_scores = false_scores.reshape((query_scores.shape[0]))  # resize

        zero_torch = torch.zeros(query_confidence.shape).cuda()
        ones_torch = torch.ones(query_confidence.shape).cuda()
        query_conf_mask = torch.where(query_confidence < self.args.conf_thr, zero_torch, query_confidence)
        # ------ MSE loss -------
        mse_loss = (query_scores_var - query_confidence) ** 2
        mse_loss = self.args.mse_weight * mse_loss.mean()
        # ------ rank loss ------
        rank_loss = self.args.margin - (query_scores - false_scores)
        if self.args.if_conf:
            rank_loss = torch.mean(F.relu(rank_loss) * query_conf_mask)
        else:
            rank_loss = torch.mean(F.relu(rank_loss))
        rank_loss = self.args.rank_weight * rank_loss
        # ic loss
        symbol_ids = symbolid_ic[:, 0].squeeze().long()
        symbol_emb_var_ = self.model.symbol_emb_var(symbol_ids)
        symbol_emb_var_norm = torch.norm(symbol_emb_var_, p=2, dim=1)
        symbol_ics = symbolid_ic[:, 1].squeeze()
        ic_loss = torch.mean(torch.square(self.model.ic_loss_w * symbol_emb_var_norm + self.model.ic_loss_b - symbol_ics))
        ic_loss = self.args.ic_weight * ic_loss

        loss = rank_loss + mse_loss + ic_loss

        return loss
