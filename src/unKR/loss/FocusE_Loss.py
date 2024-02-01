import torch
import torch.nn as nn


class FocusE_Loss(nn.Module):
    """The loss function of FocusE

    Attributes:
        args: Some pre-set parameters, etc
        model: The FocusE model for training.
    """
    def __init__(self, args, model):
        super(FocusE_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, pos_sample):
        """Calculating the loss, which includes the negative log-likelihood loss and L3 regularization.

        Args:
            pos_score: Tensor of scores for positive samples.
            neg_score: Tensor of scores for negative samples.
            pos_sample: Tensor of positive samples.

        Returns:
            The total loss of FocusE.
        """
        # neg_score = torch.mean(neg_score, dim=1).unsqueeze(1)
        neg_score = torch.sum(torch.exp(neg_score), dim=1).unsqueeze(1)

        # loss = torch.log(torch.exp(pos_score) / (torch.exp(pos_score) + torch.exp(neg_score)))
        loss = torch.log(torch.exp(pos_score) / (torch.exp(pos_score) + neg_score))
        loss = torch.sum(loss)

        # Use L3 regularization for ComplEx and DistMult
        regularization = self.args.regularization * (
            self.model.ent_emb.weight.norm(p = 3)**3 + \
            self.model.rel_emb.weight.norm(p = 3)**3
        )

        loss = -loss + regularization

        return loss