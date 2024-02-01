import torch
import torch.nn.functional as F
import torch.nn as nn


class UPGAT_Loss(nn.Module):

    def __init__(self, args, model):
        super(UPGAT_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, pos_sample, pseudo_score=None, pseudo_sample=None):
        """Calculating the loss score of UPGAT model.

        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
            pos_sample: The positive samples.
            pseudo_score: The score of pseudo samples, defaults to None.
            pseudo_sample: The pseudo samples, defaults to None.

        Returns:
            loss: The training loss for back propagation.
        """
        confidence = pos_sample[:, 3]
        pos_score = pos_score.squeeze()
        loss_1 = torch.sum((pos_score - confidence) ** 2)
        loss_2 = torch.sum(neg_score ** 2)
        loss = loss_1 + loss_2 / neg_score.shape[1]
        if pseudo_sample is not None:
            pseudo_confidence = pseudo_sample[:, 3]
            pseudo_score = pseudo_score.squeeze()
            loss_3 = torch.sum((pseudo_score - pseudo_confidence) ** 2)
            loss = loss + loss_3 * self.args.train_bs / self.args.pseudo_bs

        return loss
