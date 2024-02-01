import torch
import torch.nn.functional as F
import torch.nn as nn


class UKGE_PSL_Loss(nn.Module):
    """Loss of UKGE with PSL

    """

    def __init__(self, args, model):
        super(UKGE_PSL_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, PSL_score, pos_sample, neg_sample, PSL_sample):
        confidence = pos_sample[:, 3]  # get confidence of positive triples
        confidence_2 = neg_sample[:, 3]  # get confidence of negative triples
        confidence_3 = PSL_sample[:, 3]  # get confidence of PSL triples

        pos_score = pos_score.squeeze()
        neg_score = neg_score.squeeze()
        PSL_score = PSL_score.squeeze()

        loss_1 = torch.sum((pos_score - confidence) ** 2) # l_pos
        tmp = torch.clamp((confidence_3 - PSL_score), min=0)
        loss_2 = 0.2 * sum(tmp**2) # l_neg
        loss_3 = torch.sum(neg_score ** 2) # l_psl

        loss = loss_1 + loss_2 + loss_3 / neg_score.shape[1]
        return loss
