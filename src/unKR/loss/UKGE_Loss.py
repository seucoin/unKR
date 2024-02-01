import torch
import torch.nn.functional as F
import torch.nn as nn


class UKGE_Loss(nn.Module):

    def __init__(self, args, model):
        super(UKGE_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, pos_sample):
        confidence = pos_sample[:, 3]  # get confidence of each sample
        pos_score = pos_score.squeeze()

        loss_1 = torch.sum((pos_score - confidence) ** 2) # l_pos
        loss_2 = torch.sum(neg_score ** 2) # l_neg

        loss = loss_1 + loss_2 / neg_score.shape[1] # Divide l_neg by num_neg to balance the impact of positive and negative samples on loss
        return loss
