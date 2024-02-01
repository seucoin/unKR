import torch
import torch.nn.functional as F
import torch.nn as nn


class PASSLEAF_Loss(nn.Module):

    def __init__(self, args, model):
        super(PASSLEAF_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, pos_sample, semi_score = None ,semi_sample = None):

        confidence = pos_sample[:, 3]  # get confidence of positive triples
        pos_score = pos_score.squeeze()

        if semi_score is not None:
            semi_score = semi_score.squeeze()

        if semi_sample is not None:
            confidence_semi = semi_sample[:, 3] # get confidence of semi triples (the score was stored in the pool)

        loss_1 = torch.sum((pos_score - confidence) ** 2) # L_pos
        loss_2 = torch.sum(neg_score ** 2) # L_neg
        # loss is different when using semi-supervised learning
        if semi_score is not None:
            loss_3 = torch.sum((semi_score - confidence_semi) ** 2) # L_semi
            loss = loss_1 + (loss_2 + loss_3) / self.args.num_neg # L = L_pos + (L_semi + L_neg) / N_gen
        else:
            loss = loss_1 + loss_2 / neg_score.shape[1]

        return loss
