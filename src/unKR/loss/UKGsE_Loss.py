import torch
import torch.nn.functional as F
import torch.nn as nn


class UKGsE_Loss(nn.Module):

    def __init__(self, args, model):
        super(UKGsE_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score, neg_score, pos_sample):
        confidence = pos_sample[:, 3]  # 拿到每个三元组的置信度
        pos_score = pos_score.squeeze()

        loss_1 = torch.mean((pos_score - confidence) ** 2)
        loss_2 = torch.mean((neg_score - 1e-08) ** 2)   # MSE

        loss = loss_1 + loss_2
        return loss
