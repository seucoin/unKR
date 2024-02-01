import torch
import torch.nn.functional as F
import torch.nn as nn


class GTransE_Loss(nn.Module):

    def __init__(self, args, model):
        super(GTransE_Loss, self).__init__()
        self.args = args
        self.model = model
        self.margin = args.margin
        self.alpha = args.alpha

    def forward(self, pos_score, neg_score, pos_sample):
        confidence = pos_sample[:, 3]
        modified_margin = self.margin * (confidence ** self.alpha)
        modified_margin = modified_margin.unsqueeze(1)

        diff = neg_score - pos_score
        loss = F.relu(modified_margin + diff)

        return loss.mean()
