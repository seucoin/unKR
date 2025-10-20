import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.stats as stats
import numpy as np

# min_old, max_old = 0.41562477969017253, 0.604478177772596  # sigma06
# min_new, max_new = 0.1, 1.0

def generate_distribution(conf, sigma, size):
    gauss = stats.norm(conf, sigma)
    pdf = gauss.pdf(np.linspace(0, 1, size))
    label_dis = pdf/np.sum(pdf)
    return label_dis

class ssCDL_loss(nn.Module):
    # loss function of ssCDL
    def __init__(self, args, model):
        super(ssCDL_loss, self).__init__()
        self.args = args
        self.model = model
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred_distribution, pos_sample, conf_ldl, neg_pred_distribution, rank_score_pos, rank_score_neg, flag, semi_sample = None, pred_semi_distribution = None, conf_ldl_semi = None, rank_score_semi=None, neg_pred_semi_distribution= None, rank_score_semi_neg= None):
        if flag == 'main':
            if semi_sample == None:
                # loss to update CDL-RL without pseudo labeled data
                neg_pred_distribution = neg_pred_distribution.to('cuda')
                weights = torch.linspace(0, 1, steps=101).to('cuda')
                weights_neg = torch.linspace(0, 0, steps=101).to('cuda')
                confidence = pos_sample[:, 3]  # get confidence of each sample
                confidence = confidence.to(pred_distribution.dtype)
                expected_values = (pred_distribution * weights).sum(dim=2).squeeze()
                expected_values = (expected_values - self.args.lower_bound) * (1.0 - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1
                expected_values_neg = (neg_pred_distribution * weights).sum(dim=2).squeeze()
                rank_score_pos = rank_score_pos.squeeze(-1)
                rank_score_neg = rank_score_neg.squeeze(-1)
                expected_values_expand = expected_values.unsqueeze(1).repeat(1, self.args.num_neg)
                rank_expected_values_expand = rank_score_pos.repeat(1, self.args.num_neg)
                rank_scores = confidence.unsqueeze(1) * (0.1 + rank_score_neg - rank_expected_values_expand)
                condition = rank_scores > 0
                rank_scores[~condition] = 0
                MSE_pos = torch.sum((expected_values - confidence) ** 2) # get MSE
                MSE_neg = torch.sum(expected_values_neg**2)
                conf_ldl_stack = torch.stack(conf_ldl)
                pred_distribution = pred_distribution.squeeze(1)
                conf_ldl_stack = conf_ldl_stack.to(pred_distribution.dtype)
                pred_distribution = torch.clamp(pred_distribution, min=1e-9)
                conf_ldl_stack = torch.clamp(conf_ldl_stack, min=1e-9)
                kl_divergence = torch.sum(conf_ldl_stack * torch.log(conf_ldl_stack / pred_distribution)) # get kl divergence
                neg_pred_distribution = neg_pred_distribution.reshape(-1,101)
                loss_2 = MSE_pos + MSE_neg/self.args.num_neg
                loss_3 = rank_scores.sum()
                sigma1 = torch.exp(self.model.log_sigma1)
                sigma2 = torch.exp(self.model.log_sigma2)

                task1_loss = (1 / (2 * sigma1 ** 2)) * (kl_divergence + MSE_pos) + self.model.log_sigma1

                task2_loss = (1 / (2 * sigma2 ** 2)) * (loss_3 / 10) + self.model.log_sigma2
                # combine L_CP and L_LP to get complete loss
                loss = task1_loss + task2_loss

            else:
                # loss to update CDL-RL with pseudo labeled data
                neg_pred_distribution = neg_pred_distribution.to('cuda')
                weights = torch.linspace(0, 1, steps=101).to('cuda')
                weights_neg = torch.linspace(0, 0, steps=101).to('cuda')
                confidence = pos_sample[:, 3]  # get confidence of each sample
                confidence = confidence.to(pred_distribution.dtype)
                confidence_semi = semi_sample[:, 3]
                confidence_semi = confidence_semi.to(pred_distribution.dtype)
                expected_values = (pred_distribution * weights).sum(dim=2).squeeze()
                expected_values = (expected_values - self.args.lower_bound) * (1.0  - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1
                expected_values_neg = (neg_pred_distribution * weights).sum(dim=2).squeeze()
                expected_values_semi = (pred_semi_distribution * weights).sum(dim=2).squeeze()
                rank_score_pos = rank_score_pos.squeeze(-1)
                rank_score_neg = rank_score_neg.squeeze(-1)
                rank_score_semi = rank_score_semi.squeeze(-1)
                rank_score_semi_neg = rank_score_semi_neg.squeeze(-1)
                expected_values_expand = expected_values.unsqueeze(1).repeat(1, self.args.num_neg)
                rank_expected_values_expand = rank_score_pos.repeat(1, self.args.num_neg)
                rank_expected_values_expand_semi = rank_score_semi.repeat(1, self.args.num_neg)
                rank_scores = confidence.unsqueeze(1) * (0.1 + rank_score_neg - rank_expected_values_expand)
                condition = rank_scores > 0
                rank_scores[~condition] = 0
                rank_scores_semi = confidence_semi.unsqueeze(1) * (0.1 + rank_score_semi_neg - rank_expected_values_expand_semi)
                condition = rank_scores_semi > 0
                rank_scores_semi[~condition] = 0
                MSE_pos = torch.sum((expected_values - confidence) ** 2)  # get MSE for positive data
                MSE_semi = torch.sum((expected_values_semi - confidence_semi) ** 2) # get MSE for pseudo labeled data
                MSE_neg = torch.sum(expected_values_neg ** 2)
                conf_ldl_stack = torch.stack(conf_ldl)
                pred_distribution = pred_distribution.squeeze(1)
                conf_ldl_stack = conf_ldl_stack.to(pred_distribution.dtype)
                conf_ldl_stack_semi = torch.stack(conf_ldl_semi)
                pred_semi_distribution = pred_semi_distribution.squeeze(1)
                conf_ldl_stack_semi = conf_ldl_stack_semi.to(pred_semi_distribution.dtype)
                pred_distribution = torch.clamp(pred_distribution, min=1e-9)
                conf_ldl_stack = torch.clamp(conf_ldl_stack, min=1e-9)
                kl_divergence = torch.sum(conf_ldl_stack * torch.log(conf_ldl_stack / pred_distribution)) # get kl_divergence for positive data

                pred_semi_distribution = torch.clamp(pred_semi_distribution, min=1e-9)
                conf_ldl_stack_semi = torch.clamp(conf_ldl_stack_semi, min=1e-9)

                kl_divergence_semi = torch.sum(
                    conf_ldl_stack_semi * torch.log(conf_ldl_stack_semi / pred_semi_distribution))  # get kl_divergence for pseudo labeled data

                loss_2 = MSE_pos + MSE_neg / self.args.num_neg

                loss_3 = rank_scores.sum()

                sigma1 = torch.exp(self.model.log_sigma1)
                sigma2 = torch.exp(self.model.log_sigma2)

                # redefine L_CP when using pseudo labeled data
                task1_loss = (1 / (2 * sigma1 ** 2)) * ((kl_divergence + kl_divergence_semi * self.args.weightloss) + MSE_pos + MSE_semi * self.args.weightloss) + self.model.log_sigma1

                task2_loss = (1 / (2 * sigma2 ** 2)) * (loss_3 / 10) + self.model.log_sigma2

                loss = task1_loss + task2_loss
        else: # meta
            if semi_sample == None:
                # loss to update CDL-RL without pseudo labeled data
                neg_pred_distribution = neg_pred_distribution.to('cuda')
                weights = torch.linspace(0, 1, steps=101).to('cuda')
                weights_neg = torch.linspace(0, 0, steps=101).to('cuda')
                confidence = pos_sample[:, 3]  # get confidence of each sample
                confidence = confidence.to(pred_distribution.dtype)
                expected_values = (pred_distribution * weights).sum(dim=2).squeeze()
                expected_values = (expected_values - self.args.lower_bound) * (1.0  - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1
                expected_values_neg = (neg_pred_distribution * weights).sum(dim=2).squeeze()
                rank_score_pos = rank_score_pos.squeeze(-1)
                rank_score_neg = rank_score_neg.squeeze(-1)
                expected_values_expand = expected_values.unsqueeze(1).repeat(1, self.args.num_neg)
                rank_expected_values_expand = rank_score_pos.repeat(1, self.args.num_neg)
                rank_scores = confidence.unsqueeze(1) * (0.1 + rank_score_neg - rank_expected_values_expand)
                condition = rank_scores > 0
                rank_scores[~condition] = 0
                MSE_pos = torch.sum((expected_values - confidence) ** 2)
                MSE_neg = torch.sum(expected_values_neg ** 2)
                conf_ldl_stack = torch.stack(conf_ldl)
                pred_distribution = pred_distribution.squeeze(1)
                conf_ldl_stack = conf_ldl_stack.to(pred_distribution.dtype)

                pred_distribution = torch.clamp(pred_distribution, min=1e-9)
                conf_ldl_stack = torch.clamp(conf_ldl_stack, min=1e-9)

                kl_divergence = torch.sum(conf_ldl_stack * torch.log(conf_ldl_stack / pred_distribution))

                loss_3 = rank_scores.sum()


                sigma1 = torch.exp(self.model.log_sigma1)
                sigma2 = torch.exp(self.model.log_sigma2)

                task1_loss = (1 / (2 * sigma1 ** 2)) * (kl_divergence + MSE_pos) + self.model.log_sigma1

                task2_loss = (1 / (2 * sigma2 ** 2)) * (loss_3 / 10) + self.model.log_sigma2
                loss = task1_loss + task2_loss

            else:
                # loss to update CDL-RL with pseudo labeled data
                neg_pred_distribution = neg_pred_distribution.to('cuda')
                weights = torch.linspace(0, 1, steps=101).to('cuda')
                weights_neg = torch.linspace(0, 0, steps=101).to('cuda')
                confidence = pos_sample[:, 3]  # get confidence of each sample
                confidence = confidence.to(pred_distribution.dtype)
                confidence_semi = semi_sample[:, 3]
                confidence_semi = confidence_semi.to(pred_distribution.dtype)

                expected_values = (pred_distribution * weights).sum(dim=2).squeeze()
                expected_values = (expected_values - self.args.lower_bound) * (1.0  - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1
                expected_values_neg = (neg_pred_distribution * weights).sum(dim=2).squeeze()
                expected_values_semi = (pred_semi_distribution * weights).sum(dim=2).squeeze()
                rank_score_pos = rank_score_pos.squeeze(-1)
                rank_score_neg = rank_score_neg.squeeze(-1)
                rank_score_semi = rank_score_semi.squeeze(-1)
                rank_score_semi_neg = rank_score_semi_neg.squeeze(-1)
                expected_values_expand = expected_values.unsqueeze(1).repeat(1, self.args.num_neg)
                rank_expected_values_expand = rank_score_pos.repeat(1, self.args.num_neg)
                rank_expected_values_expand_semi = rank_score_semi.repeat(1, self.args.num_neg)
                rank_scores = confidence.unsqueeze(1) * (0.1 + rank_score_neg - rank_expected_values_expand)
                condition = rank_scores > 0
                rank_scores[~condition] = 0
                rank_scores_semi = confidence_semi.unsqueeze(1) * (0.1 + rank_score_semi_neg - rank_expected_values_expand_semi)
                condition = rank_scores_semi > 0
                rank_scores_semi[~condition] = 0
                MSE_pos = torch.sum((expected_values - confidence) ** 2)
                MSE_semi = torch.sum((expected_values_semi - confidence_semi) ** 2)
                MSE_neg = torch.sum(expected_values_neg ** 2)
                conf_ldl_stack = torch.stack(conf_ldl)
                pred_distribution = pred_distribution.squeeze(1)
                conf_ldl_stack = conf_ldl_stack.to(pred_distribution.dtype)

                pred_distribution = torch.clamp(pred_distribution, min=1e-9)
                conf_ldl_stack = torch.clamp(conf_ldl_stack, min=1e-9)

                kl_divergence = torch.sum(conf_ldl_stack * torch.log(conf_ldl_stack / pred_distribution))

                pred_semi_distribution = torch.clamp(pred_semi_distribution, min=1e-9)
                conf_ldl_semi = torch.clamp(conf_ldl_semi, min=1e-9)

                kl_divergence_semi = torch.sum(conf_ldl_semi * torch.log(conf_ldl_semi / pred_semi_distribution))

                loss_3 = rank_scores.sum()

                sigma1 = torch.exp(self.model.log_sigma1)
                sigma2 = torch.exp(self.model.log_sigma2)

                task1_loss = (1 / (2 * sigma1 ** 2)) * ((kl_divergence + kl_divergence_semi * self.args.weightloss) + MSE_pos + MSE_semi * self.args.weightloss) + self.model.log_sigma1
                task2_loss = (1 / (2 * sigma2 ** 2)) * (loss_3 / 10) + self.model.log_sigma2

                # complete loss
                loss = task1_loss + task2_loss


        return loss