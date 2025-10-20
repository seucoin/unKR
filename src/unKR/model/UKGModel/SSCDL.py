import torch.nn as nn
import torch
from .model import Model
import numpy as np
import scipy.stats as stats


# min_old, max_old = 0.41562477969017253, 0.604478177772596  # sigma06

min_new, max_new = 0.1, 1.0

class MLP(nn.Module):
    # num_in = h || r || t
    def __init__(self, num_in, num_hid1, num_hid2, num_out = 101) -> None:
        super().__init__()
        self.mlp_net = nn.Sequential(
            nn.Linear(num_in, num_hid1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hid1, num_hid2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_hid2, num_out)
        )
        self.in_dim = num_in
        self.out_dim = num_out
        self.hid_dim1 = num_hid1
        self.hid_dim2 = num_hid2

    def forward(self, cat_embedding):
        pred_distri = self.mlp_net(cat_embedding)
        # print(pred_distri)
        pred_distri = torch.softmax(pred_distri, dim=2)
        # print(pred_distri)

        return pred_distri

class MLP_rank(nn.Module):
    # num_in = h || r || t
    def __init__(self, num_in, num_hid1, num_hid2, num_out = 1) -> None:
        super().__init__()
        self.mlp_net = nn.Sequential(
            nn.Linear(num_in, num_hid1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hid1, num_hid2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_hid2, 1)
        )
        self.in_dim = num_in
        self.out_dim = num_out
        self.hid_dim1 = num_hid1
        self.hid_dim2 = num_hid2

    def forward(self, cat_embedding):
        pred_distri = self.mlp_net(cat_embedding)
        pred_distri = torch.sigmoid(pred_distri)

        return pred_distri




class ssCDL(Model):

    def __init__(self, args):
        super(ssCDL, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.mlp_conf = MLP(self.args.emb_dim * 3, 1024, 512) # mlp to compute confidence distribution
        self.mlp_rank = MLP_rank(self.args.emb_dim * 3, 1024, 512) # mlp to compute rank score
        self.mlp_conf_meta = MLP(self.args.emb_dim * 3, 1024, 512)
        self.mlp_tmp = MLP(self.args.emb_dim * 3, 1024, 512)
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # log(sigma) on CP
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # log(sigma) on LP
        self.init_emb()

    def init_emb(self):
        """
            Initialize the entity and relation embeddings in the form of a uniform distribution.
        """

        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`h^{\top} \operatorname{diag}(r) t`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        if mode == 'head_predict' and head_emb.shape[0] == 1:
            # print('head_predict')
            head_emb = head_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]
            relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb = tail_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb_repeat = tail_emb.repeat_interleave(head_emb.shape[0], dim=0)
            relation_emb_repeat = relation_emb.repeat_interleave(head_emb.shape[0], dim=0)

            head_emb_repeat = head_emb.repeat(tail_emb.shape[0], 1)
            head_emb_repeat = head_emb_repeat.view(-1, head_emb.shape[1])

            head_emb_repeat = head_emb_repeat.unsqueeze(1)
            tail_emb_repeat = tail_emb_repeat.unsqueeze(1)
            relation_emb_repeat = relation_emb_repeat.unsqueeze(1)

            # print(X.shape)
            tensor_list = [head_emb_repeat, relation_emb_repeat, tail_emb_repeat]
            X = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]

        elif mode == 'tail_predict' and tail_emb.shape[0] == 1:  # tail_predict
            # print('tail_predict')
            head_emb = head_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb = tail_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]

            head_emb_repeat = head_emb.repeat_interleave(tail_emb.shape[0], dim=0)
            head_emb_repeat = head_emb_repeat.unsqueeze(1)
            relation_emb_repeat = relation_emb.repeat_interleave(tail_emb.shape[0], dim=0)
            relation_emb_repeat = relation_emb_repeat.unsqueeze(1)
            tail_emb_repeat = tail_emb.repeat(head_emb.shape[0], 1)
            tail_emb_repeat = tail_emb_repeat.view(-1, tail_emb.shape[1])
            tail_emb_repeat = tail_emb_repeat.unsqueeze(1)

            # print(head_emb_repeat.size())
            # print(relation_emb_repeat.size())
            # print(tail_emb_repeat.size())
            tensor_list = [head_emb_repeat, relation_emb_repeat, tail_emb_repeat]
            X = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]

        else:
            tensor_list = [head_emb, relation_emb, tail_emb]
            X = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]
            # print(X.shape)

        X = X.to(self.args.gpu)
        y = self.mlp_conf(X)
        rank_score = self.mlp_rank(X)
        # print(y.size())
        weights = torch.linspace(0, 1, steps=101).to('cuda')
        expected_values = (y * weights).sum(dim=2).squeeze()
        expected_values = (expected_values - self.args.lower_bound) * (1.0 - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1

        # print(expected_values.size())
        # expected_values = (pred_distribute * weights).sum(dim=2)
        # print(y.shape)
        score = expected_values.to(self.args.gpu)
        batch_size = relation_emb.shape[0]
        # print('3', score.size())
        score = score.reshape(batch_size, -1)
        rank_score = rank_score.reshape(batch_size, -1)
        # print(score.size())
        # print('1',rank_score.size())
        # print('2',score.size())

        return score, rank_score

    def score_func_argmax(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`h^{\top} \operatorname{diag}(r) t`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        if mode == 'head_predict' and head_emb.shape[0] == 1:
            # print('head_predict')
            head_emb = head_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]
            relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb = tail_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb_repeat = tail_emb.repeat_interleave(head_emb.shape[0], dim=0)
            relation_emb_repeat = relation_emb.repeat_interleave(head_emb.shape[0], dim=0)

            head_emb_repeat = head_emb.repeat(tail_emb.shape[0], 1)
            head_emb_repeat = head_emb_repeat.view(-1, head_emb.shape[1])

            head_emb_repeat = head_emb_repeat.unsqueeze(1)
            tail_emb_repeat = tail_emb_repeat.unsqueeze(1)
            relation_emb_repeat = relation_emb_repeat.unsqueeze(1)

            # print(X.shape)
            tensor_list = [head_emb_repeat, relation_emb_repeat, tail_emb_repeat]
            X = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]

        elif mode == 'tail_predict' and tail_emb.shape[0] == 1:  # tail_predict
            # print('tail_predict')
            head_emb = head_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb = tail_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]

            head_emb_repeat = head_emb.repeat_interleave(tail_emb.shape[0], dim=0)
            head_emb_repeat = head_emb_repeat.unsqueeze(1)
            relation_emb_repeat = relation_emb.repeat_interleave(tail_emb.shape[0], dim=0)
            relation_emb_repeat = relation_emb_repeat.unsqueeze(1)
            tail_emb_repeat = tail_emb.repeat(head_emb.shape[0], 1)
            tail_emb_repeat = tail_emb_repeat.view(-1, tail_emb.shape[1])
            tail_emb_repeat = tail_emb_repeat.unsqueeze(1)

            # print(head_emb_repeat.size())
            # print(relation_emb_repeat.size())
            # print(tail_emb_repeat.size())
            tensor_list = [head_emb_repeat, relation_emb_repeat, tail_emb_repeat]
            X = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]

        else:
            tensor_list = [head_emb, relation_emb, tail_emb]
            X = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]
            # print(X.shape)

        X = X.to(self.args.gpu)
        y = self.mlp_conf(X)
        weights = torch.linspace(0, 1, steps=101).to('cuda')
        # print(y.size())
        expected_values = (y * weights).sum(dim=2).squeeze()
        max_indices = y.squeeze(1).argmax(dim=1)

        score = weights[max_indices]
        batch_size = relation_emb.shape[0]
        score = score.reshape(batch_size, -1)


        return score

    def forward(self, triples, negs=None, mode='single', stage='main', fast_weights=None):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t, c), shape:[batch_size, 4].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """

        triples = triples[:, :3].to(torch.int)
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        # print(head_emb.shape)
        # print(tail_emb.shape)
        if mode == 'single':
            tensor_list = [head_emb, relation_emb, tail_emb]
            emb_cat = torch.cat(tensor_list, dim=2)
            # print(emb_cat.size())
            if stage == 'main':
                pred_distribution = self.mlp_conf(emb_cat)
                rank_score = self.mlp_rank(emb_cat)
            elif stage == 'tmp':
                # pred_distribution = self.mlp_tmp(emb_cat)
                # rank_score = self.mlp_rank(emb_cat)
                pred_distribution = self.forward_with_fast_weights(emb_cat, fast_weights)
                rank_score = self.mlp_rank(emb_cat)
            else:
                pred_distribution = self.mlp_conf_meta(emb_cat)
                rank_score = self.mlp_rank(emb_cat)
        if mode == 'head-batch':
            # head_emb = head_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]
            # relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            # tail_emb = tail_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            tail_emb_repeat = tail_emb.repeat_interleave(self.args.num_neg, dim=1)
            relation_emb_repeat = relation_emb.repeat_interleave(self.args.num_neg, dim=1)
            head_emb_repeat = head_emb
            # head_emb_repeat = head_emb_repeat.view(-1, head_emb.shape[1])

            head_emb_repeat = head_emb_repeat
            tail_emb_repeat = tail_emb_repeat
            relation_emb_repeat = relation_emb_repeat

            # print(X.shape)
            tensor_list = [head_emb_repeat, relation_emb_repeat, tail_emb_repeat]
            # print(head_emb_repeat.size())
            # print(relation_emb_repeat.size())
            # print(tail_emb_repeat.size())
            emb_cat = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]
            if stage == 'main':
                pred_distribution = self.mlp_conf(emb_cat)
                rank_score = self.mlp_rank(emb_cat)
            elif stage == 'tmp':
                # pred_distribution = self.mlp_tmp(emb_cat)
                # rank_score = self.mlp_rank(emb_cat)
                pred_distribution = self.forward_with_fast_weights(emb_cat, fast_weights)
                rank_score = self.mlp_rank(emb_cat)
            else:
                pred_distribution = self.mlp_conf_meta(emb_cat)
                rank_score = self.mlp_rank(emb_cat)
        if mode == 'tail-batch':
            # print('tail_predict')
            # head_emb = head_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            # relation_emb = relation_emb.squeeze(1)  # [bs, 1, dim]-->[bs, dim]
            # tail_emb = tail_emb.squeeze(0)  # [1, num_ent, dim]-->[num_ent, dim]

            head_emb_repeat = head_emb.repeat_interleave(self.args.num_neg, dim=1)
            # head_emb_repeat = head_emb_repeat.unsqueeze(1)
            relation_emb_repeat = relation_emb.repeat_interleave(self.args.num_neg, dim=1)
            # relation_emb_repeat = relation_emb_repeat.unsqueeze(1)
            tail_emb_repeat = tail_emb

            # print(head_emb_repeat.size())
            # print(relation_emb_repeat.size())
            # print(tail_emb_repeat.size())
            tensor_list = [head_emb_repeat, relation_emb_repeat, tail_emb_repeat]
            emb_cat = torch.cat(tensor_list, dim=2)  # [bs, 1, dim]
            if stage == 'main':
                pred_distribution = self.mlp_conf(emb_cat)
                rank_score = self.mlp_rank(emb_cat)
            elif stage == 'tmp':
                # pred_distribution = self.mlp_tmp(emb_cat)
                # rank_score = self.mlp_rank(emb_cat)
                pred_distribution = self.forward_with_fast_weights(emb_cat, fast_weights)
                rank_score = self.mlp_rank(emb_cat)
            else:
                pred_distribution = self.mlp_conf_meta(emb_cat)
                rank_score = self.mlp_rank(emb_cat)

        # print()
        return pred_distribution, rank_score

    def forward_with_fast_weights(self, cat_embedding, fast_weights):
        x = torch.matmul(cat_embedding, fast_weights["mlp_net.0.weight"].T) + fast_weights["mlp_net.0.bias"]
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)

        x = torch.matmul(x, fast_weights["mlp_net.3.weight"].T) + fast_weights["mlp_net.3.bias"]
        x = nn.ReLU()(x)
        x = nn.Dropout(0.3)(x)

        x = torch.matmul(x, fast_weights["mlp_net.6.weight"].T) + fast_weights["mlp_net.6.bias"]
        pred_distri = torch.softmax(x, dim=2)
        return pred_distri

    def forward_with_fast_weights_rank(self, cat_embedding, fast_weights):
        x = torch.matmul(cat_embedding, fast_weights["mlp_net.0.weight"].T) + fast_weights["mlp_net.0.bias"]
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)

        # Layer 2
        x = torch.matmul(x, fast_weights["mlp_net.3.weight"].T) + fast_weights["mlp_net.3.bias"]
        x = nn.ReLU()(x)
        x = nn.Dropout(0.3)(x)

        # Output Layer
        x = torch.matmul(x, fast_weights["mlp_net.6.weight"].T) + fast_weights["mlp_net.6.bias"]
        pred_rank = torch.sigmoid(x)
        # print('rank',pred_rank)
        return pred_rank

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """

        triples = batch["positive_sample"]
        triples = triples[:, :3].to(torch.int)
        triples = triples.to(self.args.gpu)

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score, score_rank = self.score_func(head_emb, relation_emb, tail_emb, mode)
        # print(score)
        # print(score)
        return score_rank

    def get_score_argmax(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """

        triples = batch["positive_sample"]
        triples = triples[:, :3].to(torch.int)
        triples = triples.to(self.args.gpu)

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        # print(score)
        # print(score)
        # exit(0)

        return score



