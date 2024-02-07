import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

torch.set_printoptions(sci_mode=False)


class GMUCp(nn.Module):
    """`Incorporating Uncertainty of Entities and Relations into Few-Shot Uncertain Knowledge Graph Embedding`_ (GMUC+),
        which incorporates the inherent uncertainty of entities and relations into uncertain knowledge graph embedding.

    Attributes:
        args: Model configuration parameters.
        num_symbols: The sum of the number of entities and relationships.
        embed: Pretrained embedding.

    .. _Incorporating Uncertainty of Entities and Relations into Few-Shot Uncertain Knowledge Graph Embedding:
        https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2
    """

    def __init__(self, args, num_symbols, embed=None):
        super(GMUCp, self).__init__()
        self.args = args
        self.num_symbols = num_symbols
        self.pad_idx = num_symbols
        # mean&variance embeddings of symbols(ents+rels)
        self.symbol_emb_mean = nn.Embedding(num_symbols + 1, self.args.emb_dim,
                                            padding_idx=num_symbols)  # initialize as N(0, 1)
        self.symbol_emb_var = nn.Embedding(num_symbols + 1, self.args.emb_dim, padding_idx=num_symbols)

        # neighbor encoder attention
        self.neigh_att_W_mean = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.neigh_att_u_mean = nn.Linear(self.args.emb_dim, 1)
        self.neigh_att_W_var = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.neigh_att_u_var = nn.Linear(self.args.emb_dim, 1)

        # aggregate support set
        self.set_rnn_encoder = nn.LSTM(2 * self.args.emb_dim, 2 * self.args.emb_dim, 1, bidirectional=False)
        self.set_rnn_decoder = nn.LSTM(2 * self.args.emb_dim, 2 * self.args.emb_dim, 1, bidirectional=False)
        # aggregation attention
        self.set_att_W_mean = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.set_att_u_mean = nn.Linear(self.args.emb_dim, 1)
        self.set_att_W_var = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.set_att_u_var = nn.Linear(self.args.emb_dim, 1)

        # matchnet variance score
        self.w = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)

        # ic_loss function
        self.ic_loss_w = nn.Parameter(torch.randn(1), requires_grad=True)
        self.ic_loss_b = nn.Parameter(torch.randn(1), requires_grad=True)
        self.ic_loss_W = nn.Linear(self.args.emb_dim, 1)

        init.xavier_normal_(self.neigh_att_W_mean.weight)
        init.xavier_normal_(self.neigh_att_W_var.weight)
        init.xavier_normal_(self.neigh_att_u_mean.weight)
        init.xavier_normal_(self.neigh_att_u_var.weight)
        init.xavier_normal_(self.set_att_W_mean.weight)
        init.xavier_normal_(self.set_att_W_var.weight)
        init.xavier_normal_(self.set_att_u_mean.weight)
        init.xavier_normal_(self.set_att_u_var.weight)

        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = LayerNormalization(2 * self.args.emb_dim)
        dropout = self.args.dropout
        self.dropout = nn.Dropout(dropout)

        self.support_encoder_mean = SupportEncoder(2 * self.args.emb_dim, 2 * 2 * self.args.emb_dim, dropout)
        self.support_encoder_var = SupportEncoder(2 * self.args.emb_dim, 2 * 2 * self.args.emb_dim, dropout)
        self.matchnet_mean = MatchNet(2 * self.args.emb_dim, self.args.process_steps)
        self.matchnet_var = MatchNet(2 * self.args.emb_dim, self.args.process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        """Mean and variance encoding using node neighbors.

        Args:
            connections: Neighbor information.
            num_neighbors: The number of neighbors.

        Returns:
            out_mean: Mean embeddings.
            out_var: Variance embeddings.
        """
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1).long()
        entities = connections[:, :, 1].squeeze(-1).long()
        confidences = connections[:, :, 2]

        batch_size, neighbor_size = confidences.shape
        confidences = confidences.reshape((batch_size, neighbor_size, 1)).repeat((1, 1, 2 * self.args.emb_dim))

        # mean
        rel_embeds_mean = self.dropout(self.symbol_emb_mean(relations))
        ent_embeds_mean = self.dropout(self.symbol_emb_mean(entities))
        concat_embeds_mean = torch.cat((rel_embeds_mean, ent_embeds_mean), dim=-1)
        # variance
        rel_embeds_var = self.dropout(self.symbol_emb_var(relations))
        ent_embeds_var = self.dropout(self.symbol_emb_var(entities))
        concat_embeds_var = torch.cat((rel_embeds_var, ent_embeds_var), dim=-1)

        # attention mechanism  to enhance reps
        # mean
        out_mean = self.neigh_att_W_mean(concat_embeds_mean).tanh()
        att_w_mean = self.neigh_att_u_mean(out_mean)
        att_w_mean = self.softmax(att_w_mean).view(concat_embeds_mean.size()[0], 1, 30)
        out_mean = torch.bmm(att_w_mean, ent_embeds_mean).view(concat_embeds_mean.size()[0], self.args.emb_dim)
        # variance
        out_var = self.neigh_att_W_var(concat_embeds_var).tanh()
        att_w_var = self.neigh_att_u_var(out_var)
        att_w_var = self.softmax(att_w_var).view(concat_embeds_var.size()[0], 1, 30)
        out_var = torch.bmm(att_w_var, ent_embeds_mean).view(concat_embeds_var.size()[0], self.args.emb_dim)

        return out_mean.tanh(), out_var.tanh()

    def score_func(self, support, support_meta, query, query_meta, if_ne=True):
        """Calculating the score of query triples with support set.

        Args:
            support: Support set.
            support_meta: Neighbor information of the head and tail entities in support set.
            query: Query set.
            query_meta: Neighbor information of the head and tail entities in query set.
            if_ne: Whether to use neighbor node information for encoding.

        Returns:
            rank_score: The score for link prediction.
            conf_score: The confidence prediction value.
        """
        if not if_ne:
            support_left, support_right = support[:, 0].squeeze(-1).long(), support[:, 1].squeeze(-1).long()
            support_left_mean, support_right_mean = self.symbol_emb_mean(support_left), self.symbol_emb_mean(
                support_right)
            support_left_var, support_right_var = self.symbol_emb_var(support_left), self.symbol_emb_var(support_right)

            query_left, query_right = query[:, 0].squeeze(-1).long(), query[:, 1].squeeze(-1).long()
            query_left_mean, query_right_mean = self.symbol_emb_mean(query_left), self.symbol_emb_mean(query_right)
            query_left_var, query_right_var = self.symbol_emb_var(query_left), self.symbol_emb_var(query_right)

        else:
            query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
            support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta
            # ========================== neighbor encoder ===============================
            query_left_mean, query_left_var = self.neighbor_encoder(query_left_connections, query_left_degrees)
            query_right_mean, query_right_var = self.neighbor_encoder(query_right_connections, query_right_degrees)
            support_left_mean, support_left_var = self.neighbor_encoder(support_left_connections, support_left_degrees)
            support_right_mean, support_right_var = self.neighbor_encoder(support_right_connections,
                                                                          support_right_degrees)

        # query encode
        query_neighbor_mean = torch.cat((query_left_mean, query_right_mean), dim=-1)
        query_g_mean = self.support_encoder_mean(query_neighbor_mean)
        query_neighbor_var = torch.cat((query_left_var, query_right_var), dim=-1)
        query_g_var = self.support_encoder_var(query_neighbor_var)

        # support set
        support_neighbor_mean = torch.cat((support_left_mean, support_right_mean), dim=-1)
        support_g_mean = self.support_encoder_mean(support_neighbor_mean)
        support_neighbor_var = torch.cat((support_left_var, support_right_var), dim=-1)
        support_g_var = self.support_encoder_var(support_neighbor_var)

        # =============================== matching networks==============================
        query_mean = query_g_mean
        query_var = query_g_var
        # support set mean pooling
        support_mean = torch.mean(support_g_mean, dim=0, keepdim=True)
        support_var = torch.mean(support_g_var, dim=0, keepdim=True)

        # match
        matching_scores_mean = self.matchnet_mean(support_mean, query_mean)
        matching_scores_var = self.matchnet_var(support_var, query_var)

        rank_score = matching_scores_mean + self.args.conf_score_weight * matching_scores_var
        conf_score = torch.sigmoid(self.w * matching_scores_var + self.b)
        return rank_score, conf_score

    def forward(self, support, support_meta, query, query_meta, false, false_meta, if_ne=True):
        """The functions used in the training phase.

        Args:
            support: Support set.
            support_meta: Neighbor information of the head and tail entities in support set.
            query: Query set.
            query_meta: Neighbor information of the head and tail entities in query set.
            false: False set.
            false_meta: Neighbor information of the head and tail entities in false set.
            if_ne: Whether to use neighbor node information for encoding.

        Returns:
            query_scores: The match score between the query set and the support set.
            query_scores_var: Support set confidence prediction values.
            false_scores: The match score between the false set and the support set.
            query_confidence: The confidence for query set.
        """
        query_scores, query_scores_var = self.score_func(support, support_meta, query, query_meta, if_ne)
        false_scores, false_scores_var = self.score_func(support, support_meta, false, false_meta, if_ne)
        query_confidence = query[:, 2]

        return query_scores, query_scores_var, false_scores, query_confidence


class LayerNormalization(nn.Module):
    """
    Layer normalization module.
    """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class SupportEncoder(nn.Module):
    """
    Docstring for SupportEncoder
    """

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)


class MatchNet(nn.Module):
    """
    LSTM-based matching network.
    """

    def __init__(self, input_dim, process_steps=4):
        super(MatchNet, self).__init__()
        self.input_dim = input_dim
        self.process_steps = process_steps
        self.match_net = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support, query):
        assert support.size()[1] == query.size()[1]

        if self.process_steps == 0:
            return F.cosine_similarity(query, support, dim=1)
            # return torch.matmul(query, support.t()).squeeze()

        batch_size = query.size()[0]

        h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

        for step in range(self.process_steps):
            h_r_, c = self.match_net(query, (h_r, c))
            h = query + h_r_[:, :self.input_dim]
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            r = torch.matmul(attn, support)
            h_r = torch.cat((h, r), dim=1)

        matching_scores = torch.matmul(h, support.t()).squeeze()
        return matching_scores
