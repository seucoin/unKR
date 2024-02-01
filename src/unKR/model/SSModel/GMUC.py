import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class Matcher(nn.Module):
    """
    Gaussian Matching Networks.
    """

    def __init__(self, args, num_symbols, embed=None):
        super(Matcher, self).__init__()
        self.args = args
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.args.emb_dim, padding_idx=num_symbols)
        self.symbol_var_emb = nn.Embedding(num_symbols + 1, self.args.emb_dim, padding_idx=num_symbols)
        self.num_symbols = num_symbols
        self.layer_norm = LayerNormalization(2 * self.args.emb_dim)
        dropout = self.args.dropout

        if self.args.random_embed:
            self.args.use_pretrain = False
        else:
            self.args.use_pretrain = True

        self.dropout = nn.Dropout(dropout)

        # aggregation encoder and decoder
        self.set_rnn_encoder = nn.LSTM(2 * self.args.emb_dim, 2 * self.args.emb_dim, 1, bidirectional=False)
        self.set_rnn_decoder = nn.LSTM(2 * self.args.emb_dim, 2 * self.args.emb_dim, 1, bidirectional=False)

        self.set_FC_encoder = nn.Linear(3 * 2 * self.args.emb_dim, 2 * self.args.emb_dim)
        self.set_FC_decoder = nn.Linear(2 * self.args.emb_dim, 3 * 2 * self.args.emb_dim)

        # neighbor encoder attention
        self.neigh_att_W = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.neigh_att_u = nn.Linear(self.args.emb_dim, 1)
        self.neigh_var_att_W = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.neigh_var_att_u = nn.Linear(self.args.emb_dim, 1)
        # aggregation attention
        self.set_att_W = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.set_att_u = nn.Linear(self.args.emb_dim, 1)
        self.set_var_att_W = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.set_var_att_u = nn.Linear(self.args.emb_dim, 1)

        self.bn = nn.BatchNorm1d(2 * self.args.emb_dim)
        self.softmax = nn.Softmax(dim=1)

        self.FC_query_g = nn.Linear(2 * self.args.emb_dim, 2 * self.args.emb_dim)
        self.FC_support_g_encoder = nn.Linear(2 * self.args.emb_dim, 2 * self.args.emb_dim)

        init.xavier_normal_(self.neigh_att_W.weight)
        init.xavier_normal_(self.neigh_att_u.weight)
        init.xavier_normal_(self.set_att_W.weight)
        init.xavier_normal_(self.set_att_u.weight)
        init.xavier_normal_(self.FC_query_g.weight)
        init.xavier_normal_(self.FC_support_g_encoder.weight)

        # Here is symbol to embedding
        if self.args.use_pretrain:
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            self.symbol_var_emb.weight.data.copy_(torch.from_numpy(embed))

        d_model = self.args.emb_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, self.args.process_steps)

        self.w = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def neighbor_encoder(self, connections, num_neighbors):

        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1).long()  # neighbor relation
        entities = connections[:, :, 1].squeeze(-1).long()  # neighor entity
        confidences = connections[:, :, 2]  # get neighor confidence (batch, neighbor)

        (batch_size, neighbor_size) = confidences.shape
        confidences = confidences.reshape((batch_size, neighbor_size, 1)).repeat((1, 1, 2 * self.args.emb_dim))

        rel_embeds = self.dropout(self.symbol_emb(relations))  # (batch, neighbor_num, emb_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities))  # (batch, neighbor_num, emb_dim)
        concat_embeds = torch.cat((rel_embeds, ent_embeds),
                                  dim=-1)  # (batch, neighbor_num, 2*emb_dim)

        rel_var_embeds = self.dropout(self.symbol_var_emb(relations))  # (batch, neighbor_num, emb_dim)
        ent_var_embeds = self.dropout(self.symbol_var_emb(entities))  # (batch, neighbor_num, emb_dim)
        concat_var_embeds = torch.cat((rel_var_embeds, ent_var_embeds),
                                      dim=-1)  # (batch, neighbor_num, 2*emb_dim)

        # position embedding
        out = self.neigh_att_W(concat_embeds).tanh()
        att_w = self.neigh_att_u(out)
        att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, 30)
        out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.args.emb_dim)

        # variance embedding
        out_var = self.neigh_var_att_W(concat_var_embeds).tanh()
        att_var_w = self.neigh_var_att_u(out_var)
        att_var_w = self.softmax(att_var_w).view(concat_var_embeds.size()[0], 1, 30)
        out_var = torch.bmm(att_var_w, ent_embeds).view(concat_var_embeds.size()[0], self.args.emb_dim)

        return out.tanh(), out_var.tanh()

    def score_func(self, support, support_meta, query, query_meta):
        raise NotImplementedError

    def forward(self, support, support_meta, query, query_meta, false, false_meta):
        raise NotImplementedError


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
    The encoder for support set.
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


class QueryEncoder(nn.Module):
    """
    The encoder for query set, similar to
    https://proceedings.neurips.cc/paper_files/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf.
    """
    def __init__(self, input_dim, process_step=4):
        super(QueryEncoder, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support, query):
        assert support.size()[1] == query.size()[1]

        if self.process_step == 0:
            return query

        batch_size = query.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

        for step in range(self.process_step):
            h_r_, c = self.process(query, (h_r, c))
            h = query + h_r_[:, :self.input_dim]  # (batch_size, query_dim)
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            r = torch.matmul(attn, support)  # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        return h


class GMUC(Matcher):

    """`Gaussian Metric Learning for Few-Shot Uncertain Knowledge Graph Completion`_ (GMUC),
        which proposes a novel method to complete few-shot UKGs based on Gaussian metric learning.

    Attributes:
        args: Model configuration parameters.
        num_symbols: The sum of the number of entities and relationships.
        embed: Pretrained embedding.

    .. _Gaussian Metric Learning for Few-Shot Uncertain Knowledge Graph Completion:
        https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18
    """

    def __init__(self, arg, num_symbols, embed=None):
        super(GMUC, self).__init__(arg, num_symbols, embed)
        d_model = self.args.emb_dim * 2

        self.matchnet_mean = MatchNet(d_model, self.args.process_steps)
        self.matchnet_var = MatchNet(d_model, self.args.process_steps)
        if 'N3' in self.args.data_path:
            self.matchnet_mean = MatchNet(d_model, self.args.process_steps)
            self.matchnet_var = MatchNet(d_model, 1)
        self.gcn_w = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)
        self.gcn_w_var = nn.Linear(2 * self.args.emb_dim, self.args.emb_dim)

    def score_func(self, support, support_meta, query, query_meta):
        """Calculating the score of query triples with support set.

        Args:
            support: Support set.
            support_meta: Neighbor information of the head and tail entities in support set.
            query: Query set.
            query_meta: Neighbor information of the head and tail entities in query set.

        Returns:
            matching_scores: Matching scores for link prediction.
            matching_scores_var: The prediction of confidence.
            ae_loss: The loss of encoder input and decoder output.
        """
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        # ======================== 1. Guassin Neighbor Encoder ========================
        query_left, query_var_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right, query_var_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left, support_var_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right, support_var_right = self.neighbor_encoder(support_right_connections, support_right_degrees)
        support_confidence = support[:, 2]  # get confidence of support set (support_num)
        # ----------------- mean encoder ------------------
        query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1)  # tanh
        support = support_neighbor
        query = query_neighbor
        support_g = self.support_encoder(support)  # (3, 200)
        query_g = self.support_encoder(query)  # (batch, 2 * embedd_dim)

        # position encoder and decoder
        support_g_0 = support_g.view(3, 1, 2 * self.args.emb_dim)  # input of encoder
        support_g_encoder, support_g_state = self.set_rnn_encoder(support_g_0)

        # decoder
        support_g_decoder = support_g_encoder[-1].view(1, -1, 2 * self.args.emb_dim)
        support_g_decoder_state = support_g_state
        decoder_set = []
        for idx in range(3):
            support_g_decoder, support_g_decoder_state = self.set_rnn_decoder(support_g_decoder,
                                                                              support_g_decoder_state)
            decoder_set.append(support_g_decoder)
        decoder_set = torch.cat(decoder_set, dim=0)  # output of decoder
        ae_loss = nn.MSELoss()(support_g_0, decoder_set.detach())  # calculate loss of encoder input and decoder output
        # encoder
        support_g_encoder = support_g_encoder.view(3, 2 * self.args.emb_dim)
        support_g_encoder = support_g_0.view(3, 2 * self.args.emb_dim) + support_g_encoder  # output of encoder

        # add attention
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w)
        support_g_encoder = torch.matmul(support_g_encoder.transpose(0, 1), att_w)
        support_g_encoder = support_g_encoder.transpose(0, 1)
        support_g_encoder = support_g_encoder.view(1, 2 * self.args.emb_dim)

        support = support_g_encoder
        query = query_g

        # ----------------- variance encoder ------------------
        query_var_neighbor = torch.cat((query_var_left, query_var_right), dim=-1)  # tanh
        support_var_neighbor = torch.cat((support_var_left, support_var_right), dim=-1)  # tanh
        support_var = support_var_neighbor
        query_var = query_var_neighbor
        support_var = torch.mean(support_var_neighbor, dim=0, keepdim=True)

        #  ======================== 2. Guassin Matching Networks ========================
        # ----------------- mean match ------------------
        matching_scores = self.matchnet_mean(support, None, query, None)
        # ----------------- varaince match ------------------
        matching_scores_var = self.matchnet_var(support_var, None, query_var, None)
        matching_scores_var = torch.sigmoid(self.w + matching_scores_var + self.b)

        return matching_scores, matching_scores_var, ae_loss

    def forward(self, support, support_meta, query, query_meta, false, false_meta):
        """The functions used in the training phase.

        Args:
            support: Support set.
            support_meta: Neighbor information of the head and tail entities in support set.
            query: Query set.
            query_meta: Neighbor information of the head and tail entities in query set.
            false: False set.
            false_meta: Neighbor information of the head and tail entities in false set.

        Returns:
            query_scores: The match score between the query set and the support set.
            query_scores_var: Support set confidence prediction values.
            query_ae_loss: The loss of encoder input and decoder output in query set.
            false_scores: The match score between the false set and the support set.
            false_scores_var: False set confidence prediction values.
            false_ae_loss: The loss of encoder input and decoder output in false set.
            query_confidence: The confidence for query set.
        """
        query_scores, query_scores_var, query_ae_loss = self.score_func(support, support_meta, query, query_meta)
        false_scores, false_scores_var, false_ae_loss = self.score_func(support, support_meta, false, false_meta)
        query_confidence = query[:, 2]

        return query_scores, query_scores_var, query_ae_loss, false_scores, false_scores_var, false_ae_loss, query_confidence


class MatchNet(nn.Module):
    """
    RNN Match networks for support set and query set, similar to
    https://proceedings.neurips.cc/paper_files/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf.
    """
    def __init__(self, input_dim, process_step=4):
        super(MatchNet, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support_mean, support_var, query_mean, query_var):
        assert support_mean.size()[1] == query_mean.size()[1]

        if self.process == 1:
            return torch.matmul(query_mean, support_mean.t()).squeeze()

        batch_size = query_mean.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

        for step in range(self.process_step):
            h_r_, c = self.process(query_mean, (h_r, c))
            h = query_mean + h_r_[:, :self.input_dim]  # (batch_size, query_dim)

            attn = F.softmax(torch.matmul(h, support_mean.t()), dim=1)

            r = torch.matmul(attn, support_mean)  # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        matching_scores = torch.matmul(h, support_mean.t()).squeeze()
        return matching_scores
