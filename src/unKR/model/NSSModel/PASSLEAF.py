import torch.nn as nn
import torch
from .model import Model


class PASSLEAF(Model):
    """`A Pool-bAsed Semi-Supervised LEArning Framework for Uncertain Knowledge Graph Embedding`_ (PASSLEAF).

    Attributes:
        args: Model configuration parameters.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
        w: Weight when calculate confidence scores
        b: Bias when calculate confidence scores
    .. _Translating Embeddings for Modeling Multi-relational Data: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, args):
        super(PASSLEAF, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.w = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.init_emb()

    def init_emb(self):
        """
            Initialize the entity and relation embeddings in the form of a uniform distribution.
        """
        model = self.args.passleaf_score_function
        if model == 'DistMult':
            "`Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_ (DistMult)"
            self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
            self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)
        elif model == 'ComplEx':
            "`Complex Embeddings for Simple Link Prediction`_ (ComplEx)"
            self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
            self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)
        else: #RotatE
            "`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_ (RotatE)"
            """Initialize the entity and relation embeddings in the form of a uniform distribution."""
            # print(self.gama)
            """Initialize the entity and relation embeddings in the form of a uniform distribution."""
            self.epsilon = 2.0
            self.margin = nn.Parameter(
                torch.Tensor([self.args.margin]),
                requires_grad=False
            )
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
                requires_grad=False
            )
            self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
            self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
            nn.init.xavier_uniform_(self.ent_emb.weight.data)
            nn.init.xavier_uniform_(self.rel_emb.weight.data)


    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score of DistMult is :math:`h^{T} \operatorname{diag}(r) t`.

        The formula for calculating the score of ComplEx is :math:`Re(h^{T} \operatorname{diag}(r) \overline{t})`.

        The formula for calculating the score of RotatE is :math:`\gamma - \|h \circ r - t\|`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """

        model = self.args.passleaf_score_function # use different score function according to args.passleaf_score_function

        if model == 'DistMult':
            if mode == 'head-batch':
                score = head_emb * (relation_emb * tail_emb)
            else:
                score = (head_emb * relation_emb) * tail_emb

            score = score.sum(dim=-1)
            score = score.to(self.args.gpu)

            """ 1.Bounded rectifier """
            # shape = score.shape
            # tmp_max = torch.max(self.w * score + self.b, torch.zeros(shape, device=self.args.gpu))
            # score = torch.min(tmp_max, torch.ones(shape, device=self.args.gpu))

            """ 2.Logistic function"""
            score = torch.sigmoid(self.w * score + self.b) # use UKGE_logi in PASSLEAF


        if model == 'ComplEx':
            re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
            re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

            score = torch.sum(
                re_head * re_tail * re_relation
                + im_head * im_tail * re_relation
                + re_head * im_tail * im_relation
                - im_head * re_tail * im_relation,
                -1
            )
            score = torch.sigmoid(self.w * score + self.b)

        if model == 'RotatE':
            pi = 3.14159265358979323846
            re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

            # Make phases of relations uniformly distributed in [-pi, pi]

            phase_relation = relation_emb / (self.embedding_range.item() / pi)

            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)

            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                re_score = re_score - re_head
                im_score = im_score - im_head
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                re_score = re_score - re_tail
                im_score = im_score - im_tail

            score = torch.stack([re_score, im_score], dim=0)
            score = score.norm(dim=0)
            score = self.margin.item() - score.sum(dim=-1)
            score = torch.sigmoid(self.w * score + self.b)

        return score

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        triples = triples[:, :3].to(torch.int)
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

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

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score
