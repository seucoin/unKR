import torch.nn as nn
import torch
from .model import Model


class UKGE_PSL(Model):
    """`Embedding Uncertain Knowledge Graphs`_ (UKGE), which represents the relationships as translations in the embedding space for uncertain knowledge graph.

    Attributes:
        args: Model configuration parameters.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
        w: the weight of mapping function.
        b: the bias of mapping function.
    .. _Embedding Uncertain Knowledge Graphs: https://ojs.aaai.org/index.php/AAAI/article/view/4210
    """

    def __init__(self, args):
        super(UKGE_PSL, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.w = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

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
        score = torch.sigmoid(self.w * score + self.b)

        return score

    def forward(self, triples, negs=None, mode='single'):
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
