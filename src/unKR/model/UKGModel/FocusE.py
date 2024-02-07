import torch.nn as nn
import torch
from .model import Model


class FocusE(Model):
    """`Learning Embeddings from Knowledge Graphs With Numeric Edge Attributes`_ (FocusE), which injects
    numeric edge attributes into the scoring layer of a traditional knowledge graph embedding architecture.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].
        base_model: The based traditional knowledge graph embedding method.
        beta: the structural influence

    .. _Learning Embeddings from Knowledge Graphs With Numeric Edge Attributes: https://www.ijcai.org/proceedings/2021/395
    """

    def __init__(self, args):
        super(FocusE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.base_model = args.base_model
        self.beta = 1.0

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution.

        """
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False
        )

        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.uniform_(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples using different traditional knowledge graph embedding method.

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """

        if self.base_model == "DistMult":
            if mode == 'head-batch':
                score = head_emb * (relation_emb * tail_emb)
            else:
                score = (head_emb * relation_emb) * tail_emb

            score = score.sum(dim=-1)
        elif self.base_model == "ComplEX":
            re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
            re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

            score = torch.sum(re_head * re_tail * re_relation +
                              im_head * im_tail * re_relation +
                              re_head * im_tail * im_relation -
                              im_head * re_tail * im_relation, -1
                              )
        else:  # 暂时默认TransE
            score = (head_emb + relation_emb) - tail_emb
            score = self.margin.item() - torch.norm(score, p=1, dim=-1)

        return score

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The quadruple ids, as (h, r, t, c), shape:[batch_size, r].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        w = triples[:, 3]
        triples = triples[:, :3].to(torch.int)
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        softplus = nn.Softplus()
        score = softplus(score)

        if negs is None:
            alpha = self.beta + (torch.ones(w.shape).to(self.args.gpu) - w) * (1 - self.beta)
        else:
            alpha = self.beta + w * (1 - self.beta)

        alpha = alpha.unsqueeze(1)
        score = alpha * score

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples(But not the confidence).
        """
        triples = batch["positive_sample"]
        triples = triples[:, :3].to(torch.int)
        triples = triples.to(self.args.gpu)

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        softplus = nn.Softplus()
        score = softplus(score)

        return score


    def adjust_parameters(self, current_epoch):
        """To adjust the structural influence beta

        Args:
            current_epoch: Which epoch is currently being trained
        """
        self.beta = max(0.0, 1.0 - (current_epoch / (self.args.max_epochs-1)))
        print(self.beta)