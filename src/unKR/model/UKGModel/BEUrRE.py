import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform
import numpy as np


class BEUrRE(nn.Module):
    """`Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning`_ (BEUrRE), which utilizes the geometry of boxes
    to calibrate probabilistic semantics and facilitating the incorporation of relational constraints.

    Attributes:
        args: Model configuration parameters.
        min_init_value: Initial value range for the minimum boundary of entity boxes. This range is used for uniformly initializing the minimum boundary of each entity box.
        delta_init_value: Initial value range for the size change (delta) of entity boxes. This range is used for uniformly initializing the size change of each entity box.
        min_embedding: Embedding for the minimum boundary of entities, initialized using min_init_value
        delta_embedding: Embedding for the size change (delta) of entity boxes, initialized using delta_init_value
        rel_trans_for_head: Relation-specific translation parameters for head entities
        rel_scale_for_head: Relation-specific scaling parameters for head entities
        rel_trans_for_tail: Relation-specific translation parameters for tail entities
        rel_scale_for_tail: Relation-specific scaling parameters for tail entities
        true_head, true_tail: Variables used for filtering negative samples to ensure that generated negative samples are not true triples.
        gumbel_beta: The beta parameter for the Gumbel distribution, controlling the softness of intersections in box embeddings.
        device: Specifies the device for model training, such as GPU.
        ratio: The ratio of negative samples to positive samples, used for generating negative samples during training.
        vocab_size: Total number of entities
        alpha: A small numeric value to prevent division by zero in calculations.
        clamp_min, clamp_max: Used to limit the range of gradients and embedding values to prevent numeric overflow or excessively large values.
        REL_VOCAB_SIZE: Total number of relations

    .. _Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning: https://aclanthology.org/2021.naacl-main.68
    """
    def __init__(self, args, min_init_value=None, delta_init_value=None):
        super(BEUrRE, self).__init__()

        if delta_init_value is None:
            delta_init_value = [-0.1, -0.001]
        if min_init_value is None:
            min_init_value = [1e-4, 0.01]
        self.euler_gamma = 0.57721566490153286060
        self.min_init_value = min_init_value
        self.delta_init_value = delta_init_value

        min_embedding = self.init_embedding(args.num_ent, args.emb_dim, min_init_value)
        delta_embedding = self.init_embedding(args.num_ent, args.emb_dim, delta_init_value)
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)

        rel_trans_for_head = torch.empty(args.num_rel, args.emb_dim)
        rel_scale_for_head = torch.empty(args.num_rel, args.emb_dim)
        torch.nn.init.normal_(rel_trans_for_head, mean=0, std=1e-4)  # 1e-4 before
        torch.nn.init.normal_(rel_scale_for_head, mean=1, std=0.2)  # 0.2 before

        rel_trans_for_tail = torch.empty(args.num_rel, args.emb_dim)
        rel_scale_for_tail = torch.empty(args.num_rel, args.emb_dim)
        torch.nn.init.normal_(rel_trans_for_tail, mean=0, std=1e-4)
        torch.nn.init.normal_(rel_scale_for_tail, mean=1, std=0.2)

        # make nn.Parameter
        self.rel_trans_for_head, self.rel_scale_for_head = nn.Parameter(rel_trans_for_head.to(args.gpu)), nn.Parameter(
            rel_scale_for_head.to(args.gpu))
        self.rel_trans_for_tail, self.rel_scale_for_tail = nn.Parameter(rel_trans_for_tail.to(args.gpu)), nn.Parameter(
            rel_scale_for_tail.to(args.gpu))

        self.true_head, self.true_tail = None, None  # for negative sample filtering
        self.gumbel_beta = args.GUMBEL_BETA
        self.device = args.gpu
        self.ratio = args.num_neg
        self.vocab_size = args.num_ent
        self.alpha = 1e-16
        self.clamp_min = 0.0
        self.clamp_max = 1e10
        self.REL_VOCAB_SIZE = args.num_rel
        self.args = args


    def forward(self, ids, train=True):
        """The functions used in the training phase

        Args:
            ids: The quadruples ids, as (h, r, t, c), shape:[batch_size, 4].
            train: mode

        Returns:
            pos_predictions: The confidence score of triples.
        """
        head_boxes = self.transform_head_boxes(ids)
        tail_boxes = self.transform_tail_boxes(ids)

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # condition on subject or object
        log_prob = log_intersection - self.log_volumes(tail_boxes)

        pos_predictions = torch.exp(log_prob)
        return pos_predictions

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            scores: The confidence score of triples.
        """
        triples = batch["positive_sample"].to(torch.long)
        triples = triples[:, :3]

        if mode == "tail_predict":
            scores = []

            # Repeating head-entities and relations to construct all possible triples
            head_entities = triples[:, 0].unsqueeze(1).repeat(1, self.vocab_size)
            relations = triples[:, 1].unsqueeze(1).repeat(1, self.vocab_size)
            tail_entities = torch.arange(self.vocab_size, dtype=torch.long).unsqueeze(0).repeat(len(triples), 1).to(self.args.gpu)

            bs_triples = torch.stack([head_entities, relations, tail_entities], dim=2)
            for i in bs_triples:
                score = self.forward(i)
                scores.append(score)

            scores = torch.stack(scores, dim=0)

            return scores

        elif mode == "head_predict":
            scores = []

            # Repeating tail-entities and relations to construct all possible triples
            head_entities = torch.arange(self.vocab_size, dtype=torch.long).unsqueeze(0).repeat(len(triples), 1).to(self.args.gpu)
            relations = triples[:, 1].unsqueeze(1).repeat(1, self.vocab_size)
            tail_entities = triples[:, 2].unsqueeze(1).repeat(1, self.vocab_size)

            bs_triples = torch.stack([head_entities, relations, tail_entities], dim=2)
            for i in bs_triples:
                score = self.forward(i)
                scores.append(score)

            scores = torch.stack(scores, dim=0)

            return scores

        else:
            return self.forward(triples)

    def transform_head_boxes(self, ids):
        """
        Transforms head entity boxes based on the relation-specific parameters.

        Args:
            ids: The quadruples ids, as (h, r, t, c), shape:[batch_size, 4].

        Returns:
            head_boxes: Transformed head entity boxes after applying relation-specific affine transformations.
        """
        head_boxes = self.get_entity_boxes(ids[:, 0])

        rel_ids = ids[:, 1]
        relu = nn.ReLU()

        translations = self.rel_trans_for_head[rel_ids]
        scales = relu(self.rel_scale_for_head[rel_ids])

        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def transform_tail_boxes(self, ids):
        """
        Transforms tail entity boxes based on the relation-specific parameters.

        Args:
            ids: The quadruples ids, as (h, r, t, c), shape:[batch_size, 4].

        Returns:
            tail_boxes: Transformed tail entity boxes after applying relation-specific affine transformations.
        """
        tail_boxes = self.get_entity_boxes(ids[:, 2])

        rel_ids = ids[:, 1]
        relu = nn.ReLU()

        translations = self.rel_trans_for_tail[rel_ids]
        scales = relu(self.rel_scale_for_tail[rel_ids])

        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes


    def intersection(self, boxes1, boxes2):
        """
        Computes the intersection of two sets of boxes using the Gumbel softmax trick.

        Args:
            boxes1: The first set of entity boxes.
            boxes2: The second set of entity boxes.

        Returns:
            intersection_box: The intersection of the two sets of boxes.
        """
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(boxes1.min_embed, boxes2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(boxes1.max_embed, boxes2.max_embed)
        )

        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    def log_volumes(self, boxes, temp=1., gumbel_beta=1., scale=1.):
        """
        Calculates the logarithm of the volumes of boxes.

        Args:
            boxes: A set of entity boxes.
            temp: The temperature parameter for the softplus function.
            gumbel_beta: The beta parameter for the Gumbel distribution.
            scale: Scaling factor for the volumes.

        Returns:
            log_vol: The logarithm of the volumes of the given boxes.
        """
        eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        log_vol = torch.sum(
            torch.log(
                F.softplus(boxes.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)

        return log_vol

    def get_entity_boxes(self, entities):
        """
        Retrieves entity boxes based on entity indices.

        Args:
            entities: A tensor containing indices of entities.

        Returns:
            boxes: A Box object representing the boxes of the specified entities.
        """
        min_rep = self.min_embedding[entities]  # batchsize * embedding_size
        delta_rep = self.delta_embedding[entities]
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def init_embedding(self, vocab_size, embed_dim, init_value):
        """
        Initializes embeddings uniformly within a specified range.

        Args:
            vocab_size: The number of unique entities.
            embed_dim: The dimension of the embeddings.
            init_value: A tuple indicating the range for uniform initialization.

        Returns:
            box_embed: Initialized embeddings.
        """
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed

    def random_negative_sampling(self, positive_samples, pos_probs, neg_per_pos=None):
        """
        Generates negative samples by corrupting either the head or the tail of positive samples.

        Args:
            positive_samples: Tensor of positive samples.
            pos_probs: confidence score of positive samples.
            neg_per_pos: Number of negative samples to generate per positive sample.

        Returns:
            negative_samples: Tensor of generated negative samples.
            neg_probs: confidence score negative samples (0).
        """
        if neg_per_pos is None:
            neg_per_pos = self.args.num_neg
        negative_samples1 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)
        negative_samples2 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)

        corrupted_heads = [self.get_negative_samples_for_one_positive(pos, neg_per_pos, mode='corrupt_head') for pos in positive_samples]
        corrupted_tails = [self.get_negative_samples_for_one_positive(pos, neg_per_pos, mode='corrupt_tail') for pos in positive_samples]

        negative_samples1[:, 0] = torch.cat(corrupted_heads)
        negative_samples2[:, 2] = torch.cat(corrupted_tails)
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(self.args.gpu)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_probs.dtype).to(self.args.gpu)

        return negative_samples, neg_probs

    def random_negative_sampling0(self, positive_samples, pos_probs, neg_per_pos=None):
        """
        Generates negative samples randomly without checking against true triples.

        Args:
            positive_samples: Tensor of positive samples.
            pos_probs: confidence score of positive samples.
            neg_per_pos: Number of negative samples to generate per positive sample.

        Returns:
            negative_samples: Tensor of generated negative samples.
            neg_probs: confidence score negative samples (0).
        """
        if neg_per_pos is None:
            neg_per_pos = self.args.num_neg
        negative_samples1 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)
        negative_samples2 = torch.repeat_interleave(positive_samples, neg_per_pos, dim=0)

        # corrupt tails
        corrupted_heads = torch.randint(self.vocab_size, (negative_samples1.shape[0],)).to(self.args.gpu)
        corrupted_tails = torch.randint(self.vocab_size, (negative_samples1.shape[0],)).to(self.args.gpu)

        #filter
        bad_heads_idxs = (corrupted_heads == negative_samples1[:,0])
        bad_tails_idxs = (corrupted_tails == negative_samples2[:,2])
        corrupted_heads[bad_heads_idxs] = torch.randint(self.vocab_size, (torch.sum(bad_heads_idxs),)).to(self.device)
        corrupted_tails[bad_tails_idxs] = torch.randint(self.vocab_size, (torch.sum(bad_tails_idxs),)).to(self.device)

        negative_samples1[:, 0] = corrupted_heads
        negative_samples2[:, 2] = corrupted_tails
        negative_samples = torch.cat((negative_samples1, negative_samples2), 0).to(self.args.gpu)
        neg_probs = torch.zeros(negative_samples.shape[0], dtype=pos_probs.dtype).to(self.args.gpu)

        return negative_samples, neg_probs

    def get_negative_samples_for_one_positive(self, positive_sample, neg_per_pos, mode):
        """
        Generates negative samples for one positive sample by corrupting head or tail.

        Args:
            positive_sample: A single positive sample.
            neg_per_pos: Number of negative samples to generate.
            mode: 'corrupt_head' or 'corrupt_tail' indicating which part of the triple to corrupt.

        Returns:
            negative_sample: Generated negative samples for the given positive sample.
        """
        head, relation, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < neg_per_pos:
            negative_sample = np.random.randint(self.args.num_ent, size=neg_per_pos * 2)

            # filter true values
            if mode == 'corrupt_head' and (int(relation), int(tail)) in self.true_head:  # filter true heads
                # For test data, some (relation, tail) pairs may be unseen and not in self.true_head
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(int(relation), int(tail))],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
            elif mode == 'corrupt_tail' and (int(head), int(relation)) in self.true_tail:
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(int(head), int(relation))],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:neg_per_pos]

        negative_sample = torch.from_numpy(negative_sample)
        return negative_sample


    def head_transformation(self, head_boxes, rel_ids):
        """
        Applies affine transformation to head boxes based on the relation.

        Args:
            head_boxes: Box objects for head entities.
            rel_ids: Relation indices corresponding to the head entities.

        Returns:
            head_boxes: Transformed Box objects for head entities.
        """
        relu = nn.ReLU()
        translations = self.rel_trans_for_head[rel_ids]
        scales = relu(self.rel_scale_for_head[rel_ids])
        # affine transformation
        head_boxes.min_embed += translations
        head_boxes.delta_embed *= scales
        head_boxes.max_embed = head_boxes.min_embed + head_boxes.delta_embed

        return head_boxes

    def tail_transformation(self, tail_boxes, rel_ids):
        """
        Applies affine transformation to tail boxes based on the relation.

        Args:
            tail_boxes: Box objects for tail entities.
            rel_ids: Relation indices corresponding to the tail entities.

        Returns:
            tail_boxes: Transformed Box objects for tail entities.
        """
        relu = nn.ReLU()
        translations = self.rel_trans_for_tail[rel_ids]
        scales = relu(self.rel_scale_for_tail[rel_ids])
        # affine transformation
        tail_boxes.min_embed += translations
        tail_boxes.delta_embed *= scales
        tail_boxes.max_embed = tail_boxes.min_embed + tail_boxes.delta_embed

        return tail_boxes

    def get_entity_boxes_detached(self, entities):
        """
        For logic constraint. We only want to optimize relation parameters, so detach entity parameters

        Args:
            entities: A tensor containing indices of entities.

        Returns:
            boxes: A Box object representing the boxes of the specified entities without attached gradients.
        """
        min_rep = self.min_embedding[entities].detach()
        delta_rep = self.delta_embedding[entities].detach()
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = Box(min_rep, max_rep)
        return boxes

    def transitive_rule_loss(self, ids):
        """
        Computes the loss for enforcing the transitive rule in the model.

        Args:
            ids: Tensor of triple ids.

        Returns:
            vol_loss: Volume loss enforcing the transitivity rule.
        """
        subsets = [ids[(ids[:,1] == r).nonzero().squeeze(1),:] for r in self.args.RULE_CONFIGS['transitive']['relations']]
        sub_ids = torch.cat(subsets, dim=0)

        # only optimize relation parameters
        head_boxes = self.get_entity_boxes_detached(sub_ids[:, 0])
        tail_boxes = self.get_entity_boxes_detached(sub_ids[:, 2])
        head_boxes = self.head_transformation(head_boxes, sub_ids[:,1])
        tail_boxes = self.tail_transformation(tail_boxes, sub_ids[:,1])

        intersection_boxes = self.intersection(head_boxes, tail_boxes)

        log_intersection = self.log_volumes(intersection_boxes)

        # P(f_r(epsilon_box)|g_r(epsilon_box)) should be 1
        vol_loss = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(tail_boxes)))

        return vol_loss

    def composition_rule_loss(self, ids):
        """
        Computes the loss for enforcing the composition rule in the model.

        Args:
            ids: Tensor of triple ids.

        Returns:
            vol_loss: Volume loss enforcing the composition rule.
        """
        def rels(size, rid):
            # fill a tensor with relation id
            return torch.full((size,), rid, dtype=torch.long)

        def biconditioning(boxes1, boxes2):
            intersection_boxes = self.intersection(boxes1, boxes2)
            log_intersection = self.log_volumes(intersection_boxes)
            # || 1-P(Box2|Box1) ||
            condition_on_box1 = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(boxes1)))
            # || 1-P(Box1|Box2) ||
            condition_on_box2 = torch.norm(1 - torch.exp(log_intersection - self.log_volumes(boxes2)))
            loss = condition_on_box1 + condition_on_box2

            return loss

        vol_loss = 0
        for rule_combn in self.args.RULE_CONFIGS['composite']['relations']:
            r1, r2, r3 = rule_combn
            r1_triples = ids[(ids[:, 1] == r1).nonzero().squeeze(1), :]
            r2_triples = ids[(ids[:, 1] == r2).nonzero().squeeze(1), :]

            # use heads and tails from r1, r2 as reasonable entity samples to help optimize relation parameters
            if len(r1_triples) > 0 and len(r2_triples) > 0:
                entities = torch.cartesian_prod(r1_triples[:,0], r2_triples[:,2])
                head_ids, tail_ids = entities[:,0], entities[:,1]
                size = len(entities)

                # only optimize relation parameters
                head_boxes_r1r2 = self.get_entity_boxes_detached(head_ids)
                tail_boxes_r1r2 = self.get_entity_boxes_detached(tail_ids)
                r1r2_head = self.head_transformation(head_boxes_r1r2, rels(size, r1))
                r1r2_head = self.head_transformation(r1r2_head, rels(size, r2))
                r1r2_tail = self.tail_transformation(tail_boxes_r1r2, rels(size, r1))
                r1r2_tail = self.tail_transformation(r1r2_tail, rels(size, r2))

                # head_boxes_r1r2 have been modified in transformation
                # so make separate box objects with the same parameters
                head_boxes_r3 = self.get_entity_boxes_detached(head_ids)
                tail_boxes_r3 = self.get_entity_boxes_detached(tail_ids)
                r3_head = self.head_transformation(head_boxes_r3, rels(size, r3))
                r3_tail = self.tail_transformation(tail_boxes_r3, rels(size, r3))

                head_transform_loss = biconditioning(r1r2_head, r3_head)
                tail_transform_loss = biconditioning(r1r2_tail, r3_tail)
                vol_loss += head_transform_loss
                vol_loss += tail_transform_loss
        return vol_loss


class Box:
    """
    A class representing an n-dimensional axis-aligned hyperrectangle (box) in the embedding space.

    Attributes:
        min_embed: The minimum boundary vector of the box in the embedding space.
        max_embed: The maximum boundary vector of the box in the embedding space.
        delta_embed: The difference between the maximum and minimum boundaries, representing the size of the box in each dimension.
    """
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed
