import numpy as np
from collections import defaultdict as ddict
from .BaseLitModel import BaseLitModel
from ..eval_task import *
from torch.autograd import Variable


class GMUCLitModel(BaseLitModel):
    """
    Processing of training, evaluation and testing for GMUC and GMUC+.
    """

    def __init__(self, model, train_sampler, args):
        super().__init__(model, args)
        self.sampler = train_sampler
        self.rel2candidates = train_sampler.rel2candidates
        self.symbol2id = train_sampler.symbol2id
        self.ent2id = train_sampler.ent2id
        self.e1rel_e2 = train_sampler.e1rel_e2
        self.rele1_e2 = train_sampler.rele1_e2

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser

    def training_step(self, batch, batch_idx):

        support          = batch["support"]
        query            = batch["query"]
        false            = batch["false"]
        support_meta     = batch["support_meta"]
        query_meta       = batch["query_meta"]
        false_meta       = batch["false_meta"]
        symbolid_ic      = batch["symbolid_ic"]

        if self.args.has_ont:
            query_scores, query_scores_var, false_scores, query_confidence = self.model(support, support_meta, query,
                                                                                        query_meta, false, false_meta,
                                                                                        self.args.if_ne)
            loss = self.loss(query_scores, query_scores_var, false_scores, query_confidence, symbolid_ic)
        else:
            (query_scores, query_scores_var, query_ae_loss, false_scores, false_scores_var, false_ae_loss,
             query_confidence) = self.model(support, support_meta, query, query_meta, false, false_meta)
            loss = self.loss(query_scores, query_scores_var, query_ae_loss, false_scores, query_confidence)

        self.log("Train|loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        results = self.val_test(batch)

        return results

    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        results = self.val_test(batch)

        return results

    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def val_test(self, batch):
        task = batch["query"]
        test_tasks = batch["triples"]

        support_triples = test_tasks[:self.args.few]  # pos and neg support

        support_pairs = [[self.symbol2id[triple[0]], self.symbol2id[triple[2]], float(triple[3])] for triple in
                         support_triples]
        support_left = [self.ent2id[triple[0]] for triple in support_triples]
        support_right = [self.ent2id[triple[2]] for triple in support_triples]
        support_meta = self.sampler.get_meta(support_left, support_right)

        if self.args.if_GPU:
            support = Variable(torch.Tensor(support_pairs)).cuda()
        else:
            support = Variable(torch.Tensor(support_pairs))

        if self.args.type_constrain:
            candidates = self.rel2candidates[task]
        else:
            candidates = list(self.ent2id.keys())
            candidates = candidates[:1000]

        all_conf, raw_conf_filter, filter_conf_filter, ndcg = [], [], [], []
        pos_mae, pos_mse = [], []
        neg_mae, neg_mse = [], []
        mr, mrr = [], []
        hits = ddict(list)
        r_mr, r_mrr = [], []
        r_hits = ddict(list)

        for e1, e2s in self.rele1_e2[task].items():
            query_pairs = []
            query_left = []
            query_right = []
            query_confidence = []  # len=len(true_e2)

            true_e2 = []
            true_s = []

            for _ in e2s:
                if [e1, task, _[0], _[1]] in support_triples:
                    continue
                true_e2.append(_[0])
                true_s.append(float(_[1]))

            num_e2 = len(true_e2)
            if num_e2 == 0:
                continue

            for i in range(num_e2):
                query_pairs.append([self.symbol2id[e1], self.symbol2id[true_e2[i]]])
                query_left.append(self.ent2id[e1])
                query_right.append(self.ent2id[true_e2[i]])
                query_confidence.append(true_s[i])  # pos queries

            for ent in candidates:
                # if (ent not in self.e1rel_e2[e1 + task]) and (ent not in true_e2):
                query_pairs.append([self.symbol2id[e1], self.symbol2id[ent]])
                query_left.append(self.ent2id[e1])
                query_right.append(self.ent2id[ent])
            query = Variable(torch.LongTensor(query_pairs)).cuda()
            query_meta = self.sampler.get_meta(query_left, query_right)

            # match support set and each query
            if self.args.has_ont:
                scores, scores_var = self.model.score_func(support, support_meta, query, query_meta)
            else:
                scores, scores_var, _ = self.model.score_func(support, support_meta, query, query_meta)

            scores.detach()
            scores = scores.data
            scores = scores.cpu().numpy()
            scores_var.detach()
            scores_var = scores_var.data
            scores_var = scores_var.cpu().numpy()

            f_scores = np.array([])
            f_scores = np.append(f_scores, [scores[i] for i in range(num_e2)])

            for idx, ent in enumerate(candidates):
                if (ent not in self.e1rel_e2[e1 + task]) and (ent not in true_e2):
                    f_scores = np.append(f_scores, scores[idx + num_e2])

            # filter ndcg
            # score_sort = list(np.argsort(f_scores))[::-1]
            # ranks = [score_sort.index(i) + 1 for i in range(num_e2)]  # 所有queries的排名，前num_e2个是pos
            #
            # confidence_arr = np.array(query_confidence)
            # rank_arr = np.array(ranks)
            # discounts = np.log2(rank_arr + 1)
            # discounted_gains = confidence_arr / discounts
            # dcg = np.sum(discounted_gains)
            #
            # confidence_sort = sorted(query_confidence, reverse=True)
            # confidence_sort = np.array(confidence_sort)
            # idcg = np.sum(confidence_sort / np.log2(np.arange(len(confidence_sort)) + 2))
            # ndcg_ = dcg / idcg
            # ndcg.append(ndcg_)

            all_conf.extend(query_confidence)

            # ============================= raw tail entity prediction =============================
            scores_filter = np.array([])
            num_e2_filter = 0

            for i in range(scores.size):
                if i < num_e2:
                    if query_confidence[i] > self.args.confidence_filter:
                        scores_filter = np.append(scores_filter, scores[i])
                        raw_conf_filter.append(query_confidence[i])
                        num_e2_filter += 1
                else:
                    scores_filter = np.append(scores_filter, scores[i])

            score_sort = list(np.argsort(scores_filter))[::-1]
            ranks = [score_sort.index(i) + 1 for i in range(num_e2_filter)]

            ranks_sort = sorted(ranks)
            for i in range(len(ranks)):
                rank = ranks_sort[i] - i
                if rank <= 0:
                    rank = 1
                for k in self.args.calc_hits:
                    r_hits['r_hits@{}'.format(k)].append(1.0 if rank <= k else 0.0)
                r_mrr.append(1.0 / rank)
                r_mr.append(rank)

            # ============================= filter tail entity prediction =============================
            scores_filter = np.array([])
            num_e2_filter = 0

            for i in range(f_scores.size):
                if i < num_e2:
                    if query_confidence[i] > self.args.confidence_filter:
                        scores_filter = np.append(scores_filter, f_scores[i])
                        filter_conf_filter.append(query_confidence[i])
                        num_e2_filter += 1
                else:
                    scores_filter = np.append(scores_filter, f_scores[i])

            score_sort = list(np.argsort(scores_filter))[::-1]
            ranks = [score_sort.index(i) + 1 for i in range(num_e2_filter)]

            ranks_sort = sorted(ranks)
            for i in range(len(ranks)):
                rank = ranks_sort[i] - i
                if rank <= 0:
                    rank = 1
                for k in self.args.calc_hits:
                    hits['hits@{}'.format(k)].append(1.0 if rank <= k else 0.0)
                mrr.append(1.0 / rank)
                mr.append(rank)

            # ==================== confidence prediction ============================
            pos_true_score = np.array(query_confidence)
            pos_pred_score = np.array(scores_var[:num_e2])

            pos_mae.extend(abs(pos_true_score - pos_pred_score))
            pos_mse.extend((pos_true_score - pos_pred_score) ** 2)

            neg_pred_score = np.array(scores_var[num_e2:])
            neg_mae.extend(abs(neg_pred_score))
            neg_mse.extend((neg_pred_score) ** 2)


        all_conf_arr = np.array(raw_conf_filter)
        all_rank_arr = np.array(r_mr)
        r_wmr_ = np.sum(all_conf_arr * all_rank_arr) / np.sum(all_conf_arr)
        all_rr_arr = np.array(r_mrr)
        r_wmrr_ = np.sum(all_conf_arr * all_rr_arr) / np.sum(all_conf_arr)

        all_conf_arr = np.array(filter_conf_filter)
        all_rank_arr = np.array(mr)
        wmr_ = np.sum(all_conf_arr * all_rank_arr) / np.sum(all_conf_arr)
        all_rr_arr = np.array(mrr)
        wmrr_ = np.sum(all_conf_arr * all_rr_arr) / np.sum(all_conf_arr)

        results = dict()
        results["MAE"] = np.mean(pos_mae)
        results["MSE"] = np.mean(pos_mse)
        # raw result
        results["raw_mr"] = np.mean(r_mr) if r_mr else 0
        results["raw_mrr"] = np.mean(r_mrr) if r_mrr else 0
        results["raw_wmr"] = r_wmr_
        results["raw_wmrr"] = r_wmrr_
        for k in self.args.calc_hits:
            results['raw_hits@{}'.format(k)] = np.mean(r_hits['r_hits@{}'.format(k)]) if r_hits['r_hits@{}'.format(k)] else 0
        # filter result
        results["mr"] = np.mean(mr) if mr else 0
        results["mrr"] = np.mean(mrr) if mrr else 0
        # results["ndcg"] = np.mean(ndcg)
        results["wmr"] = wmr_
        results["wmrr"] = wmrr_
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = np.mean(hits['hits@{}'.format(k)]) if hits['hits@{}'.format(k)] else 0

        return results

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.
        """
        if self.args.has_ont:
            milestones = 10000
            gamma = 0.25
        else:
            milestones = int(self.args.max_epochs / 2)
            gamma = 0.1
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=gamma)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def get_results(self, results, mode):
        outputs = ddict(float)
        count = len(results)

        for metric in list(results[0].keys()):
            final_metric = "_".join([mode, metric])
            outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count, decimals=3).item()

        return outputs
