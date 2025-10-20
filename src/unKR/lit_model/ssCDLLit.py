from .BaseLitModel import BaseLitModel
from ..eval_task import *
import pickle
from ..eval_task.nDCG import mean_ndcg
from collections import OrderedDict
import pickle
from functools import partial
import numpy as np
import scipy.stats as stats

from ..eval_task.nDCG import mean_ndcg

# min_old, max_old = 0.41562477969017253, 0.604478177772596  # sigma06
# min_new, max_new = 0.1, 1.0

def generate_distribution_tensor(conf, sigma, size):
    linspace = torch.linspace(0, 1, size, device=conf.device)
    gauss = torch.exp(-0.5 * ((linspace - conf) / sigma) ** 2)
    # print(conf)
    gauss = gauss / gauss.sum(dim=-1, keepdim=True)
    return gauss

def generate_distribution(conf, sigma, size):
    gauss = stats.norm(conf, sigma)
    pdf = gauss.pdf(np.linspace(0, 1, size))
    label_dis = pdf/np.sum(pdf)
    return label_dis

class ssCDLLitModel(BaseLitModel):
    """Processing of training, evaluation and testing.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.model = model
        # self.meta_model = meta_model
        self.training_stage = 'main'
        self.epoch_num = 0
        self.optimizer_flag = 0  #
        self.batch_num = 0  #
        self.semi_sample_dict = {}
        self.sigma = args.sigma
        self.size = args.size
        self.switch = 0
        # self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser

    def on_epoch_start(self):
        if self.epoch_num % 5 == 0:
            self.optimizer_flag = 1 - self.optimizer_flag
            self.switch = 1  # control when to negative sample

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0: # optimizer_idx=0 is to update CDL-RL
            pos_sample = batch["positive_sample"] # get positive samples
            neg_sample = batch["negative_sample"] # get negative samples
            mode = batch["mode"]
            conf_ldl = batch["conf_ldl"]
            pred_distribution, rank_score_pos = self.model(pos_sample)    # get predicted confidence distribution and rank score for positive samples
            neg_pred_distribution, rank_score_neg = self.model(pos_sample, neg_sample, mode)  # get predicted confidence distribution and for negative samples
            if self.epoch_num < 30 + 1:
                loss = self.loss(pred_distribution, pos_sample, conf_ldl, neg_pred_distribution, rank_score_pos,
                                 rank_score_neg, 'main') # update CDL-RL without pseudo labeled data
            else:
                semi_sample = self.semi_sample_dict[str(self.batch_num)]["positive_semi_sample"] # get pseudo labeled data
                neg_semi_sample = self.semi_sample_dict[str(self.batch_num)]["negative_semi_sample"]
                data_ldl = semi_sample.clone()
                pred_semi_distribution, rank_score_semi = self.model(semi_sample) # get predicted confidence distribution and rank score for positive samples for pseudo labeled data
                neg_pred_semi_distribution, rank_score_semi_neg = self.model(semi_sample, neg_semi_sample, mode)
                ori_semi_distribution = data_ldl[:, -101:].unsqueeze(1)
                max_scores, max_index = torch.max(ori_semi_distribution, dim=2)
                max_value, max_index = torch.max(max_scores, dim=0)
                indices = max_scores.squeeze() > self.args.thereshold  # (4096,)

                semi_sample = semi_sample[indices]
                pred_semi_distribution = pred_semi_distribution[indices]
                neg_pred_semi_distribution = neg_pred_semi_distribution[indices]
                rank_score_semi = rank_score_semi[indices]
                rank_score_semi_neg = rank_score_semi_neg[indices]
                conf = []
                for triples in data_ldl:
                    ldl_new = triples[-101:]
                    conf.append(ldl_new)
                conf_ldl_semi = conf
                position_list = indices.nonzero(as_tuple=False).squeeze().tolist()
                if isinstance(position_list, int):
                    position_list = [position_list]
                filtered_conf_ldl_semi = [conf_ldl_semi[i] for i in position_list]
                if len(filtered_conf_ldl_semi) == 0:
                    semi_sample = None
                loss = self.loss(pred_distribution, pos_sample, conf_ldl, neg_pred_distribution, rank_score_pos,
                                 rank_score_neg, 'main', semi_sample, pred_semi_distribution, filtered_conf_ldl_semi,
                                 rank_score_semi, neg_pred_semi_distribution, rank_score_semi_neg) # update CDL-RL with pseudo labeled data

        if optimizer_idx == 1:# optimizer_idx=1 is to update PCDG

            if self.epoch_num < 30:
                return None
            for param in self.model.mlp_conf.parameters():
                param.requires_grad = True
            for param in self.model.mlp_rank.parameters():
                param.requires_grad = True
            for param in self.model.mlp_tmp.parameters():
                param.requires_grad = True
            self.training_stage = 'meta'
            pos_sample = batch["positive_sample"] # get positive samples
            neg_sample = batch["negative_sample"] # get negative samples
            mode = batch["mode"]
            conf_ldl = batch["conf_ldl"]
            pred_distribution, rank_score_pos = self.model(pos_sample) # get predicted confidence distribution and rank score for positive samples
            neg_pred_distribution, rank_score_neg = self.model(pos_sample, neg_sample, mode)   # get predicted confidence distribution and for negative samples

            if self.switch == 1: # if switch, we need to sample new unlabeled data
                device = pos_sample.device
                data = batch["ori_data"]
                pos_sample = batch["positive_sample"]
                mode = batch["mode"]
                semi_sample = []
                sample_all = {}
                neg_semi_sample = []
                if mode == "head-batch":
                    for h, r, t, _ in data:
                        neg_head = self.head_batch(h, r, t, 1)[0]
                        semi_sample.append([neg_head, r, t])
                else:
                    for h, r, t, _ in data:
                        neg_tail = self.tail_batch(h, r, t, 1)[0]
                        semi_sample.append([h, r, neg_tail])
                # semi_sample = torch.tensor(np.array(semi_sample)).to(device)
                for h, r, t in semi_sample:
                    if mode == "head-batch":
                        neg_head = self.head_batch(h, r, t, self.args.num_neg)
                        neg_semi_sample.append(neg_head)
                    else:
                        neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                        neg_semi_sample.append(neg_tail)
                sample_all["positive_semi_sample"] = torch.tensor(np.array(semi_sample)).to(device)
                sample_all["negative_semi_sample"] = torch.LongTensor(np.array(neg_semi_sample)).to(device)
                semi_sample = sample_all["positive_semi_sample"].to(device)
                neg_semi_sample = sample_all["negative_semi_sample"].to(device)
                self.semi_sample_dict[str(self.batch_num)] = sample_all
            else: # if don't switch, we can use the unlabeled data which have been sampled
                semi_sample = self.semi_sample_dict[str(self.batch_num)]["positive_semi_sample"][:, :3].to('cuda')
                neg_semi_sample = self.semi_sample_dict[str(self.batch_num)]["negative_semi_sample"].to('cuda')
            self.batch_num = self.batch_num + 1
            # semi_sample = self.semi_sample_dict[str(self.batch_num)][:, :3]
            # print(semi_sample)
            # print(semi_sample.size())
            neg_pred_semi_distribution, rank_score_semi_neg = self.model(semi_sample, neg_semi_sample, mode,
                                                                         stage='meta')
            pred_distribution_semi_meta, rank_score_pos_semi_meta = self.model(semi_sample,
                                                                               stage='meta')
            weights = torch.linspace(0, 1, steps=101).to('cuda')
            semi_scores = (pred_distribution_semi_meta * weights).sum(dim=2).squeeze()
            semi_scores = (semi_scores - self.args.lower_bound) * (1.0 - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1

            pred_distribution_semi_meta_tmp = pred_distribution_semi_meta.squeeze(1)
            semi_scores = semi_scores.unsqueeze(1)
            semi_sample = torch.cat((semi_sample, semi_scores, pred_distribution_semi_meta_tmp), dim=1).to('cuda')
            data_ldl = semi_sample.clone()
            pred_distribution_semi, rank_score_semi = self.model(semi_sample, stage='main')  # use PCDG to generate D_tmp, which aims to get the gradient in PCDG
            max_scores, _ = torch.max(pred_distribution_semi_meta, dim=2)
            max_value, max_index = torch.max(max_scores, dim=0)
            if self.epoch_num <= 50:
                indices = max_scores.squeeze() >= 0  # (4096,)
            else:
                indices = max_scores.squeeze() >= self.args.thereshold # threshold of pseudo labeled data selection

            semi_sample = semi_sample[indices]
            if semi_sample.size()[0] == 0:
                semi_sample = None

            pred_distribution_semi = pred_distribution_semi[indices]
            neg_pred_semi_distribution = neg_pred_semi_distribution[indices]
            rank_score_semi = rank_score_semi[indices]
            rank_score_semi_neg = rank_score_semi_neg[indices]
            conf = []

            conf_ldl_semi_tensor = data_ldl[:, -101:]
            conf_ldl_semi_tensor = conf_ldl_semi_tensor.unsqueeze(1) #(4096, 1, 101)


            conf_ldl_semi = conf
            position_list = indices.nonzero(as_tuple=False).squeeze().tolist()
            # print(position_list)
            if isinstance(position_list, int):  #
                position_list = [position_list]
            filtered_conf_ldl_semi_tensor = conf_ldl_semi_tensor[indices]

            # use labeled data and unlabeled data to update CDL-RL once
            loss = self.loss(pred_distribution, pos_sample, conf_ldl, neg_pred_distribution, rank_score_pos,
                             rank_score_neg, 'meta', semi_sample, pred_distribution_semi,
                             filtered_conf_ldl_semi_tensor, rank_score_semi, neg_pred_semi_distribution,
                             rank_score_semi_neg)

            # get the parameters of CDL-RL after updating it once, i.e., theta
            for param in self.model.mlp_conf.parameters():
                param.requires_grad = True
            fast_weights = OrderedDict(
                (name, param) for (name, param) in self.model.mlp_conf.named_parameters())

            if semi_sample is not None:
                grads = torch.autograd.grad(loss, self.model.mlp_conf.parameters(), allow_unused=True,
                                            retain_graph=True, create_graph=True)
            else:
                grads = [torch.zeros_like(param) for param in self.model.mlp_conf.parameters()]

            data = [p.data for p in list(self.model.mlp_conf.parameters())]
            if self.epoch_num < 100:
                lr = self.args.lr
            else:
                lr = self.args.lr / 10

            # get theta+
            fast_weights = OrderedDict(
                (name, param - lr * grad if grad is not None else param) for
                ((name, param), grad, data) in
                zip(fast_weights.items(), grads, data))

            pred_distribution, rank_score_pos = self.model(pos_sample, stage='tmp', fast_weights=fast_weights)

            # update PCDG
            loss = self.loss(pred_distribution, pos_sample, conf_ldl, neg_pred_distribution, rank_score_pos,
                             rank_score_neg, 'meta')

        output = {'loss': loss, 'idx': optimizer_idx}
        self.log("Train|loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # print(loss)
        return loss

    def training_epoch_end(self, results) -> None:
        self.epoch_num = self.epoch_num + 1
        self.batch_num = 0
        if self.epoch_num >= 5 - 1:
            for batchs_num in self.semi_sample_dict.keys():
                semi_sample = self.semi_sample_dict[batchs_num]["positive_semi_sample"]
                pred_distribution, rank_score = self.model(semi_sample, stage='meta')
                semi_sample = semi_sample[:, :3]

                weights = torch.linspace(0, 1, steps=101).to('cuda')
                semi_scores = (pred_distribution * weights).sum(dim=2).squeeze()
                semi_scores = (semi_scores - self.args.lower_bound) * (1.0 - 0.1) / (self.args.upper_bound - self.args.lower_bound) + 0.1

                semi_scores = semi_scores.clone().detach()
                semi_scores = semi_scores.unsqueeze(1)
                pred_distribution = pred_distribution.squeeze(1)
                pred_distribution = pred_distribution.clone().detach()
                semi_sample = torch.cat((semi_sample, semi_scores), dim=1).to('cuda')
                semi_sample = torch.cat((semi_sample, pred_distribution), dim=1).to('cuda')

                semi_sample.requires_grad_(False)
                # print(semi_sample.size())
                self.semi_sample_dict[batchs_num]["positive_semi_sample"] = semi_sample  # record pseudo labeled data based on batch_num
        if self.switch == 1:
            self.switch = 0

    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.

        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: MAE, MSE, wmr, wmrr, mr, mrr and hits@1,3,5,10.
        """
        results = dict()

        results["count_for_conf"] = batch['positive_sample'].shape[0]
        MAE, MSE = conf_predict_two(batch, self.model, self.args.lower_bound, self.args.upper_bound)
        results["MAE"] = MAE.item()
        results["MSE"] = MSE.item()

        prediction = "tail"

        confidence = self.args.confidence_filter
        ranks = link_predict(batch, self.model, prediction=prediction)
        ranks_link_predict = link_predict_filter(batch, self.model, confidence, prediction=prediction)
        ranks_link_predict_raw = link_predict_raw(batch, self.model, confidence, prediction=prediction)
        results["count_for_link"] = torch.numel(ranks_link_predict)
        results["mrr"] = torch.sum(1.0 / ranks_link_predict).item()
        results["mr"] = torch.sum(ranks_link_predict).item()
        results["raw_count_for_link"] = torch.numel(ranks_link_predict_raw)
        results["raw_mrr"] = torch.sum(1.0 / ranks_link_predict_raw).item()
        results["raw_mr"] = torch.sum(ranks_link_predict_raw).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks_link_predict[ranks_link_predict <= k])
            results['raw_hits@{}'.format(k)] = torch.numel(ranks_link_predict_raw[ranks_link_predict_raw <= k])

        pos_triple = batch["positive_sample"]
        mask = pos_triple[:, -1] >= confidence
        """calculate WMR(Weighted-MR) and WMRR """
        if prediction == "all":
            conf = torch.cat([batch['positive_sample'][:, 3]] * 2)
            conf_high_score = conf[mask]
            results["wmr"] = torch.sum(ranks_link_predict * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict
            results["wmrr"] = torch.sum(ranks_mrr * conf_high_score)
            results["sum_for_conf"] = torch.sum(conf_high_score)
            results["raw_wmr"] = torch.sum(ranks_link_predict_raw * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict_raw
            results["raw_wmrr"] = torch.sum(ranks_mrr * conf_high_score)

        else:
            conf = batch['positive_sample'][:, 3]
            conf_high_score = conf[mask]
            results["wmr"] = torch.sum(ranks_link_predict * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict
            results["wmrr"] = torch.sum(ranks_mrr * conf_high_score)
            results["sum_for_conf"] = torch.sum(conf_high_score)
            results["raw_wmr"] = torch.sum(ranks_link_predict_raw * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict_raw
            results["raw_wmrr"] = torch.sum(ranks_mrr * conf_high_score)

        return results

    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        # with open(os.path.join(self.args.data_path, "ndcg_test.pickle"), 'rb') as f:
        #     hr_map = pickle.load(f)  # unpickle
        # ent_num = self.args.num_ent
        # ndcg, exp_ndcg = mean_ndcg(hr_map, self.model, ent_num)
        # outputs["Test_ndcg"] = ndcg
        # outputs["Test_exp_ndcg"] = exp_ndcg
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        results = dict()

        results["count_for_conf"] = batch['positive_sample'].shape[0]
        MAE, MSE = conf_predict_two(batch, self.model, self.args.lower_bound, self.args.upper_bound)

        results["MAE"] = MAE.item()
        results["MSE"] = MSE.item()

        prediction = "tail"
        confidence = self.args.confidence_filter
        ranks = link_predict(batch, self.model, prediction=prediction)
        ranks_link_predict = link_predict_filter(batch, self.model, confidence, prediction=prediction)
        ranks_link_predict_raw = link_predict_raw(batch, self.model, confidence, prediction=prediction)
        results["count_for_link"] = torch.numel(ranks_link_predict)
        results["mrr"] = torch.sum(1.0 / ranks_link_predict).item()
        results["mr"] = torch.sum(ranks_link_predict).item()
        results["raw_count_for_link"] = torch.numel(ranks_link_predict_raw)
        results["raw_mrr"] = torch.sum(1.0 / ranks_link_predict_raw).item()
        results["raw_mr"] = torch.sum(ranks_link_predict_raw).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks_link_predict[ranks_link_predict <= k])
            results['raw_hits@{}'.format(k)] = torch.numel(ranks_link_predict_raw[ranks_link_predict_raw <= k])

        """calculate WMR(Weighted-MR) and WMRR """
        pos_triple = batch["positive_sample"]
        mask = pos_triple[:, -1] >= confidence
        if prediction == "all":
            conf = torch.cat([batch['positive_sample'][:, 3]] * 2)
            conf_high_score = conf[mask]
            results["wmr"] = torch.sum(ranks_link_predict * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict
            results["wmrr"] = torch.sum(ranks_mrr * conf_high_score)
            results["sum_for_conf"] = torch.sum(conf_high_score)
            results["raw_wmr"] = torch.sum(ranks_link_predict_raw * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict_raw
            results["raw_wmrr"] = torch.sum(ranks_mrr * conf_high_score)

        else:
            conf = batch['positive_sample'][:, 3]
            conf_high_score = conf[mask]
            results["wmr"] = torch.sum(ranks_link_predict * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict
            results["wmrr"] = torch.sum(ranks_mrr * conf_high_score)
            results["sum_for_conf"] = torch.sum(conf_high_score)
            results["raw_wmr"] = torch.sum(ranks_link_predict_raw * conf_high_score)
            ranks_mrr = 1.0 / ranks_link_predict_raw
            results["raw_wmrr"] = torch.sum(ranks_mrr * conf_high_score)

        return results

    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")

        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.
        Returns:
            optim_dict: Record the optimizer and optimizer for meta self-training
        """
        # milestones = int(self.args.max_epochs / 4)
        milestones = 100
        beta1 = 0.9  # set β1
        beta2 = 0.99  # set β2
        confs_paras = (list(self.model.ent_emb.parameters()) + list(self.model.rel_emb.parameters()) + list(
            self.model.mlp_conf.parameters()) + list(self.model.mlp_rank.parameters()))
        meta_paras = (list(self.model.mlp_conf_meta.parameters()))
        optimizer = self.optimizer_class(confs_paras, lr=self.args.lr, betas=(beta1, beta2), eps=1e-5)
        optimizer_meta = self.optimizer_class(meta_paras, lr=self.args.lr, betas=(beta1, beta2), eps=1e-5)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        # StepLR_meta = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        # optim_dict = [{'optimizer': optimizer, 'lr_scheduler': StepLR},
        #               {'optimizer': optimizer_meta, 'lr_scheduler': StepLR_meta}]
        return [optimizer, optimizer_meta]
        # return optim_dict
