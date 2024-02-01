import numpy as np
from .BaseLitModel import BaseLitModel
from ..eval_task import *
import random


class PASSLEAFLitModel(BaseLitModel):
    """Processing of training, evaluation and testing for PASSLEAF.
    """

    def __init__(self, model, args):
        super().__init__(model, args)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser


    def training_step(self, batch, batch_idx):
        """Getting samples and training in KG model.

        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        #
        if self.train_epoch >= self.args.T_new_semi:
            # when epoch > T_new_semi, get negative samples Dynamically
            pos_sample = batch["positive_sample"]
            device = pos_sample.device
            data = batch["ori_data"]
            mode = batch["mode"]
            neg_ent_sample = []
            # get semi-supervised samples in the same way as getting negative samples
            if mode == "head-batch":
                for h, r, t, _ in data:
                    neg_head = self.head_batch(h, r, t, self.args.num_neg)
                    neg_ent_sample.append(neg_head)
            else:
                for h, r, t, _ in data:
                    neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                    neg_ent_sample.append(neg_tail)
            batch['semi_sample'] = torch.LongTensor(np.array(neg_ent_sample)).to(device)

        if self.train_epoch < self.args.T_new_semi:
            # when self.train_epoch < self.args.T_new_semi, train as normal UKGE
            pos_sample = batch["positive_sample"] # get positive samples
            neg_sample = batch["negative_sample"] # get negative samples
            mode = batch["mode"]
            pos_score = self.model(pos_sample) # calculate the score of postive samples
            neg_score = self.model(pos_sample, neg_sample, mode) # calculate the score of negative samples
            head_emb, relation_emb, tail_emb = self.model.tri2emb(pos_sample[:, :3].to(torch.int))
            head_emb = head_emb.squeeze(1)
            relation_emb = relation_emb.squeeze(1)
            tail_emb = tail_emb.squeeze(1)

            regularizer = (torch.sum(torch.square(head_emb)) / head_emb.shape[0]) + \
                          (torch.sum(torch.square(relation_emb)) / relation_emb.shape[0]) + \
                          (torch.sum(torch.square(tail_emb)) / tail_emb.shape[0])

            loss = self.loss(pos_score, neg_score, pos_sample) + 0.005 * regularizer # the same loss as UKGE
            self.log("Train|loss", loss, on_step=False, on_epoch=True)

            if self.train_epoch == 0: # get all entities and relations in epoch 0
                for data in batch['positive_sample']:
                    self.entity.add(data[0])
                    self.entity.add(data[2])
                    self.relation.add(data[1])
                    self.training_set.append(data)

        if self.train_epoch >= self.args.T_new_semi and self.train_epoch < self.args.T_semi_train:
            # begin to add samples to the pool
            pos_sample = batch["positive_sample"]
            neg_sample = batch["negative_sample"]
            semi_sample = batch["semi_sample"] # get semi_samples, the semi_samples were sampled in semi_sample, they will be different in each epoch
            # the number of semi_samples is equal to the number of neg_samples
            mode = batch["mode"]
            device = pos_sample.device
            pos_score = self.model(pos_sample)
            neg_score = self.model(pos_sample, neg_sample, mode)
            semi_score = self.model(pos_sample, semi_sample, mode) # calculate the score of semi-supervised samples to be added to the pool
            semi_sample_format = self.convert_semi_samples_batch(semi_sample, semi_score, pos_sample, mode, device) # convert entities into concrete semi triples
            semi_sample_format_list = torch.tensor(semi_sample_format, device=device, dtype=torch.float64)
            if len(self.pool) <= self.args.max_pool: # add samples to the pool
                self.pool.extend(semi_sample_format_list)

            head_emb, relation_emb, tail_emb = self.model.tri2emb(pos_sample[:, :3].to(torch.int))
            head_emb = head_emb.squeeze(1)
            relation_emb = relation_emb.squeeze(1)
            tail_emb = tail_emb.squeeze(1)

            regularizer = (torch.sum(torch.square(head_emb)) / head_emb.shape[0]) + \
                          (torch.sum(torch.square(relation_emb)) / relation_emb.shape[0]) + \
                          (torch.sum(torch.square(tail_emb)) / tail_emb.shape[0])
            loss = self.loss(pos_score, neg_score, pos_sample) + 0.005 * regularizer
            self.log("Train|loss", loss, on_step=False, on_epoch=True)

        if self.train_epoch >= self.args.T_semi_train:
            # begin to use samples in the pool
            n_generated_samples = len(batch['positive_sample']) * self.neg_per_positive  # the number of all samples(neg + semi) used for training
            n_semi_samples = int(
                min(n_generated_samples * 0.8, max(0,
                                                   -self.args.T_semi_train + self.train_epoch + 1) * n_generated_samples * self.args.alpha_PASSLEAF)) # number of semi samples to be used in this epoch
            N_neg = n_generated_samples - n_semi_samples # number of negative samples to be used in this epoch

            pos_sample = batch["positive_sample"]
            neg_sample = batch["negative_sample"]
            mode = batch["mode"]

            semi_sample = random.sample(self.pool, n_semi_samples)  # selected semi_samples from the pool
            semi_sample_stacked = torch.stack(semi_sample)

            format_neg_sample = self.convert_neg_samples_batch(pos_sample, neg_sample, mode) # convert entities into concrete neg triples
            selected_neg_samples = self.select_random_elements(format_neg_sample, N_neg) # selected neg_samples in the epoch
            pos_score = self.model(pos_sample) # get the score of positive triples
            semi_score = self.model(semi_sample_stacked)
            # the format of pos_sample, semi_sample_stacked, selected_neg_samples are the same
            # So we can directly emulate the calculation method of pos_score to calculate neg_score and semi_score
            neg_score = self.model(selected_neg_samples)
            head_emb, relation_emb, tail_emb = self.model.tri2emb(pos_sample[:, :3].to(torch.int))
            head_emb = head_emb.squeeze(1)
            relation_emb = relation_emb.squeeze(1)
            tail_emb = tail_emb.squeeze(1)

            regularizer = (torch.sum(torch.square(head_emb)) / head_emb.shape[0]) + \
                          (torch.sum(torch.square(relation_emb)) / relation_emb.shape[0]) + \
                          (torch.sum(torch.square(tail_emb)) / tail_emb.shape[0])
            # The loss will change when add semi_score
            loss = self.loss(pos_score, neg_score, pos_sample, semi_score, semi_sample_stacked) + 0.005 * regularizer
            self.log("Train|loss", loss, on_step=False, on_epoch=True)

            pos_sample = batch["positive_sample"]
            neg_sample = batch["negative_sample"]
            semi_sample = batch["semi_sample"]

            mode = batch["mode"]
            device = pos_sample.device
            semi_score = self.model(pos_sample, semi_sample, mode)
            semi_sample_format = self.convert_semi_samples_batch(semi_sample, semi_score, pos_sample, mode, device) # get semi-samples in the same way as negative samples
            semi_sample_format_list = torch.tensor(semi_sample_format, device=device, dtype=torch.float64)
            if len(self.pool) <= self.args.max_pool: # add samples to the pool
                self.pool.extend(semi_sample_format_list)

        return loss

    def training_epoch_end(self, loss) -> None:
        # get hr2t_train and rt2h_train when epoch is 0, they will be used in the sampling of semi samples
        if self.train_epoch == 0:
            for i in self.training_set:
                self.training_set_triples_list.append(i[0:3].tolist())
            for h, r, t in self.training_set_triples_list:
                self.hr2t_train[(h, r)].add(t)
                self.rt2h_train[(r, t)].add(h)
        self.train_epoch += 1


    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.

        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: MAE, MSE, mrr, mr, wmrr, wmr, hits@1,3,5,10, raw_mrr, raw_wmr, raw_wmrr, raw_wmr, raw_hits@1,3,5,10
        """
        results = dict()

        results["count_for_conf"] = batch['positive_sample'].shape[0]
        MAE, MSE = conf_predict(batch, self.model)
        results["MAE"] = MAE.item()
        results["MSE"] = MSE.item()

        prediction = "tail"
        confidence = self.args.confidence_filter
        ranks = link_predict(batch, self.model, prediction=prediction)
        ranks_link_predict = link_predict_filter(batch, self.model, confidence, prediction=prediction) # get the ranks of entities in link prediction
        ranks_link_predict_raw = link_predict_raw(batch, self.model, confidence, prediction=prediction) # get the raw rank (no filter when calculate the rank)
        results["count_for_link"] = torch.numel(ranks_link_predict)
        results["mrr"] = torch.sum(1.0 / ranks_link_predict).item()
        results["mr"] = torch.sum(ranks_link_predict).item()
        results["raw_count_for_link"] = torch.numel(ranks_link_predict_raw)
        results["raw_mrr"] = torch.sum(1.0 / ranks_link_predict_raw).item()
        results["raw_mr"] = torch.sum(ranks_link_predict_raw).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks_link_predict[ranks_link_predict <= k])
            results['raw_hits@{}'.format(k)] = torch.numel(ranks_link_predict_raw[ranks_link_predict_raw <= k])

        """calculate WMR   Weighted-MR """
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


    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.

        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: MAE, MSE, mrr, mr, wmrr, wmr, hits@1,3,5,10, raw_mrr, raw_wmr, raw_wmrr, raw_wmr, raw_hits@1,3,5,10
        """
        results = dict()

        results["count_for_conf"] = batch['positive_sample'].shape[0]
        MAE, MSE = conf_predict(batch, self.model)
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

        """calculate WMR   Weighted-MR """
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
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.
        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.
        """
        milestones = int(self.args.max_epochs / 8)
        beta1 = 0.9  # set β1
        beta2 = 0.99  # set β2
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, betas=(beta1, beta2))
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
