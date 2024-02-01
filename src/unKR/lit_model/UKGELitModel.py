from .BaseLitModel import BaseLitModel
from ..eval_task import *
import pickle
from ..eval_task.nDCG import mean_ndcg


class UKGELitModel(BaseLitModel):
    """Processing of training, evaluation and testing.
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

        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]

        pos_score = self.model(pos_sample)
        neg_score = self.model(pos_sample, neg_sample, mode)

        head_emb, relation_emb, tail_emb = self.model.tri2emb(pos_sample[:, :3].to(torch.int))
        head_emb = head_emb.squeeze(1)
        relation_emb = relation_emb.squeeze(1)
        tail_emb = tail_emb.squeeze(1)
        regularizer = (torch.sum(torch.square(head_emb)) / head_emb.shape[0]) + \
                      (torch.sum(torch.square(relation_emb)) / relation_emb.shape[0]) + \
                      (torch.sum(torch.square(tail_emb)) / tail_emb.shape[0])
        loss = self.loss(pos_score, neg_score, pos_sample) + 0.005 * regularizer
        self.log("Train|loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.
        
        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,5,10.
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

        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
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
