from .BaseLitModel import BaseLitModel
from ..eval_task import *


class BEUrRELitModel(BaseLitModel):
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
        """Getting samples and training in BEUrRE model.

        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]

        self.model.true_head = batch['true_head']
        self.model.true_tail = batch['true_tail']

        loss = self.loss(self.model, pos_sample, neg_sample, self.args)
        self.log("Train|loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in BEUrRE model.

        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mae, mse, wmr, wmrr, mr mrr and hits@1,3,5,10.
        """
        results = dict()
        """以MSE为指标 Early Stop"""
        results["count_for_conf"] = batch['positive_sample'].shape[0]
        MAE, MSE = conf_predict(batch, self.model)
        results["MAE"] = MAE.item()
        results["MSE"] = MSE.item()

        prediction = "tail"
        """以MRR为指标 Early Stop"""
        ranks = link_predict(batch, self.model, prediction=prediction)
        results["count_for_link"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        results["mr"] = torch.sum(ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])

        """计算WMR   Weighted-MR """
        if prediction == "all":  # 头尾实体预测平均
            conf = torch.cat([batch['positive_sample'][:, 3]] * 2)
            results["wmr"] = torch.sum(ranks * conf)
            ranks_mrr = 1.0 / ranks
            results["wmrr"] = torch.sum(ranks_mrr * conf)
            results["sum_for_conf"] = torch.sum(conf)
        else: # 头尾实体预测
            conf = batch['positive_sample'][:, 3]
            results["wmr"] = torch.sum(ranks * conf)
            ranks_mrr = 1.0 / ranks
            results["wmrr"] = torch.sum(ranks_mrr * conf)
            results["sum_for_conf"] = torch.sum(conf)
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
        """Getting samples and test in BEUrRE model.

        Args:
            batch: The evaluation data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mae, mse, wmr, wmrr, mr mrr and hits@1,3,5,10.
        """
        results = dict()

        """以MSE为指标 Early Stop"""
        results["count_for_conf"] = batch['positive_sample'].shape[0]
        MAE, MSE = conf_predict(batch, self.model)
        results["MAE"] = MAE.item()
        results["MSE"] = MSE.item()

        prediction = "tail"
        """以MRR为指标 Early Stop"""
        ranks = link_predict(batch, self.model, prediction=prediction)
        results["count_for_link"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        results["mr"] = torch.sum(ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])

        """计算WMR   Weighted-MR """
        if prediction == "all":  # 头尾实体预测平均
            conf = torch.cat([batch['positive_sample'][:, 3]] * 2)
            results["wmr"] = torch.sum(ranks * conf)
            ranks_mrr = 1.0 / ranks
            results["wmrr"] = torch.sum(ranks_mrr * conf)
            results["sum_for_conf"] = torch.sum(conf)
        else: # 头尾实体预测
            conf = batch['positive_sample'][:, 3]
            results["wmr"] = torch.sum(ranks * conf)
            ranks_mrr = 1.0 / ranks
            results["wmrr"] = torch.sum(ranks_mrr * conf)
            results["sum_for_conf"] = torch.sum(conf)

        return results

    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")

        # with open(os.path.join(self.args.data_path, "ndcg_test.pickle"), 'rb') as f:
        #     hr_map = pickle.load(f)  # unpickle
        # ent_num = self.args.num_ent
        # ndcg, exp_ndcg = mean_ndcg(hr_map, self.model, ent_num)
        # outputs["Test_ndcg"] = ndcg
        # outputs["Test_exp_ndcg"] = exp_ndcg

        self.log_dict(outputs, prog_bar=True, on_epoch=True)


    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.
        """
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        # optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        optim_dict = {'optimizer': optimizer}
        return optim_dict
