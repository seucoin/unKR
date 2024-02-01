import torch
import torch.nn as nn

class BEUrRE_Loss(nn.Module):
    """The loss function of BEUrRE

    Attributes:
        args: Some pre-set parameters, etc
        model: The BEUrRE model for training.
    """
    def __init__(self, args, model):
        super(BEUrRE_Loss, self).__init__()
        self.args = args
        self.model = model

    def get_logic_loss(self, model, ids, args):
        """
        Calculates the logic loss for the model based on transitive and composite rule regularizations.

        Args:
            model: The BEUrRE model instance.
            ids: Tensor of triple ids.
            args: Model configuration parameters including regularization coefficients.

        Returns:
            The logic loss calculated from transitive and composite rule regularizations.
        """
        ids = ids[:, :3].to(torch.long)
        # transitive rule loss regularization
        transitive_coff = torch.tensor(args.regularization['transitive']).to(args.gpu)
        if transitive_coff > 0:
            transitive_rule_reg = transitive_coff * model.transitive_rule_loss(ids)
        else:
            transitive_rule_reg = 0

        # composite rule loss regularization
        composite_coff = torch.tensor(args.regularization['composite']).to(args.gpu)
        if composite_coff > 0:
            composition_rule_reg = composite_coff * model.composition_rule_loss(ids)
        else:
            composition_rule_reg = 0

        return (transitive_rule_reg + composition_rule_reg) / len(ids)


    def main_mse_loss(self, model, ids):
        """
        Computes the Mean Squared Error (MSE) loss for the model.

        Args:
            model: The BEUrRE model instance.
            ids: Tensor of triple ids(the confidence of negative samples is zero).

        Returns:
            mse: The MSE loss for the given triples.
        """
        criterion = nn.MSELoss(reduction='mean')
        prediction = model(ids[:, :3].to(torch.long), train=True).to(torch.float64)

        if ids.shape[1] > 3:
            truth = ids[:, 3]
        else:
            truth = torch.zeros(ids.shape[0], dtype=torch.float64).to(self.args.gpu)

        mse = criterion(prediction, truth)

        return mse


    def L2_regularization(self, model, ids, args):
        """
        Computes the L2 regularization loss for the model.

        Args:
            model: The BEUrRE model instance.
            ids: Tensor of triple ids.
            args: Model configuration parameters including regularization coefficients.

        Returns:
            L2_reg: The L2 regularization loss.
        """
        ids = ids[:, :3].to(torch.long)
        regularization = args.regularization
        device = args.gpu

        # regularization on delta
        delta_coff, min_coff = torch.tensor(regularization['delta']).to(device), torch.tensor(regularization['min']).to(
            device)
        delta_reg1 = delta_coff * torch.norm(torch.exp(model.delta_embedding[ids[:, 0]]), dim=1).mean()
        delta_reg2 = delta_coff * torch.norm(torch.exp(model.delta_embedding[ids[:, 2]]), dim=1).mean()

        min_reg1 = min_coff * torch.norm(model.min_embedding[ids[:, 0]], dim=1).mean()
        min_reg2 = min_coff * torch.norm(model.min_embedding[ids[:, 2]], dim=1).mean()

        rel_trans_coff = torch.tensor(regularization['rel_trans']).to(device)
        rel_trans_reg = rel_trans_coff * (
                torch.norm(torch.exp(model.rel_trans_for_head[ids[:, 1]]), dim=1).mean() + \
                torch.norm(torch.exp(model.rel_trans_for_tail[ids[:, 1]]), dim=1).mean()
        )

        rel_scale_coff = torch.tensor(regularization['rel_scale']).to(device)
        rel_scale_reg = rel_scale_coff * (
                torch.norm(torch.exp(model.rel_scale_for_head[ids[:, 1]]), dim=1).mean() + \
                torch.norm(torch.exp(model.rel_scale_for_tail[ids[:, 1]]), dim=1).mean()
        )

        L2_reg = delta_reg1 + delta_reg2 + min_reg1 + min_reg2 + rel_trans_reg + rel_scale_reg

        return L2_reg


    def forward(self, model, ids, negative_samples, args):
        """
        Calculating the total loss including MSE loss, logic loss, and L2 regularization.

        Args:
            model: The BEUrRE model instance.
            ids: Tensor of positive triple ids.
            negative_samples: Tensor of negative triple ids.
            args: Model configuration parameters including regularization coefficients.

        Returns:
            loss: The total loss combining MSE loss, logic loss, and L2 regularization.
        """
        NEG_RATIO = 1
        pos_loss = self.main_mse_loss(model, ids)

        negative_samples, _ = model.random_negative_sampling(ids[:, :3].to(torch.long), ids[:, 3])

        neg_loss = self.main_mse_loss(model, negative_samples)

        main_loss = pos_loss + NEG_RATIO * neg_loss

        if hasattr(args, 'RULE_CONFIGS'):
            logic_loss = self.get_logic_loss(model, ids, args)
        else:
            logic_loss = 0

        L2_reg = self.L2_regularization(model, ids, args)

        loss = main_loss + L2_reg + logic_loss

        return loss