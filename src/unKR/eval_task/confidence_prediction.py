import torch
import os
import torch.nn as nn


def conf_predict(batch, model):
    """The goal of evaluate task is to predict the confidence of triples.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.

    Returns:
        MAE: Mean absolute error.
        MSE: Mean Square error.
    """
    pos_triple = batch["positive_sample"]
    confidence = pos_triple[:, 3]

    pred_score = model.get_score(batch, "single")

    pred_score = pred_score.squeeze()  # 维度压缩
    MAE_loss = nn.L1Loss(reduction="sum")
    # MAE = MAE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MAE = MAE_loss(pred_score, confidence)

    MSE_loss = nn.MSELoss(reduction="sum")
    # MSE = MSE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MSE = MSE_loss(pred_score, confidence)

    return MAE, MSE
