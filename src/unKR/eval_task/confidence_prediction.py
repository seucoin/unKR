import torch
import os
import torch.nn as nn


import csv

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

    pred_score = pred_score.squeeze()
    MAE_loss = nn.L1Loss(reduction="sum")
    # MAE = MAE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MAE = MAE_loss(pred_score, confidence)

    MSE_loss = nn.MSELoss(reduction="sum")
    # MSE = MSE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MSE = MSE_loss(pred_score, confidence)

    return MAE, MSE


def conf_predict_two(batch, model, lower_bound, upper_bound):
    """The goal of evaluate task is to predict the confidence of triples.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.

    Returns:
        MAE: Mean absolute error.
        MSE: Mean Square error.
    """
    weights = torch.linspace(0, 1, steps=101).to('cuda')
    pos_triple = batch["positive_sample"]
    confidence = pos_triple[:, 3]

    # get predicted confidence distribution and rank score
    pred_distribute, rank_score = model.forward(pos_triple)
    expected_values = (pred_distribute * weights).sum(dim=2).squeeze() # get predicted confidence
    max_indices = pred_distribute.squeeze(1).argmax(dim=1)

    expected_values = (expected_values - lower_bound) * (1.0 - 0.1) / (upper_bound - lower_bound) + 0.1 # normalize predicted confidence

    MAE_loss = nn.L1Loss(reduction="sum")
    MAE = MAE_loss(expected_values, confidence)

    MSE_loss = nn.MSELoss(reduction="sum")
    # MSE = MSE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MSE = MSE_loss(expected_values, confidence)

    return MAE, MSE
