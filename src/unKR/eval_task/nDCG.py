import time
import torch
import os
import numpy as np
from tqdm import tqdm


def mean_ndcg(hr_map, model, ent_num):

    ndcg_sum = 0  # nDCG with linear gain
    exp_ndcg_sum = 0  # nDCG with exponential gain
    count = 0

    for h in tqdm(hr_map, desc="Testing_nDCG: "):
        for r in hr_map[h]:
            tw_dict = hr_map[h][r]  # {h:{r:{t:w}}}
            tw_truth = [IndexScore(t, w) for t, w in tw_dict.items()]
            tw_truth.sort(reverse=True)  # descending on w
            ndcg, exp_ndcg = nDCG(h, r, tw_truth, model, ent_num)  # nDCG with linear gain and exponential gain
            # print(ndcg, exp_ndcg)
            ndcg_sum += ndcg
            exp_ndcg_sum += exp_ndcg
            count += 1

    return ndcg_sum / count, exp_ndcg_sum / count


def nDCG(h, r, tw_truth, model, ent_num):
    """
    Compute nDCG(normalized discounted cummulative gain)
    sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
    :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
    :return:
    """
    # prediction
    ts = [tw.index for tw in tw_truth]             # 真实的排名       [25,60,75,3,7]
    ranks = get_t_ranks(h, r, ts, model, ent_num)  # 根据模型给出的排序 [24,68,222,1,6]

    # linear gain
    gains = np.array([tw.score for tw in tw_truth])  # 从大到小排序好的
    discounts = np.log2(ranks + 1)
    discounted_gains = gains / discounts
    dcg = np.sum(discounted_gains)  # discounted cumulative gain
    # normalize 理想中的排序
    idcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
    ndcg = dcg / idcg  # normalized discounted cumulative gain

    # exponential gain
    exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
    exp_discounted_gains = exp_gains / discounts
    exp_dcg = np.sum(exp_discounted_gains)
    # normalize
    exp_idcg = np.sum(
        exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
    exp_ndcg = exp_dcg / exp_idcg  # normalized discounted cumulative gain

    return ndcg, exp_ndcg


def get_t_ranks(h, r, ts, model, ent_num):
    """
    Given some t index, return the ranks for each t
    :return:
    """
    # prediction
    # scores = get_score(h, r, ts, model)  # 对当前的hr{t:w}匹配对进行预测

    N = ent_num   # 预测所有的hrt_1...N
    score_list = get_score(h, r, range(N), model)

    # way1:
    # ranks = np.ones(len(ts), dtype=int)  # initialize rank as all 1
    # for i in range(N):  # compute scores for all concept vectors as t
    #     score_i = score_list[i]
    #     rankplus = (scores < score_i).astype(int)  # rank+1 if score<score_i
    #     ranks += rankplus
    # print("ranks1:", ranks)

    # 使用argsort函数得到排序后的索引
    rank_indices = list(np.argsort(score_list)[::-1])
    ranks = [rank_indices.index(i) + 1 for i in ts]
    ranks = np.array(ranks)
    # print("ranks2:", ranks)

    return ranks


def get_score(h, r, ts, model):
    batch = {}
    # triples = [(h, r, t) for t in ts]
    triples = [(h, r, t) for t in ts]
    batch["positive_sample"] = torch.tensor(np.array(triples))
    scores = model.get_score(batch, "single")
    scores = scores.squeeze()
    scores = scores.cpu()
    scores = scores.detach().numpy()

    return scores


class IndexScore:
    """
    The score of a tail when h and r is given.
    It's used in the ranking task to facilitate comparison and sorting.
    Print w as 3 digit precision float.
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return "(index: %d, w:%.3f)" % (self.index, self.score)
        return "(%d, %.3f)" % (self.index, self.score)

    def __str__(self):
        return "(index: %d, w:%.3f)" % (self.index, self.score)