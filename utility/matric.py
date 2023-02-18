import math
from collections import Counter

import numpy as np
from tensorboardX import SummaryWriter


def p_r_f1(labels_y, ranks, k):
    precisions = 0
    n_labels = 0
    for pre, gt in zip(ranks, labels_y):
        pre = sorted(list(enumerate(pre)), key=lambda x: x[1], reverse=True)

        for i in range(k):
            if gt[pre[i][0]] == 1:
                precisions = precisions + 1.0
        n_labels += Counter(gt)[1]

    P = precisions / (k * len(labels_y))
    R = precisions / n_labels
    if P == 0 and R == 0:
        f1 = 0.0
    else:
        f1 = 2 * P * R / (P + R)
    return P, R, f1


def ndcg(labels_y, ranks, k):
    log = [1 / math.log2(x + 2) for x in range(k)]
    result = []

    for pre, gt in zip(ranks, labels_y):
        pre = sorted(list(enumerate(pre)), key=lambda x: x[1], reverse=True)
        res = np.zeros(k)
        for i in range(k):
            if gt[pre[i][0]] == 1:
                res[i] = 1
        if np.sum(res) == 0:
            ndcg_cur = 0
        else:
            ndcg_cur = np.dot(np.array(res), log) / np.dot(-np.sort(-res), log)
        result.append(ndcg_cur)

    return np.sum(np.array(result)) / len(result)


def write_summary_tensorboard(config):
    writer = SummaryWriter(log_dir="runs/{}".format(config["alias"]))
    writer.add_scalar("model/")


def report_performance(y, pred, loss_value, ex_type, config, epoch_id):
    print("\n ********************" + ex_type + "********************")
    writer = SummaryWriter(log_dir="runs/{}".format(config["alias"]))
    for i in [1, 5, 10, 30, 50]:
        prf_ans = p_r_f1(y, pred, i)
        ndcg_ans = ndcg(y, pred, i)
        print(
            " P_{}:{:.6f}, R_{}:{:.6f}, f1_{}:{:.6f}, ndcg_{}:{:.6f}".format(
                i, prf_ans[0], i, prf_ans[1], i, prf_ans[2], i, ndcg_ans
            )
        )
        if ex_type == "val result":
            writer.add_scalar("model/valid/P_{}".format(i), prf_ans[0], epoch_id)
            writer.add_scalar("model/valid/R_{}".format(i), prf_ans[1], epoch_id)
            writer.add_scalar("model/valid/f1_{}".format(i), prf_ans[2], epoch_id)
            writer.add_scalar("model/valid/ndcg_{}".format(i), ndcg_ans, epoch_id)
        if ex_type == "test result":
            writer.add_scalar("model/test/P_{}".format(i), prf_ans[0], epoch_id)
            writer.add_scalar("model/test/R_{}".format(i), prf_ans[1], epoch_id)
            writer.add_scalar("model/test/f1_{}".format(i), prf_ans[2], epoch_id)
            writer.add_scalar("model/test/ndcg_{}".format(i), ndcg_ans, epoch_id)
    print("loss:{:.2f} ".format(loss_value))
