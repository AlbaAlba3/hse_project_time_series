import pandas as pd
import numpy as np
from collections import defaultdict

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)

    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def evaluate_all(pred_list, true_list):
    smape_list, mae_list, rmse_list = [], [], []

    for pred, true in zip(pred_list, true_list):
        smape_list.append(smape(true, pred))
        mae_list.append(mae(true, pred))
        rmse_list.append(rmse(true, pred))

    return {
        "SMAPE_mean": np.mean(smape_list),
        "MAE_mean": np.mean(mae_list),
        "RMSE_mean": np.mean(rmse_list),
        # "SMAPE_all": smape_list,
    }


def evaluate_all_by_cluster(pred_list, true_list, cluster_labels):
    cluster_metrics = defaultdict(lambda: {"SMAPE": [], "MAE": [], "RMSE": []})
    smape_all, mae_all, rmse_all = [], [], []

    for pred, true, cl in zip(pred_list, true_list, cluster_labels):
        s = smape(true, pred)
        m = mae(true, pred)
        r = rmse(true, pred)

        cluster_metrics[cl]["SMAPE"].append(s)
        cluster_metrics[cl]["MAE"].append(m)
        cluster_metrics[cl]["RMSE"].append(r)

        smape_all.append(s)
        mae_all.append(m)
        rmse_all.append(r)

    rows = []
    for cl, metrics in cluster_metrics.items():
        rows.append({
            "cluster": cl,
            "SMAPE_mean": np.mean(metrics["SMAPE"]),
            "MAE_mean": np.mean(metrics["MAE"]),
            "RMSE_mean": np.mean(metrics["RMSE"])
        })

    rows.append({
        "cluster": "overall",
        "SMAPE_mean": np.mean(smape_all),
        "MAE_mean": np.mean(mae_all),
        "RMSE_mean": np.mean(rmse_all)
    })

    df = pd.DataFrame(rows)
    return df