import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(series, H=6):
    return np.array([series[-H-1]] * H)

def theta_forecast(series, H=6):
    series = np.asarray(series)
    slope = series[-H-1] - series[-H-2] if len(series) >= 2 else 0
    return series[-H-1] + slope * np.arange(1, H+1)

def ets_forecast(series, H=6):
    series = np.asarray(series)
    try:
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fit = model.fit()
        return fit.forecast(H)
    except Exception:
        return np.array([series[-H-1]] * H)
    
def predict_baselines(series_list, cluster_labels, H=6):
    naive_preds, theta_preds, ets_preds, true_list, row_indices = [], [], [], [], []

    for idx, series in enumerate(series_list):
        if len(series) >= H:
            row_indices.append(idx)
            true_vals = series[-H:]
            true_list.append(true_vals)
            naive_preds.append(naive_forecast(series, H))
            theta_preds.append(theta_forecast(series, H))
            ets_preds.append(ets_forecast(series, H))

    return {
        "naive": naive_preds,
        "theta": theta_preds,
        "ets": ets_preds,
        "true": true_list,
        "row_indices": row_indices,
        "cluster": [cluster_labels[idx] for idx in row_indices]
    }