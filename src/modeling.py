import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from src.transforms import DiffTransform


def get_train(series_list, cluster_labels, H=6, L=35):
    X, y = [], []
    for idx, s in enumerate(series_list):
        cluster = cluster_labels[idx]
        N = len(s)
        if N >= L + H:
            for t in range(L, N - H + 1):
                feats = s[t-L:t]
                # Добавляем кластера, чтобы идентфицировать ряд
                X.append(np.append(feats, cluster))
                y.append(s[t:t+H])
    
    feature_names = [f'lag_{i}' for i in range(L)] + ['cluster']
    X = pd.DataFrame(X, columns=feature_names)
    X['cluster'] = X['cluster'].astype(int)
    return X, np.array(y), feature_names


def fit_models(X, y):
    models = {}
    for cluster in sorted(X["cluster"].unique()):
        mask = X["cluster"] == cluster
        x_train = X.loc[mask].drop(columns="cluster")
        y_train = y[mask]
        model = CatBoostRegressor(loss_function='MultiRMSE', iterations=1000, depth=6, verbose=3)
        model.fit(x_train, y_train)
        models[cluster] = model
    return models


def predict(
    models,
    raw_series_list,
    cluster_labels,
    pipeline,
    feature_names,
    H=6,
    L=35,
):
    transformed = pipeline.transform(raw_series_list)

    predictions = []
    true = []
    row_indices = []

    for idx, s in enumerate(transformed):
        cluster = cluster_labels[idx]
        N = len(s)

        if N < L + H:
            continue

        feats = s[N - H - L:N - H]
        X_test = pd.DataFrame(
            [feats],
            columns=feature_names[:-1]
        )

        model = models[cluster]
        pred = model.predict(X_test).flatten()

        predictions.append(pred)
        true.append(raw_series_list[idx][-H:])
        row_indices.append(idx)

    for t in pipeline.transforms:
        # В дифференцировании возникает проблема с индексми,
        # для этого мы приводим индексы в соответсвии
        # с порядком дифференцирования
        if isinstance(t, DiffTransform):
            t.last_values = [
                raw_series_list[idx][-H - t.order:-H]
                for idx in row_indices
            ]

    pred_real = pipeline.inverse(predictions)

    return pred_real, true, row_indices