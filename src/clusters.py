from sktime.datasets import load_tsf_to_dataframe
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import pandas as pd
import numpy as np
import pickle


def load_and_prepare_series(tsf_path, min_length=40, max_rows=None):
    """
    Загружает TSF, фильтрует по минимальной длине ряда и возвращает списки рядов и их имена.
    """
    df, _ = load_tsf_to_dataframe(tsf_path)
    df_reset = df.reset_index()
    
    grouped = df_reset.groupby('series_name')['series_value'].apply(list).reset_index()
    grouped.columns = ['series_name', 'values']
    
    grouped = grouped[grouped['values'].apply(len) >= min_length]
    grouped = grouped.sort_values('series_name')
    
    if max_rows is not None:
        grouped = grouped.head(max_rows)
    
    series_list = grouped['values'].tolist()
    series_names = grouped['series_name'].tolist()
    
    print(f"Отобрано {len(series_list)} рядов с длиной >= {min_length}")
    print("Минимальная длина среди отобранных:", min(len(v) for v in series_list))
    print("Средняя длина среди отобранных:", np.mean([len(v) for v in series_list]))
    
    return series_list, series_names

def scale_series(series_list):
    X = to_time_series_dataset(series_list)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled

def cluster_series(X_scaled, n_clusters=7, metric="softdtw", max_iter=5, max_iter_barycenter=5, random_state=0):
    km_model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        max_iter=max_iter,
        max_iter_barycenter=max_iter_barycenter,
        random_state=random_state
    ).fit(X_scaled)
    
    return km_model.labels_

def save_cluster_labels(series_names, cluster_labels, filename="cluster_labels.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"series_name": series_names, "cluster_label": cluster_labels}, f)
    print(f"Сохранено {len(cluster_labels)} меток кластеров в '{filename}'")
    return cluster_labels