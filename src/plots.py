import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_forecasts(
    series_list,
    pred_real,
    true,
    row_indices,
    cluster_labels,
    baseline_predictions,
    output_dir,
    H=6,
    n_rows=5
):
    """
    series_list: оригинальные ряды
    pred_real: предсказания модели
    true: фактические значения
    row_indices: индексы рядов
    cluster_labels: метки кластеров
    baseline_predictions: словарь с прогнозами бейзлайнов
    output_dir: Path или str к папке для сохранения графиков
    """
    output_dir = Path(output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(row_indices):
        history = series_list[idx]
        test_start = len(history) - H
        test_indices = np.arange(test_start, len(history))
        hist_indices = np.arange(len(history))

        plt.figure(figsize=(10,4))
        plt.plot(hist_indices, history, 'b-o', label='История', markersize=4)
        plt.plot(test_indices, true[i], 'g-o', label='Факт', markersize=4)
        plt.plot(test_indices, pred_real[i], 'r--o', label='Прогноз CatBoost', markersize=4)
        plt.plot(test_indices, baseline_predictions["naive"][i], 'k--', label='Наивный прогноз', markersize=4)
        plt.plot(test_indices, baseline_predictions["theta"][i], 'c--', label='Theta', markersize=4)
        plt.plot(test_indices, baseline_predictions["ets"][i], 'm--', label='ETS', markersize=4)
        
        plt.title(f'Ряд {idx}, кластер {cluster_labels[idx]}')
        plt.xlabel('t')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = output_dir / f"forecast_series_{idx}_cluster_{cluster_labels[idx]}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        if i + 1 >= n_rows:
            break