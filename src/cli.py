from pathlib import Path

import click
import warnings

from src.clusters import load_and_prepare_series, scale_series, cluster_series, save_cluster_labels
import pickle
import pandas as pd
from src.transforms import NullTransform, Log1pTransform, DiffTransform, BoxCoxTransform, TransformPipeline
from src.modeling import get_train, fit_models, predict
from src.baselines import naive_forecast, theta_forecast, ets_forecast
from src.metrics import evaluate_all, evaluate_all_by_cluster
from src.plots import plot_forecasts

warnings.simplefilter("ignore")


@click.command()
@click.option('--tsf_file', type=click.Path(exists=True), required=True, help="Путь к TSF файлу")
@click.option('--max_rows', type=int, default=150, help="Максимальная длина ряда")
@click.option('--n_clusters', type=int, default=7, help="Количество кластеров")
@click.option('--output_csv', type=str, required=True, help="Имя CSV для сохранения меток")
def get_cluster_labels(tsf_file, max_rows, n_clusters, output_csv):
    """Загрузка рядов, кластеризация и сохранение меток в CSV"""
    print("Загрузка и подготовка рядов...")
    series_list, series_names = load_and_prepare_series(tsf_file, max_rows=max_rows)
    
    print("Масштабирование рядов...")
    X_scaled = scale_series(series_list)
    
    print(f"Кластеризация с n_clusters={n_clusters}...")
    cluster_labels = cluster_series(X_scaled, n_clusters=n_clusters)
    
    print("Сохранение меток кластеров в Pickle...")
    save_cluster_labels(series_names, cluster_labels, output_csv)
    print("Резльтат сохранен!")


BASELINE_FUNCS = {
    "naive": naive_forecast,
    "theta": theta_forecast,
    "ets": ets_forecast,
}

TRANSFORMS_MAP = {
    "null": NullTransform,
    "log1p": Log1pTransform,
    "diff": DiffTransform,
    "boxcox": BoxCoxTransform,
}
@click.command()
@click.option('--tsf_file', type=click.Path(exists=True), required=True, help="Путь к TSF файлу")
@click.option('--max_rows', type=int, default=150, help="Максимальная длина ряда")
@click.option('--cluster_file', required=True, type=click.Path(exists=True), help="Файл с метками кластеров Pickle")
@click.option('--transform', type=click.Choice(["null", "log1p", "diff", "boxcox", "all"]), default="null", help="Выбор трансформации")
@click.option('--h', type=int, default=6, help="Горизонт прогноза")
@click.option('--l', type=int, default=35, help="Длина окна лагов")
@click.option('--output', type=click.Path(), default="output", help="Папка для результатов")
@click.option('--n_plot', type=int, default=5, help="Количество графиков для сохранения")
def run_pipeline(tsf_file, max_rows, cluster_file, transform, h, l, output, n_plot):
    """CLI для обучения моделей на рядах с трансформациями и кластеризацией"""
    
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    series_list, _ = load_and_prepare_series(tsf_file, max_rows=max_rows)

    with open(cluster_file, "rb") as f:
        data = pickle.load(f)
    cluster_labels = data["cluster_label"]

    transforms_to_run = TRANSFORMS_MAP.items() if transform == "all" else [(transform, TRANSFORMS_MAP[transform])]

    for name, transform_class in transforms_to_run:
        print(f"\n=== Прогон пайплайна для трансформации: {name} ===")
        transform_pipeline = TransformPipeline([transform_class()])
        transformed = transform_pipeline.fit_transform(series_list)

        X_train, y_train, feature_names = get_train(transformed, cluster_labels, H=h, L=l)

        print("Обучение модели CatBoost...")
        models = fit_models(X_train, y_train)

        print("Получаем предсказания и обратное преобразование...")
        pred_real, true, row_indices = predict(models, series_list, cluster_labels, transform_pipeline, feature_names, H=h, L=l)

        baseline_preds = {fn_name: [] for fn_name in BASELINE_FUNCS}
        for s in series_list:
            for fn_name, fn in BASELINE_FUNCS.items():
                baseline_preds[fn_name].append(fn(s, H=h))

        metrics_catboost = evaluate_all(pred_real, true)
        metrics_baseline = {fn_name: evaluate_all(preds, true) for fn_name, preds in baseline_preds.items()}
        metrics_by_cluster = evaluate_all_by_cluster(pred_real, true, cluster_labels)

        transform_dir = output / name
        transform_dir.mkdir(exist_ok=True)

        pd.DataFrame([metrics_catboost]).to_csv(transform_dir / "metrics_cb.csv", index=False)
        pd.DataFrame([{"model": n, **m} for n, m in metrics_baseline.items()]).to_csv(transform_dir / "metrics_baseline.csv", index=False)
        metrics_by_cluster.to_csv(transform_dir / "metrics_by_cluster.csv", index=False)

        with open(transform_dir / "predictions.pkl", "wb") as f:
            pickle.dump({"pred_real": pred_real, "true": true, "row_indices": row_indices, "baseline": baseline_preds}, f)

        plot_forecasts(series_list, pred_real, true, row_indices, cluster_labels, baseline_preds, output_dir=transform_dir, H=h, n_rows=n_plot)


@click.group()
def cli():
    """Главная группа CLI команд"""
    pass

cli.add_command(get_cluster_labels, name="get-cluster-labels")
cli.add_command(run_pipeline, name="run")

if __name__ == "__main__":
    cli()