# HSE Project Time Series

Содержит исследование, а также CLI-приложение для работы с временными рядами: загрузка, кластеризация, трансформации, обучение моделей и генерация метрик и графиков.

---

## Установка

1. Клонировать репозиторий:

```bash
git clone git@github.com:AlbaAlba3/hse_project_time_series.git
cd hse_project_time_series
```

2. Создаем виртуальное окружение

```bash
python3 -m venv venv
```

и активируем его:

```bash
. venv/bin/activate
```

3. Устанавливаем зависимости

```bash
pip install -e .
```

## CLI

Если вы все сделали правильно, то у вас будут доступны команды для реализации пайплайна, аналогичному пайплайну из ноутбука. Для начала создадим лейблы для кластеризации:

```bash
(venv) experiment get-cluster-labels \
  --tsf_file /PATH/TO/data/m4_yearly_dataset.tsf \
  --output_csv /PATH/TO/results/clusters.pkl \
  --max_rows 150
```
замените `/PATH/TO` на нужный путь к файлам. У данной команды есть еще несколько аргументов:
* --tsf_file – путь к TSF-файлу с временными рядами (обязательный).
* --max_rows – максимальное количество рядов для обработки (по умолчанию 150).
* --n_clusters – количество кластеров (по умолчанию 7).
* --output_csv – путь к pickle-файлу для сохранения меток кластеров (обязательный).

После можем запустить пайплайн:

```bash
experiment run \
  --tsf_file /PATH/TO/data/m4_yearly_dataset.tsf \
  --cluster_file /PATH/TO/results/clusters.pkl \
  --transform null \
  --output /PATH/TO/results
```
где `cluster_file` указывает на файл, который мы создали в предыдущей команде. Данная команда имеет еще несколько аргументов:

* --tsf_file – путь к TSF-файлу с рядами (обязательный).
* --max_rows – максимальное количество рядов (по умолчанию 150).
* --cluster_file – pickle-файл с метками кластеров (обязательный).
* --transform – трансформация ряда: null, log1p, diff, boxcox или all (по умолчанию null).
* --h – горизонт прогноза (по умолчанию 6).
* --l – длина окна лагов (по умолчанию 35).
* --output – папка для сохранения результатов (по умолчанию output).
* --n_plot – количество графиков для сохранения (по умолчанию 5).

В папке output (или указанной) для каждой трансформации создается подпапка с результатами:

```
output/
├─ null/             # или log1p/diff/boxcox
│  ├─ metrics_cb.csv
│  ├─ metrics_baseline.csv
│  ├─ metrics_by_cluster.csv
│  ├─ predictions.pkl
│  └─ plots/
│     ├─ forecast_0.png
│     ├─ forecast_1.png
│     └─ ...
```
* metrics_cb.csv – метрики CatBoost модели.
* metrics_baseline.csv – метрики бейзлайнов (naive, theta, ets).
* metrics_by_cluster.csv – метрики по каждому кластеру.
* predictions.pkl – предсказания модели и бейзлайнов.
* plots/ – графики прогнозов