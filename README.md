# Teta ML 2026 — ML Inference Service (MLOps HW)

Сервис следит за папкой `./input`, принимает `test.csv`, выполняет препроцессинг и скоринг моделью, затем сохраняет результаты в `./output`.

## Архитектура

```
teta_mlops_service/
├── app/app.py                 # Watchdog-сервис: загрузка → препроцессинг → скоринг → выгрузка
├── src/preprocessing.py       # Feature engineering (логика из teta_ml.ipynb)
├── src/scorer.py              # Inference + top-5 FI + график плотности
├── src/constants.py           # Список признаков и гиперпараметры HGBR
├── scripts/build_model.py     # Обучение модели на train.csv (inference-only в контейнере)
├── models/                    # model_bundle.joblib (создаётся при сборке)
├── train_data/train.csv       # Reference-данные (скачать с Kaggle)
├── input/                     # Сюда кладётся test.csv
└── output/                    # Результаты скоринга
```

## Что делает сервис

| Этап | Скрипт | Описание |
|------|--------|----------|
| Загрузка | `app/app.py` | Читает CSV из `/app/input` |
| Препроцессинг | `src/preprocessing.py` | 41 признак: даты, geo, частоты, target encoding |
| Скоринг | `src/scorer.py` | HistGradientBoostingRegressor, clip [0, 365] |
| Выгрузка | `app/app.py` | CSV + JSON + PNG в `/app/output` |

### Выходные файлы (зачёт на 5)

После обработки `test.csv` в `./output` появятся:

1. **`sample_submission_<имя>_<timestamp>.csv`** — формат сабмита (`index`, `prediction`)
2. **`feature_importance_top5_<имя>_<timestamp>.json`** — топ-5 feature importances модели
3. **`score_density_<имя>_<timestamp>.png`** — график плотности распределения предсказаний

## Быстрый старт

### Требования

- Docker 20.10+
- ~2 ГБ свободного места
- Файлы соревнования: `train.csv`, `test.csv`

### 1. Подготовка данных

1. Скачайте `train.csv` и `test.csv` с Kaggle ([teta-ml-2026](https://www.kaggle.com/competitions/teta-ml-2026)).
2. Положите **`train.csv`** в папку `./train_data/`.
3. Создайте локальные папки (если их нет):

```bash
mkdir -p input output
```

### 2. Сборка Docker-образа

Из корня проекта:

```bash
docker build -t teta_ml_service .
```

При сборке автоматически запускается `scripts/build_model.py`: модель обучается на `train.csv` и сохраняется в `models/model_bundle.joblib`.

### 3. Запуск контейнера

```bash
docker run -it --rm \
  -v ./input:/app/input \
  -v ./output:/app/output \
  teta_ml_service
```

Дождитесь в логах строки:

```
File observer started. Put test.csv into ./input
```

### 4. Тест работоспособности

1. Скопируйте `test.csv` в `./input`:

```bash
cp /path/to/test.csv ./input/
```

2. Подождите завершения обработки (в логах: `Scoring pipeline finished successfully`).
3. Проверьте `./output/` — должны появиться CSV, JSON и PNG.



## Модель

### Что было в исходном решении (`teta_ml.ipynb`)

| Компонент | Описание |
|-----------|----------|
| **Признаки** | 41 числовой признак: даты, geo, логарифмы, частоты категорий, target encoding (OOF по 5-fold), квантильные бины по `sum` |
| **Базовые модели** | HistGradientBoosting + RandomForest + ExtraTrees + XGBoost |
| **Подбор гиперпараметров** | Optuna для HGBR и XGBoost |
| **Мета-уровень** | RidgeCV / простое среднее / simplex blend — выбор по OOF MSE |
| **Обучение** | 5-fold CV для OOF + финальный сабмит на test |
| **Цель** | Максимальное качество на Kaggle (MSE) |

### Что используется в Docker-сервисе (текущее решение)

| Компонент | Описание |
|-----------|----------|
| **Признаки** | **Те же 41 признак** — логика `add_features`, частот, TE и `sum_qbin_ord` перенесена из ноутбука |
| **Модель** | Только **HistGradientBoostingRegressor** с Optuna-гиперпараметрами из ноутбука |
| **Мета-уровень** | Нет (одна модель) |
| **Обучение** | Один раз при `docker build` на полном `train.csv`; в runtime — только inference |
| **TE на inference** | Карты TE считаются по всему train (корректно для продакшн-inference, без OOF) |
| **Частоты на inference** | Train-частоты + текущий test-батч (как train+test в ноутбуке) |
| **Цель** | Стабильный Docker-сервис для сдачи MLOps ДЗ (зачёт на 4/5) |





### Цепочка сборки Docker (логика)

```
docker build
  → pip install -r requirements.txt
  → COPY исходников + train_data/train.csv
  → python scripts/build_model.py
       → fit_artifacts(train)        # артефакты препроцессинга
       → run_preproc(train без target)
       → fit HistGradientBoostingRegressor
       → save models/model_bundle.joblib
  → CMD python app/app.py

docker run + test.csv в ./input
  → watchdog on_created
  → read_csv → run_preproc → make_pred → save CSV/JSON/PNG
```

**Важно для сборки:** файл `train_data/train.csv` должен быть на месте **до** `docker build` (в GitHub его нет - пожалуйста, добавьте его из кэггл моревноания).

