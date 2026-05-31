# Teta ML 2026 — ML Inference Service (MLOps HW)

Сервис для **batch-скоринга** модели из соревнования [teta-ml-2026](https://www.kaggle.com/competitions/teta-ml-2026) (Kaggle).  
Шаблон архитектуры взят с практики MLOps: [mts25_mlops_hw1_fraud_detector](https://github.com/NikitaMalykhin/mts25_mlops_hw1_fraud_detector).

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

### Локальный запуск без Docker (опционально)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/build_model.py
python app/app.py
```

> Для локального запуска замените пути `/app/input`, `/app/output`, `/app/logs` в `app/app.py` на `./input`, `./output`, `./logs` или создайте симлинки.

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

> **Почему упрощено:** задание оценивает навыки Docker и упаковки ML-сервиса, а не качество Kaggle-модели. Один HGBR быстрее собирается (~2–5 мин vs 10–30+ мин у ансамбля), проще поддерживать и достаточен для демонстрации inference.

### Технические детали inference

- Предсказания обрезаются к `[0, 365]` (ограничение соревнования)
- Feature importances — из `model.feature_importances_` (топ-5 в JSON)
- График плотности — KDE/hist по предсказанным скорам (PNG)

Обучение — один раз при `docker build`, в runtime только inference.

---

## Соответствие критериям задания

| Критерий | Статус |
|----------|--------|
| `test.csv` через `./input` | ✅ Watchdog в `app/app.py` |
| `sample_submission.csv` в `./output` | ✅ Колонки `index`, `prediction` |
| JSON top-5 feature importances | ✅ `feature_importance_top5_*.json` |
| PNG график плотности скоров | ✅ `score_density_*.png` |
| Препроцессинг отдельным скриптом | ✅ `src/preprocessing.py` |
| Скоринг отдельным скриптом | ✅ `src/scorer.py` |
| Модель реально применяется | ✅ HGBR из `model_bundle.joblib` |
| Inference-only в контейнере | ✅ Обучение только при `docker build` |
| CPU inference | ✅ scikit-learn, без GPU |
| README + requirements + Dockerfile | ✅ |

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

**Важно для сборки:** файл `train_data/train.csv` должен быть на месте **до** `docker build` (в GitHub его нет — проверяющий добавляет сам).

---

## Как выложить проект на GitHub

### Шаг 1. Создайте аккаунт и репозиторий

1. Зарегистрируйтесь на [github.com](https://github.com), если аккаунта ещё нет.
2. Нажмите **+ → New repository**.
3. Имя, например: `teta-ml-2026-mlops-hw`.
4. Выберите **Public** (репозиторий должен быть публичным для проверки).
5. **Не** добавляйте README, .gitignore и license — они уже есть в проекте.
6. Нажмите **Create repository**.

### Шаг 2. Установите Git (если не установлен)

- Windows: [https://git-scm.com/download/win](https://git-scm.com/download/win)
- После установки откройте **Git Bash** или PowerShell.

### Шаг 3. Инициализируйте git в папке проекта

```bash
cd путь/к/teta_mlops_service

git init
git add .
git commit -m "MLOps HW: teta-ml-2026 inference service"
```

> `train.csv`, `test.csv` и `models/*.joblib` в `.gitignore` не попадут в репозиторий — это нормально. Проверяющий положит `train.csv` в `train_data/` перед `docker build`.

### Шаг 4. Привяжите удалённый репозиторий и отправьте код

На странице созданного репозитория GitHub скопируйте URL (HTTPS), затем:

```bash
git branch -M main
git remote add origin https://github.com/ВАШ_ЛОГИН/teta-ml-2026-mlops-hw.git
git push -u origin main
```

GitHub попросит авторизацию (логин + Personal Access Token или браузер).

### Шаг 5. Проверка перед сдачей

- Репозиторий **Public**
- В README есть инструкция по `docker build` / `docker run`
- После `git clone` + `train.csv` в `train_data/` образ собирается без ошибок
- Сервис принимает `test.csv` и выдаёт `sample_submission`, JSON с FI и PNG

### Что отправить преподавателю

Ссылку на публичный репозиторий, например:

```
https://github.com/ВАШ_ЛОГИН/teta-ml-2026-mlops-hw
```

---

## Disclaimer

Учебный проект для курса MLOps. Датасет — [teta-ml-2026](https://www.kaggle.com/competitions/teta-ml-2026).
