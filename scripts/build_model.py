"""Обучение модели и сохранение артефактов перед сборкой Docker-образа."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from constants import HGB_PARAMS, NUM_COLS, RANDOM_STATE
from preprocessing import fit_artifacts, load_train_data, run_preproc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    train_path = ROOT / "train_data" / "train.csv"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "model_bundle.joblib"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Не найден {train_path}. Скачайте train.csv из соревнования и положите в train_data/."
        )

    train_df = load_train_data(str(train_path))
    artifacts = fit_artifacts(train_df)

    features = run_preproc(
        train_df.drop(columns=["target"]),
        artifacts,
        update_freq_with_batch=False,
    )
    y = train_df["target"].values.astype(float)

    model = HistGradientBoostingRegressor(
        **HGB_PARAMS,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        random_state=RANDOM_STATE,
    )
    logger.info("Training HistGradientBoostingRegressor on %s samples...", len(features))
    model.fit(features[NUM_COLS].values, y)

    bundle = {
        "model": model,
        "artifacts": artifacts,
        "feature_names": NUM_COLS,
    }
    joblib.dump(bundle, out_path)
    logger.info("Saved model bundle to %s", out_path)


if __name__ == "__main__":
    main()
