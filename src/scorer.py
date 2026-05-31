"""Скоринг и формирование дополнительных артефактов (grade 5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from constants import NUM_COLS

logger = logging.getLogger(__name__)

MODEL_PATH = Path("./models/model_bundle.joblib")
_bundle = None


def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        logger.info("Loading model bundle from %s", MODEL_PATH)
        _bundle = joblib.load(MODEL_PATH)
        logger.info("Model bundle loaded")
    return _bundle


def make_pred(features_df: pd.DataFrame, source_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    bundle = _load_bundle()
    model = bundle["model"]
    scores = model.predict(features_df[NUM_COLS].values)
    scores = np.clip(scores, 0.0, 365.0)

    raw = pd.read_csv(source_path)
    index_col = "index" if "index" in raw.columns else raw.index
    submission = pd.DataFrame({"index": index_col, "prediction": scores})
    logger.info("Prediction complete for %s rows", len(submission))
    return submission, scores


def save_feature_importance_top5(output_path: str) -> None:
    bundle = _load_bundle()
    model = bundle["model"]
    importances = model.feature_importances_
    pairs = sorted(zip(NUM_COLS, importances), key=lambda x: x[1], reverse=True)[:5]
    payload = {name: float(value) for name, value in pairs}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Top-5 feature importances saved to %s", output_path)


def save_score_density_plot(scores: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    if len(scores) > 1 and np.std(scores) > 0:
        xs = np.linspace(scores.min(), scores.max(), 200)
        kde = gaussian_kde(scores)
        plt.plot(xs, kde(xs), color="#2563eb", linewidth=2)
        plt.fill_between(xs, kde(xs), alpha=0.25, color="#2563eb")
    else:
        plt.hist(scores, bins=20, density=True, color="#2563eb", alpha=0.7)

    plt.title("Распределение предсказанных скоров")
    plt.xlabel("prediction")
    plt.ylabel("density")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    logger.info("Score density plot saved to %s", output_path)
