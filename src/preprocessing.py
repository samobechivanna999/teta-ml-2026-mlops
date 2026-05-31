"""Пайплайн препроцессинга данных teta-ml-2026."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constants import COMBO_TE_COLS, FREQ_COLS, NUM_COLS, TE_GROUP_COLS, TE_SMOOTH_M

logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    d = df.copy()
    d["host_name"] = d["host_name"].fillna("unknown")
    d["name"] = d["name"].fillna("")
    d["location"] = d["location"].fillna("unknown")
    d["last_dt_parsed"] = pd.to_datetime(d["last_dt"], errors="coerce")
    d["days_since_activity"] = (ref_date - d["last_dt_parsed"]).dt.days.fillna(9999)
    d["has_last_dt"] = d["last_dt_parsed"].notna().astype(np.int8)
    d["year"] = d["last_dt_parsed"].dt.year.fillna(0).astype(np.int32)
    d["month"] = d["last_dt_parsed"].dt.month.fillna(0).astype(np.int32)
    d["dow"] = d["last_dt_parsed"].dt.dayofweek.fillna(-1).astype(np.int32)
    d["avg_reviews_filled"] = d["avg_reviews"].fillna(0.0)
    d["amt_reviews_log"] = np.log1p(d["amt_reviews"].astype(float))
    d["total_host_log"] = np.log1p(d["total_host"].astype(float))
    md = d["min_days"].replace(0, np.nan)
    d["sum_per_minday"] = (d["sum"] / md).fillna(d["sum"])
    d["name_len"] = d["name"].astype(str).str.len().astype(np.int32)
    d["reviews_x_sum"] = d["amt_reviews"].astype(float) * np.log1p(d["sum"].astype(float))
    la = d["lat"].astype(float)
    lo = d["lon"].astype(float)
    d["dist_center_sq"] = (la - 40.7589) ** 2 + (lo - (-73.9851)) ** 2
    d["sum_log"] = np.log1p(d["sum"].clip(lower=0).astype(float))
    d["min_days_log"] = np.log1p(d["min_days"].clip(lower=0).astype(float))
    d["amt_reviews_sqrt"] = np.sqrt(d["amt_reviews"].astype(float))
    d["no_reviews"] = (d["amt_reviews"].astype(int) == 0).astype(np.int8)

    s = d["name"].astype(str)
    d["name_words"] = s.str.split().str.len().fillna(0).astype(np.int32)
    d["name_has_private"] = s.str.contains("private", case=False, na=False).astype(np.int8)
    d["name_has_shared"] = s.str.contains("shared", case=False, na=False).astype(np.int8)
    d["name_has_lux"] = s.str.contains("lux|luxury", case=False, na=False, regex=True).astype(np.int8)
    return d


def _sum_to_qbin(series: pd.Series, sum_bins: np.ndarray) -> pd.Series:
    cut = pd.cut(series.astype(float), bins=sum_bins, include_lowest=True)
    out = cut.astype(object)
    return out.where(pd.notna(out), "edge").astype(str)


def _add_combo_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["loc_type"] = d["location"].astype(str) + "||" + d["type_house"].astype(str)
    d["host_cluster"] = d["host_name"].astype(str) + "||" + d["location_cluster"].astype(str)
    d["geo_bin_100"] = (
        (d["lat"].astype(float) * 100).round().astype(int).astype(str)
        + "|"
        + (d["lon"].astype(float) * 100).round().astype(int).astype(str)
    )
    d["geo_bin_20"] = (
        (d["lat"].astype(float) * 20).round().astype(int).astype(str)
        + "|"
        + (d["lon"].astype(float) * 20).round().astype(int).astype(str)
    )
    return d


def _apply_freq_features(df: pd.DataFrame, freq_maps: dict[str, dict[str, float]]) -> pd.DataFrame:
    d = df.copy()
    for col in FREQ_COLS:
        mapping = freq_maps[col]
        d[f"cnt_{col}"] = d[col].astype(str).map(mapping).fillna(1).astype(np.float64)
    d["cnt_geo_bin_100"] = (
        d["geo_bin_100"].astype(str).map(freq_maps["geo_bin_100"]).fillna(1).astype(np.float64)
    )
    return d


def _apply_te_features(
    df: pd.DataFrame,
    te_maps: dict[str, dict[str, float]],
    global_mean: float,
) -> pd.DataFrame:
    d = df.copy()
    for col, mapping in te_maps.items():
        d[f"te_{col}"] = d[col].astype(str).map(mapping).fillna(global_mean).astype(np.float64)
    return d


def fit_artifacts(train_df: pd.DataFrame) -> dict[str, Any]:
    """Строит артефакты препроцессинга по train.csv."""
    logger.info("Fitting preprocessing artifacts on train data...")
    train = train_df.copy()
    train["last_dt_parsed"] = pd.to_datetime(train["last_dt"], errors="coerce")
    ref_date = train["last_dt_parsed"].max() + pd.Timedelta(days=1)

    train_fe = add_features(train, ref_date)
    y = train_fe["target"].values.astype(np.float64)
    global_mean = float(np.mean(y))

    _, sum_bins = pd.qcut(train_fe["sum"], q=10, duplicates="drop", retbins=True)
    train_fe["sum_qbin"] = _sum_to_qbin(train_fe["sum"], sum_bins)

    le_sq = LabelEncoder()
    classes = train_fe["sum_qbin"].astype(str).unique().tolist()
    if "edge" not in classes:
        classes.append("edge")
    le_sq.fit(classes)

    train_fe = _add_combo_columns(train_fe)

    freq_maps: dict[str, dict[str, float]] = {}
    for col in FREQ_COLS:
        vc = train_fe[col].astype(str).value_counts()
        freq_maps[col] = vc.to_dict()
    freq_maps["geo_bin_100"] = train_fe["geo_bin_100"].astype(str).value_counts().to_dict()

    def smoothed_map(series: pd.Series, y_values: np.ndarray, col_name: str) -> dict[str, float]:
        grouped = (
            pd.DataFrame({col_name: series.values, "_y": y_values})
            .groupby(col_name, observed=False)["_y"]
            .agg(["sum", "count"])
        )
        encoded = (grouped["sum"] + TE_SMOOTH_M * global_mean) / (grouped["count"] + TE_SMOOTH_M)
        return encoded.to_dict()

    te_maps: dict[str, dict[str, float]] = {}
    for col in TE_GROUP_COLS + COMBO_TE_COLS:
        te_maps[col] = smoothed_map(train_fe[col], y, col)

    return {
        "ref_date": ref_date.isoformat(),
        "sum_bins": sum_bins.tolist(),
        "sum_qbin_classes": le_sq.classes_.tolist(),
        "freq_maps": freq_maps,
        "te_maps": te_maps,
        "global_mean_target": global_mean,
    }


def run_preproc(
    test_df: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    update_freq_with_batch: bool = True,
) -> pd.DataFrame:
    """Преобразует test.csv в матрицу признаков для модели."""
    logger.info("Starting preprocessing for %s rows", len(test_df))
    ref_date = pd.Timestamp(artifacts["ref_date"])
    sum_bins = np.array(artifacts["sum_bins"], dtype=float)

    test_fe = add_features(test_df, ref_date)
    test_fe = _add_combo_columns(test_fe)

    le_sq = LabelEncoder()
    le_sq.classes_ = np.array(artifacts["sum_qbin_classes"], dtype=object)
    test_fe["sum_qbin"] = _sum_to_qbin(test_fe["sum"], sum_bins)
    known = set(le_sq.classes_)
    test_fe["sum_qbin"] = test_fe["sum_qbin"].apply(lambda x: x if x in known else "edge")
    test_fe["sum_qbin_ord"] = le_sq.transform(test_fe["sum_qbin"].astype(str)).astype(np.float64)

    freq_maps = {k: dict(v) for k, v in artifacts["freq_maps"].items()}
    if update_freq_with_batch:
        for col in FREQ_COLS:
            test_counts = test_fe[col].astype(str).value_counts()
            merged = pd.Series(freq_maps[col], dtype=float).add(test_counts, fill_value=0)
            freq_maps[col] = merged.to_dict()
        geo_test = test_fe["geo_bin_100"].astype(str).value_counts()
        merged_geo = pd.Series(freq_maps["geo_bin_100"], dtype=float).add(geo_test, fill_value=0)
        freq_maps["geo_bin_100"] = merged_geo.to_dict()

    test_fe = _apply_freq_features(test_fe, freq_maps)
    test_fe = _apply_te_features(test_fe, artifacts["te_maps"], artifacts["global_mean_target"])

    matrix = np.nan_to_num(
        test_fe[NUM_COLS].to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    logger.info("Preprocessing finished. Feature matrix shape: %s", matrix.shape)
    return pd.DataFrame(matrix, columns=NUM_COLS)


def load_train_data(train_path: str = "./train_data/train.csv") -> pd.DataFrame:
    logger.info("Loading train data from %s", train_path)
    return pd.read_csv(train_path)
