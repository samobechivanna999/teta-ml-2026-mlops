"""Константы и утилиты feature engineering для teta-ml-2026."""

from __future__ import annotations

RANDOM_STATE = 42
TE_SMOOTH_M = 20.0

TE_GROUP_COLS = ["location", "host_name", "type_house", "location_cluster"]
COMBO_TE_COLS = ["loc_type", "host_cluster", "geo_bin_20"]
FREQ_COLS = ["location", "host_name", "type_house", "location_cluster", "loc_type", "host_cluster"]

NUM_COLS = [
    "lat",
    "lon",
    "sum",
    "min_days",
    "amt_reviews",
    "avg_reviews_filled",
    "total_host",
    "days_since_activity",
    "has_last_dt",
    "year",
    "month",
    "dow",
    "amt_reviews_log",
    "total_host_log",
    "sum_per_minday",
    "name_len",
    "reviews_x_sum",
    "dist_center_sq",
    "sum_log",
    "min_days_log",
    "amt_reviews_sqrt",
    "no_reviews",
    "name_words",
    "name_has_private",
    "name_has_shared",
    "name_has_lux",
    "cnt_location",
    "cnt_host_name",
    "cnt_type_house",
    "cnt_location_cluster",
    "cnt_loc_type",
    "cnt_host_cluster",
    "cnt_geo_bin_100",
    "te_location",
    "te_host_name",
    "te_type_house",
    "te_location_cluster",
    "te_loc_type",
    "te_host_cluster",
    "te_geo_bin_20",
    "sum_qbin_ord",
]

HGB_PARAMS = {
    "max_iter": 494,
    "learning_rate": 0.03622630037371023,
    "max_depth": 13,
    "max_leaf_nodes": 122,
    "l2_regularization": 11.405466032994426,
    "min_samples_leaf": 34,
}
