#!/usr/bin/env python3
"""
V4.1: 3-Sigma baseline + gateway-relative Isolation Forest.

V4.1 keeps the V4 feature engineering and model design, but corrects
the ML historical window.

Historical behavior:
    28 complete days immediately before the recent 7-day window.

Recent scoring behavior:
    7 days immediately before each scored Monday.

Therefore, for a scoring Monday:

    historical window = Monday - 35 days  ->  Monday - 7 days
    recent window     = Monday - 7 days   ->  Monday

The original 3-sigma method is preserved as the benchmark.

V4/V4.1 improvements:
    - loads all monthly telemetry parquet files
    - removes exact duplicate gateway/timestamp rows
    - uses connectivity, reboot, radio, communication and
      system-health telemetry
    - derives gateway-relative ratio features
    - compares recent behavior against each gateway's own history
    - includes history-quality features
    - handles short-history gateways safely
    - uses only telemetry strictly before the scored week
    - ranks gateways using Isolation Forest

Usage:

    python baseline_3sigma.py --data path/to/data --out predictions_ml_v4_1.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ============================================================
# CONFIGURATION
# ============================================================

SCORED_WEEKS = [
    dt.date(2026, 2, 2) + dt.timedelta(days=7 * i)
    for i in range(8)
]

VISITS_PER_WEEK = 15

# ML historical window:
# 28 complete days BEFORE the recent 7-day scoring window.
#
# Therefore:
#
# history_start = Monday - (28 + 7) days = Monday - 35 days
# history_end   = Monday - 7 days
#
BASELINE_DAYS = 28
RECENT_DAYS = 7
HISTORY_GAP_DAYS = RECENT_DAYS

SIGMA = 3.0


# ============================================================
# TELEMETRY FEATURE GROUPS
# ============================================================

# Original 3-sigma benchmark metrics.
CORE_METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]


# Additional telemetry features used by V4.1.
OPTIONAL_METRICS = [
    "avg_offline_duration",
    "online_duration_mins",
    "no_conn_importance",
    "reboot_duration_sec",
    "avg_reboot_duration",
    "reboot_importance",
    "r_cnt_power_cycle",
    "r_cnt_reboot",
    "r_cnt_unknown",
    "r_dur_power_cycle",
    "r_dur_reboot",
    "r_dur_unknown",
    "avg_load1",
    "avg_memfree",
    "avg_idletime",
]


# Derived radio/network/communication features.
RATIO_FEATURES = [
    "rx_crc_error_ratio",
    "tx_busy_ratio",
    "network_unknown_ratio",
    "rssi_bad_ratio",
    "rscp_rsrp_bad_ratio",
    "ecio_rsrq_bad_ratio",
]


ML_METRICS = CORE_METRICS + OPTIONAL_METRICS + RATIO_FEATURES


# Raw columns required to calculate the derived ratios.
RATIO_SOURCE_COLUMNS = [
    "rx_nr_pkts",
    "rx_crc_bad",
    "tx_success",
    "tx_busy",
    "network_2g",
    "network_3g",
    "network_4g",
    "network_unknown",
    "rssi_good",
    "rssi_normal",
    "rssi_bad",
    "rscp_rsrp_good",
    "rscp_rsrp_normal",
    "rscp_rsrp_bad",
    "ecio_rsrq_good",
    "ecio_rsrq_normal",
    "ecio_rsrq_bad",
]


# ============================================================
# DATA LOADING
# ============================================================

def load(data_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load all monthly telemetry parquet files.

    Only month=*/part-0.parquet files are loaded.

    Exact duplicate gateway/timestamp rows are removed because
    the telemetry inspection showed duplicate groups with identical
    values.
    """

    telemetry_dir = data_dir / "telemetry"

    if not telemetry_dir.exists():
        raise SystemExit(
            f"Telemetry directory not found: {telemetry_dir}"
        )

    parquet_files = sorted(
        telemetry_dir.glob("month=*/part-0.parquet")
    )

    if not parquet_files:
        raise SystemExit(
            f"No telemetry parquet files found in {telemetry_dir}"
        )

    required_columns = [
        "gateway_id",
        "ts_utc",
        *CORE_METRICS,
        *OPTIONAL_METRICS,
        *RATIO_SOURCE_COLUMNS,
    ]

    frames = []

    for parquet_file in parquet_files:
        print(f"loading {parquet_file}")

        frame = pd.read_parquet(
            parquet_file,
            columns=required_columns,
        )

        frames.append(frame)

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print(f"loaded {len(frame):,} telemetry rows")

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    frame["ts"] = pd.to_datetime(
        frame["ts_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = frame["ts"].isna().sum()

    if invalid_timestamps:
        print(
            f"removed {invalid_timestamps:,} rows "
            "with invalid timestamps"
        )

        frame = frame.dropna(
            subset=["ts"]
        ).copy()

    frame = frame.drop(
        columns=["ts_utc"]
    )

    # --------------------------------------------------------
    # Remove exact duplicate gateway/timestamp rows
    # --------------------------------------------------------

    duplicate_mask = frame.duplicated(
        subset=["gateway_id", "ts"],
        keep="first",
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count:
        frame = frame.loc[
            ~duplicate_mask
        ].copy()

        print(
            f"removed {duplicate_count:,} exact duplicate "
            "gateway/timestamp rows"
        )
    else:
        print(
            "removed 0 duplicate gateway/timestamp rows"
        )

    frame = frame.sort_values(
        ["gateway_id", "ts"]
    ).reset_index(drop=True)

    print(
        f"unique telemetry rows: {len(frame):,}"
    )

    print(
        f"gateways: {frame['gateway_id'].nunique()}"
    )

    print(
        f"date range: {frame['ts'].min()} "
        f"to {frame['ts'].max()}"
    )

    return frame


# ============================================================
# ORIGINAL 3-SIGMA BASELINE
# ============================================================

def rank_week(
    frame: pd.DataFrame,
    monday: dt.date,
) -> pd.DataFrame:
    """
    Original 3-sigma anomaly ranking.

    This function is intentionally preserved as the benchmark.

    It uses:
        - 28 days immediately before Monday
        - recent 7 days inside that 28-day window
        - three original metrics
    """

    end = pd.Timestamp(
        monday,
        tz="UTC",
    )

    # --------------------------------------------------------
    # Original 28-day benchmark window
    # --------------------------------------------------------

    window = frame[
        (frame["ts"] >= end - dt.timedelta(days=BASELINE_DAYS))
        & (frame["ts"] < end)
    ].copy()

    if window.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "flagged_hours",
                "worst_metric",
            ]
        )

    # --------------------------------------------------------
    # Gateway-specific historical statistics
    # --------------------------------------------------------

    stats = (
        window
        .groupby("gateway_id")[CORE_METRICS]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    # --------------------------------------------------------
    # Recent 7-day portion
    # --------------------------------------------------------

    recent = window[
        window["ts"] >= end - dt.timedelta(days=RECENT_DAYS)
    ].copy()

    flags = pd.Series(
        0,
        index=recent.index,
        dtype=int,
    )

    worst = pd.Series(
        "",
        index=recent.index,
        dtype=object,
    )

    # --------------------------------------------------------
    # Check each original metric
    # --------------------------------------------------------

    for metric in CORE_METRICS:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = (
            recent["gateway_id"]
            .map(stats[(metric, "std")])
            .replace(0, np.nan)
        )

        exceeded = (
            recent[metric] - mean
        ) > SIGMA * std

        exceeded = exceeded.fillna(False)

        flags = (
            flags
            + exceeded.astype(int)
        )

        worst = worst.where(
            ~exceeded | (worst != ""),
            metric,
        )

    recent["flagged"] = flags
    recent["worst_metric"] = worst

    grouped = (
        recent
        .groupby("gateway_id")
        .agg(
            flagged_hours=(
                "flagged",
                "sum",
            ),
            worst_metric=(
                "worst_metric",
                lambda s: next(
                    (
                        value
                        for value in s
                        if value
                    ),
                    "",
                ),
            ),
        )
    )

    return (
        grouped
        .sort_values(
            "flagged_hours",
            ascending=False,
        )
        .reset_index()
    )


# ============================================================
# DERIVED RATIO FEATURES
# ============================================================

def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Calculate numerator / denominator safely.

    Zero denominators become NaN and are subsequently filled
    during model preparation.
    """

    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


def add_ratio_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add communication, network and radio-quality ratios.
    """

    frame = frame.copy()

    # --------------------------------------------------------
    # RX CRC error ratio
    # --------------------------------------------------------

    frame["rx_crc_error_ratio"] = safe_ratio(
        frame["rx_crc_bad"],
        frame["rx_crc_bad"] + frame["rx_nr_pkts"],
    )

    # --------------------------------------------------------
    # TX busy ratio
    # --------------------------------------------------------

    frame["tx_busy_ratio"] = safe_ratio(
        frame["tx_busy"],
        frame["tx_success"] + frame["tx_busy"],
    )

    # --------------------------------------------------------
    # Network unknown ratio
    # --------------------------------------------------------

    frame["network_unknown_ratio"] = safe_ratio(
        frame["network_unknown"],
        (
            frame["network_2g"]
            + frame["network_3g"]
            + frame["network_4g"]
            + frame["network_unknown"]
        ),
    )

    # --------------------------------------------------------
    # RSSI bad ratio
    # --------------------------------------------------------

    frame["rssi_bad_ratio"] = safe_ratio(
        frame["rssi_bad"],
        (
            frame["rssi_good"]
            + frame["rssi_normal"]
            + frame["rssi_bad"]
        ),
    )

    # --------------------------------------------------------
    # RSCP/RSRP bad ratio
    # --------------------------------------------------------

    frame["rscp_rsrp_bad_ratio"] = safe_ratio(
        frame["rscp_rsrp_bad"],
        (
            frame["rscp_rsrp_good"]
            + frame["rscp_rsrp_normal"]
            + frame["rscp_rsrp_bad"]
        ),
    )

    # --------------------------------------------------------
    # ECIO/RSRQ bad ratio
    # --------------------------------------------------------

    frame["ecio_rsrq_bad_ratio"] = safe_ratio(
        frame["ecio_rsrq_bad"],
        (
            frame["ecio_rsrq_good"]
            + frame["ecio_rsrq_normal"]
            + frame["ecio_rsrq_bad"]
        ),
    )

    return frame


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_ml_features(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create gateway-relative anomaly features.

    Historical behavior:
        28 complete days immediately before the recent window.

    Recent behavior:
        7 days.

    For each telemetry metric, features include:

        - recent mean
        - recent maximum
        - recent standard deviation
        - recent sum
        - historical mean
        - historical standard deviation
        - historical maximum
        - historical median
        - deviation from historical mean
        - gateway-relative z-score
        - recent/historical ratio
        - maximum deviation
        - maximum z-score

    Additional history-quality features are included so that
    short-history gateways are represented explicitly rather
    than being automatically treated as anomalous.
    """

    baseline = baseline.copy()
    recent = recent.copy()

    # ========================================================
    # Historical statistics
    # ========================================================

    stats = (
        baseline
        .groupby("gateway_id")[ML_METRICS]
        .agg(
            [
                "mean",
                "std",
                "max",
                "median",
            ]
        )
    )

    # ========================================================
    # Recent statistics
    # ========================================================

    recent_agg = (
        recent
        .groupby("gateway_id")[ML_METRICS]
        .agg(
            [
                "mean",
                "max",
                "std",
                "sum",
            ]
        )
    )

    # ========================================================
    # History-quality information
    # ========================================================

    baseline_counts = (
        baseline
        .groupby("gateway_id")
        .size()
        .rename("history_hours")
    )

    recent_counts = (
        recent
        .groupby("gateway_id")
        .size()
        .rename("recent_hours")
    )

    baseline_days = (
        baseline
        .assign(
            day=baseline["ts"].dt.floor("D")
        )
        .groupby("gateway_id")["day"]
        .nunique()
        .rename("history_days")
    )

    history_quality = pd.concat(
        [
            baseline_counts,
            recent_counts,
            baseline_days,
        ],
        axis=1,
    )

    history_quality["history_hours"] = (
        history_quality["history_hours"]
        .fillna(0)
        .astype(float)
    )

    history_quality["recent_hours"] = (
        history_quality["recent_hours"]
        .fillna(0)
        .astype(float)
    )

    history_quality["history_days"] = (
        history_quality["history_days"]
        .fillna(0)
        .astype(float)
    )

    # Expected number of hours in 28 historical days.
    expected_history_hours = (
        BASELINE_DAYS * 24
    )

    history_quality["history_coverage"] = (
        history_quality["history_hours"]
        / expected_history_hours
    ).clip(
        lower=0,
        upper=1,
    )

    # ========================================================
    # Build one feature row per gateway
    # ========================================================

    rows = []

    for gateway_id in recent_agg.index:

        feature_row = {
            "gateway_id": gateway_id,
        }

        # ----------------------------------------------------
        # History quality
        # ----------------------------------------------------

        if gateway_id in history_quality.index:

            quality = history_quality.loc[
                gateway_id
            ]

            feature_row["history_days"] = float(
                quality["history_days"]
            )

            feature_row["history_hours"] = float(
                quality["history_hours"]
            )

            feature_row["recent_hours"] = float(
                quality["recent_hours"]
            )

            feature_row["history_coverage"] = float(
                quality["history_coverage"]
            )

        else:

            feature_row["history_days"] = 0.0
            feature_row["history_hours"] = 0.0
            feature_row["recent_hours"] = 0.0
            feature_row["history_coverage"] = 0.0

        # ----------------------------------------------------
        # Metric-level features
        # ----------------------------------------------------

        for metric in ML_METRICS:

            # =================================================
            # Historical values
            # =================================================

            if gateway_id in stats.index:

                hist_mean = stats.loc[
                    gateway_id,
                    (metric, "mean"),
                ]

                hist_std = stats.loc[
                    gateway_id,
                    (metric, "std"),
                ]

                hist_max = stats.loc[
                    gateway_id,
                    (metric, "max"),
                ]

                hist_median = stats.loc[
                    gateway_id,
                    (metric, "median"),
                ]

            else:

                hist_mean = np.nan
                hist_std = np.nan
                hist_max = np.nan
                hist_median = np.nan

            # =================================================
            # Recent values
            # =================================================

            recent_mean = recent_agg.loc[
                gateway_id,
                (metric, "mean"),
            ]

            recent_max = recent_agg.loc[
                gateway_id,
                (metric, "max"),
            ]

            recent_std = recent_agg.loc[
                gateway_id,
                (metric, "std"),
            ]

            recent_sum = recent_agg.loc[
                gateway_id,
                (metric, "sum"),
            ]

            # =================================================
            # Safe statistics
            # =================================================

            safe_hist_mean = (
                hist_mean
                if pd.notna(hist_mean)
                else 0.0
            )

            safe_hist_std = (
                hist_std
                if pd.notna(hist_std)
                and hist_std > 0
                else 1.0
            )

            safe_hist_max = (
                hist_max
                if pd.notna(hist_max)
                else 0.0
            )

            safe_hist_median = (
                hist_median
                if pd.notna(hist_median)
                else 0.0
            )

            safe_mean_for_ratio = (
                hist_mean
                if pd.notna(hist_mean)
                and abs(hist_mean) > 1e-9
                else 1.0
            )

            safe_recent_std = (
                recent_std
                if pd.notna(recent_std)
                else 0.0
            )

            # =================================================
            # Raw recent behavior
            # =================================================

            feature_row[
                f"{metric}_recent_mean"
            ] = recent_mean

            feature_row[
                f"{metric}_recent_max"
            ] = recent_max

            feature_row[
                f"{metric}_recent_std"
            ] = safe_recent_std

            feature_row[
                f"{metric}_recent_sum"
            ] = recent_sum

            # =================================================
            # Historical behavior
            # =================================================

            feature_row[
                f"{metric}_hist_mean"
            ] = safe_hist_mean

            feature_row[
                f"{metric}_hist_std"
            ] = hist_std if pd.notna(
                hist_std
            ) else 0.0

            feature_row[
                f"{metric}_hist_max"
            ] = safe_hist_max

            feature_row[
                f"{metric}_hist_median"
            ] = safe_hist_median

            # =================================================
            # Deviation
            # =================================================

            deviation = (
                recent_mean
                - safe_hist_mean
            )

            feature_row[
                f"{metric}_deviation"
            ] = deviation

            # =================================================
            # Gateway-relative z-score
            # =================================================

            z_score = (
                deviation
                / safe_hist_std
            )

            feature_row[
                f"{metric}_zscore"
            ] = z_score

            # =================================================
            # Recent / historical ratio
            # =================================================

            ratio = (
                recent_mean
                / safe_mean_for_ratio
            )

            feature_row[
                f"{metric}_ratio"
            ] = ratio

            # =================================================
            # Maximum deviation
            # =================================================

            max_deviation = (
                recent_max
                - safe_hist_mean
            )

            feature_row[
                f"{metric}_max_deviation"
            ] = max_deviation

            # =================================================
            # Maximum z-score
            # =================================================

            max_zscore = (
                max_deviation
                / safe_hist_std
            )

            feature_row[
                f"{metric}_max_zscore"
            ] = max_zscore

        rows.append(feature_row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# ISOLATION FOREST
# ============================================================

def ml_rank_week(
    frame: pd.DataFrame,
    monday: dt.date,
) -> pd.DataFrame:
    """
    Rank gateways using feature-engineered Isolation Forest.

    V4.1 historical window:
        28 complete days before the recent 7-day window.

    V4.1 recent window:
        7 days immediately before Monday.

    Example for Monday 2026-02-02:

        Historical:
            2025-12-29 00:00
            through
            2026-01-25 23:00

        Recent:
            2026-01-26 00:00
            through
            2026-02-01 23:00

    No data on or after the scoring Monday is used.
    """

    end = pd.Timestamp(
        monday,
        tz="UTC",
    )

    # ========================================================
    # CORRECTED V4.1 HISTORICAL WINDOW
    # ========================================================

    # We need:
    #
    # 28 complete historical days
    # immediately before
    # 7 recent scoring days.
    #
    # Therefore the history starts 35 days before Monday.

    history_start = (
        end
        - dt.timedelta(
            days=BASELINE_DAYS + HISTORY_GAP_DAYS
        )
    )

    history_end = (
        end
        - dt.timedelta(
            days=RECENT_DAYS
        )
    )

    baseline = frame[
        (frame["ts"] >= history_start)
        & (frame["ts"] < history_end)
    ].copy()

    # ========================================================
    # RECENT 7-DAY SCORING WINDOW
    # ========================================================

    recent_start = (
        end
        - dt.timedelta(
            days=RECENT_DAYS
        )
    )

    recent = frame[
        (frame["ts"] >= recent_start)
        & (frame["ts"] < end)
    ].copy()

    if baseline.empty or recent.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "anomaly_score",
            ]
        )

    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    baseline = add_ratio_features(
        baseline
    )

    recent = add_ratio_features(
        recent
    )

    # ========================================================
    # CREATE GATEWAY-RELATIVE FEATURES
    # ========================================================

    features = create_ml_features(
        baseline,
        recent,
    )

    if features.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "anomaly_score",
            ]
        )

    # ========================================================
    # SEPARATE GATEWAY IDS
    # ========================================================

    gateway_ids = features[
        "gateway_id"
    ].copy()

    X = features.drop(
        columns=["gateway_id"]
    ).copy()

    # ========================================================
    # CLEAN NUMERICAL FEATURES
    # ========================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    # Make sure every feature is numeric.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    model = IsolationForest(
        n_estimators=400,
        contamination="auto",
        max_samples="auto",
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X)

    # sklearn's score_samples:
    #     higher = more normal
    #
    # Negating it gives:
    #     higher = more anomalous

    anomaly_scores = (
        -model.score_samples(X)
    )

    # ========================================================
    # RANK
    # ========================================================

    ranked = pd.DataFrame(
        {
            "gateway_id": gateway_ids,
            "anomaly_score": anomaly_scores,
        }
    )

    ranked = (
        ranked
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return ranked


# ============================================================
# BUILD PREDICTIONS
# ============================================================

def build_ml_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate ML predictions for all eight scored weeks.

    Exactly 15 gateways are selected for each week.
    """

    rows = []

    for monday in SCORED_WEEKS:

        print(
            f"processing week {monday}..."
        )

        ranked = ml_rank_week(
            frame,
            monday,
        )

        if len(ranked) < VISITS_PER_WEEK:
            raise SystemExit(
                f"only {len(ranked)} gateways have "
                f"enough data before {monday}"
            )

        top_gateways = ranked.head(
            VISITS_PER_WEEK
        )

        for rank, row in enumerate(
            top_gateways.itertuples(
                index=False
            ),
            1,
        ):

            rows.append(
                {
                    "week_start": monday.isoformat(),
                    "rank": rank,
                    "gateway_id": row.gateway_id,
                    "score": float(
                        row.anomaly_score
                    ),
                    "reason": (
                        "V4.1 gateway-relative "
                        "Isolation Forest anomaly"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main(
    argv: list[str] | None = None,
) -> int:

    here = pathlib.Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    default_data = (
        here / "data"
        if (here / "data").exists()
        else (
            here.parent
            / "student-brief"
            / "data"
        )
    )

    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=default_data,
        help="Path containing the telemetry directory",
    )

    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=(
            here
            / "predictions_ml_v4_1.csv"
        ),
        help="Output predictions CSV path",
    )

    args = parser.parse_args(
        argv
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    frame = load(
        args.data
    )

    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    predictions = build_ml_predictions(
        frame
    )

    # ========================================================
    # SAVE
    # ========================================================

    predictions.to_csv(
        args.out,
        index=False,
    )

    print()
    print(
        f"wrote {args.out}"
    )

    print(
        f"rows: {len(predictions)}"
    )

    print(
        f"weeks: "
        f"{predictions['week_start'].nunique()}"
    )

    # ========================================================
    # FINAL BASIC CHECKS
    # ========================================================

    expected_rows = (
        len(SCORED_WEEKS)
        * VISITS_PER_WEEK
    )

    if len(predictions) != expected_rows:
        raise SystemExit(
            f"ERROR: expected {expected_rows} rows "
            f"but produced {len(predictions)}"
        )

    rows_per_week = (
        predictions
        .groupby("week_start")
        .size()
    )

    if not (
        rows_per_week == VISITS_PER_WEEK
    ).all():
        raise SystemExit(
            "ERROR: not every week has exactly "
            f"{VISITS_PER_WEEK} predictions"
        )

    if predictions[
        ["week_start", "gateway_id"]
    ].duplicated().any():
        raise SystemExit(
            "ERROR: duplicate gateway/week prediction found"
        )

    print()
    print(
        "validation checks:"
    )

    print(
        f"  expected rows: {expected_rows}"
    )

    print(
        f"  actual rows:   {len(predictions)}"
    )

    print(
        "  rows/week:     15"
    )

    print(
        "  duplicate gateway/week: none"
    )

    print()
    print(
        "Top predictions:"
    )

    print(
        predictions.head(15).to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )