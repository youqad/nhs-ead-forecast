"""Long-format CSV → daily wide DataFrame, with -9999 masking and Parquet cache."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

from submission.src.data.schema import DUMMY_VALUE, TARGET_COL as TARGET_METRIC_NAME


def read_long_csv(zip_path: str | Path) -> pl.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        if len(csvs) != 1:
            raise ValueError(f"Expected exactly one CSV in {zip_path}, found {csvs}")
        inner = csvs[0]
        with zf.open(inner) as f:
            return pl.read_csv(
                f,
                try_parse_dates=True,
                schema_overrides={"value": pl.Float64},
                null_values=["NA", ""],
            )


def long_to_daily_wide(df_long: pl.DataFrame) -> pd.DataFrame:
    df = df_long.with_columns(
        pl.when(pl.col("value") == DUMMY_VALUE)
        .then(None)
        .otherwise(pl.col("value"))
        .alias("value"),
        pl.col("dt").cast(pl.Datetime).alias("dt"),
    )

    df = df.with_columns(
        pl.when(pl.col("dt").dt.hour() < 12)
        .then(pl.col("dt").dt.date())
        .otherwise(pl.col("dt").dt.date() + pl.duration(days=1))
        .alias("midday_day")
    )

    target = (
        df.filter(pl.col("metric_name") == TARGET_METRIC_NAME)
        .group_by("midday_day")
        .agg(pl.col("value").mean().alias(TARGET_METRIC_NAME))
    )

    features = df.filter(pl.col("metric_name") != TARGET_METRIC_NAME).with_columns(
        (pl.col("metric_name") + " - " + pl.col("coverage_label")).alias("metric_label")
    )
    features_wide = (
        features.group_by(["midday_day", "metric_label"])
        .agg(pl.col("value").mean().alias("value"))
        .pivot(index="midday_day", on="metric_label", values="value")
    )

    wide = features_wide.join(target, on="midday_day", how="full", coalesce=True)
    pdf = wide.to_pandas().rename(columns={"midday_day": "date"}).set_index("date").sort_index()
    pdf.index = pd.to_datetime(pdf.index)
    return pdf


def load_or_build_daily(zip_path: str | Path, cache_dir: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_dir) / "daily_wide.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    long_df = read_long_csv(zip_path)
    daily = long_to_daily_wide(long_df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache_path)
    return daily
