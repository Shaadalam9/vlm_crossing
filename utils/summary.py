"""Per-city summary of the crowd-city crossing metrics."""

import polars as pl


def city_summary(crossings: pl.DataFrame, videos: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate crossings per city.

    Speed is crowd-city's relative motion index (1.0 = the median comparable
    pedestrian in the same video). Hesitation is in seconds and, as in
    crowd-city, only exists for pedestrians who stood still before crossing
    and were seen doing so, so its statistics cover only those crossings.

    Args:
        crossings (pl.DataFrame): One row per crossing, as written by analysis.py.
        videos (pl.DataFrame): One row per analysed video with city and duration_s.

    Returns:
        pl.DataFrame: One row per city.
    """
    footage = videos.group_by("city").agg(
        pl.len().alias("videos"),
        (pl.col("duration_s").sum() / 3600.0).alias("footage_h"),
    )
    speed = pl.col("speed_index")
    hes = pl.col("hesitation_s")
    per_city = crossings.group_by("city").agg(
        pl.len().alias("crossings"),
        speed.drop_nulls().len().alias("crossings_with_speed"),
        speed.mean().alias("speed_index_mean"),
        speed.median().alias("speed_index_median"),
        speed.std().alias("speed_index_std"),
        hes.drop_nulls().len().alias("crossings_with_hesitation"),
        hes.mean().alias("hesitation_mean_s"),
        hes.median().alias("hesitation_median_s"),
        hes.std().alias("hesitation_std_s"),
        pl.col("crossing_time_s").mean().alias("crossing_time_mean_s"),
        pl.col("speed_bbox_index").mean().alias("speed_bbox_index_mean"),
        pl.col("hesitation_bbox_s").mean().alias("hesitation_bbox_mean_s"),
    )
    summary = footage.join(per_city, on="city", how="left").with_columns(
        pl.col("crossings", "crossings_with_speed", "crossings_with_hesitation").fill_null(0),
    ).with_columns(
        (pl.col("crossings") / pl.col("footage_h")).alias("crossings_per_hour"),
    )
    return summary.sort("city")
