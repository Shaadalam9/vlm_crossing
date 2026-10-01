"""Export the README figures to readme/ as PNG and interactive HTML.

Run after analysis.py. The PNGs are shown in README.md and link to the HTML
versions through htmlpreview.github.io, as in crowd-dataset/crowd. The HTML
loads plotly.js from its CDN, so each file stays small enough to commit.
"""

import os

import polars as pl

import common
import utils.plotting.crossings as plots

# Figures shown in README.md.
README_FIGURES = (
    "crossings_city",
    "speed_crossing_city",
    "box_speed_index",
    "box_hesitation_bbox_s",
    "scatter_speed_bbox-segmentation",
    "box_crossing_time_s",
)
README_DIR = os.path.join(common.root_dir, "readme")


def save(fig, name, **kwargs) -> None:
    if name not in README_FIGURES:
        return
    fig.write_image(os.path.join(README_DIR, name + ".png"), width=1400, height=650, scale=1)
    fig.write_html(os.path.join(README_DIR, name + ".html"), include_plotlyjs="cdn")


if __name__ == "__main__":
    os.makedirs(README_DIR, exist_ok=True)
    output = common.get_output_dir()
    crossings = pl.read_parquet(os.path.join(output, "crossings.parquet"))
    summary = pl.read_csv(os.path.join(output, "city_summary.csv"))
    plots.io.save_plotly_figure = save
    plots.Crossings().plot_all(crossings, summary)
    print(f"Wrote {len(README_FIGURES)} figures to {README_DIR}.")
