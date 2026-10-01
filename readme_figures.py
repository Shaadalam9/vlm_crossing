"""Export the README figures to readme/ as PNG and interactive HTML.

Run after analysis.py. The PNGs are shown in README.md and link to the HTML
versions through htmlpreview.github.io, as in crowd-dataset/crowd. Each HTML
file loads plotly.js from its CDN and draws the figure once it has loaded,
which keeps the files small; htmlpreview runs inline scripts before external
ones have loaded, so the usual plotly CDN output stays blank there.
"""

import os

import plotly
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
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body style="margin:0">
<div id="figure" style="width:100%;height:100vh"></div>
<script>
(function () {{
  var figure = {figure};
  var script = document.createElement("script");
  script.src = "https://cdn.plot.ly/plotly-{version}.min.js";
  script.onload = function () {{
    Plotly.newPlot("figure", figure.data, figure.layout, {{responsive: true}});
  }};
  document.head.appendChild(script);
}})();
</script>
</body>
</html>
"""


def save(fig, name, **kwargs) -> None:
    if name not in README_FIGURES:
        return
    fig.write_image(os.path.join(README_DIR, name + ".png"), width=1400, height=650, scale=1)
    html = HTML_TEMPLATE.format(title=name, figure=fig.to_json(), version=plotly.offline.get_plotlyjs_version())
    with open(os.path.join(README_DIR, name + ".html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    os.makedirs(README_DIR, exist_ok=True)
    output = common.get_output_dir()
    crossings = pl.read_parquet(os.path.join(output, "crossings.parquet"))
    summary = pl.read_csv(os.path.join(output, "city_summary.csv"))
    plots.io.save_plotly_figure = save
    plots.Crossings().plot_all(crossings, summary)
    print(f"Wrote {len(README_FIGURES)} figures to {README_DIR}.")
