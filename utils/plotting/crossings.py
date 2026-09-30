"""Figures of crossing speed and hesitation time per city."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

import common
from utils.plotting.io import IO

io = IO()

METRIC_LABELS = {
    "speed_mps": "Crossing speed (m/s)",
    "hesitation_s": "Hesitation time (s)",
    "crossing_time_s": "Crossing time (s)",
}


class Crossings:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _layout(fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            font=dict(family=common.get_configs("font_family"), size=common.get_configs("font_size")),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        return fig

    def bar_mean_per_city(self, summary: pl.DataFrame, mean_col: str, std_col: str, n_col: str, label: str,
                          name: str) -> None:
        """Bar chart of a per-city mean with one standard deviation as error bar, sorted by the mean."""
        data = summary.filter(pl.col(mean_col).is_not_null()).sort(mean_col, descending=True)
        if data.height == 0:
            return
        fig = go.Figure(go.Bar(
            x=data["city"].to_list(),
            y=data[mean_col].to_list(),
            error_y=dict(type="data", array=data[std_col].fill_null(0).to_list()),
            text=[f"n={n}" for n in data[n_col].fill_null(0).to_list()],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="City", yaxis_title=label)
        io.save_plotly_figure(self._layout(fig), name)

    @staticmethod
    def _values(crossings: pl.DataFrame, metric: str) -> pl.DataFrame:
        """Crossings with a value; hesitation only for those who waited, matching city_summary."""
        data = crossings.filter(pl.col(metric).is_not_null())
        return data.filter(pl.col(metric) > 0) if metric == "hesitation_s" else data

    def box_per_city(self, crossings: pl.DataFrame, metric: str, name: str) -> None:
        """Box plot of a per-crossing metric by city."""
        data = self._values(crossings, metric)
        if data.height == 0:
            return
        fig = px.box(data.to_dict(as_series=False), x="city", y=metric, points="all",
                     labels={"city": "City", metric: METRIC_LABELS.get(metric, metric)})
        io.save_plotly_figure(self._layout(fig), name)

    def hist_metric(self, crossings: pl.DataFrame, metric: str, name: str) -> None:
        """Overlaid histogram of a per-crossing metric, one colour per city."""
        data = self._values(crossings, metric)
        if data.height == 0:
            return
        fig = px.histogram(data.to_dict(as_series=False), x=metric, color="city", barmode="overlay", opacity=0.6,
                           labels={"city": "City", metric: METRIC_LABELS.get(metric, metric)})
        fig.update_layout(yaxis_title="Crossings")
        io.save_plotly_figure(self._layout(fig), name)

    def scatter_speed_hesitation(self, summary: pl.DataFrame, name: str) -> None:
        """Scatter of mean hesitation time against mean crossing speed, one point per city."""
        data = summary.filter(pl.col("speed_mean_mps").is_not_null() & pl.col("hesitation_mean_s").is_not_null())
        if data.height == 0:
            return
        fig = px.scatter(data.to_dict(as_series=False), x="speed_mean_mps", y="hesitation_mean_s", text="city",
                         size="crossings",
                         labels={"speed_mean_mps": METRIC_LABELS["speed_mps"],
                                 "hesitation_mean_s": METRIC_LABELS["hesitation_s"]})
        fig.update_traces(textposition="top center")
        io.save_plotly_figure(self._layout(fig), name)

    def plot_all(self, crossings: pl.DataFrame, summary: pl.DataFrame) -> None:
        """Write every figure to <output>/figures."""
        self.bar_mean_per_city(summary, "speed_mean_mps", "speed_std_mps", "crossings_with_speed",
                               METRIC_LABELS["speed_mps"], "speed_crossing_city")
        self.bar_mean_per_city(summary, "hesitation_mean_s", "hesitation_std_s", "crossings_hesitated",
                               METRIC_LABELS["hesitation_s"], "hesitation_time_city")
        for metric in ("speed_mps", "hesitation_s"):
            self.box_per_city(crossings, metric, f"box_{metric}")
            self.hist_metric(crossings, metric, f"hist_{metric}")
        self.scatter_speed_hesitation(summary, "scatter_speed-hesitation")
