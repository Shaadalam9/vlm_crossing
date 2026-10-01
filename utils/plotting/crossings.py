"""Plotly figures of crossing speed and hesitation time per city."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

import common
from utils.plotting.io import IO

io = IO()

METRIC_LABELS = {
    "speed_index": "Crossing speed (relative index, 1 = video median)",
    "speed_bbox_index": "Speed, bounding boxes (relative index)",
    "speed_seg_index": "Speed, on-road frames (relative index)",
    "hesitation_s": "Hesitation time (s)",
    "hesitation_bbox_s": "Hesitation, bounding boxes (s)",
    "hesitation_seg_s": "Hesitation before road entry (s)",
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
            margin=dict(l=70, r=20, t=50, b=80),
        )
        return fig

    def bar_mean_per_city(self, summary: pl.DataFrame, mean_col: str, std_col: str, n_col: str, label: str,
                          name: str, reference: float = None) -> None:
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
        if reference is not None:
            fig.add_hline(y=reference, line_dash="dot", line_color="grey")
        fig.update_layout(xaxis_title="City", yaxis_title=label)
        io.save_plotly_figure(self._layout(fig), name)

    def crossings_per_city(self, summary: pl.DataFrame, name: str) -> None:
        """Counted crossings per city, next to how many of them got each measurement."""
        data = summary.sort("crossings", descending=True)
        if data.height == 0:
            return
        fig = go.Figure([
            go.Bar(name="Crossings", x=data["city"].to_list(), y=data["crossings"].to_list()),
            go.Bar(name="with speed", x=data["city"].to_list(), y=data["crossings_with_speed"].to_list()),
            go.Bar(name="with hesitation", x=data["city"].to_list(), y=data["crossings_with_hesitation"].to_list()),
        ])
        fig.update_layout(barmode="group", xaxis_title="City", yaxis_title="Crossings")
        io.save_plotly_figure(self._layout(fig), name)

    def box_per_city(self, crossings: pl.DataFrame, metric: str, name: str) -> None:
        """Box plot of a per-crossing metric by city, every crossing shown as a point."""
        data = crossings.filter(pl.col(metric).is_not_null())
        if data.height == 0:
            return
        fig = px.box(data.to_dict(as_series=False), x="city", y=metric, points="all",
                     hover_data=["video", "track_id"],
                     labels={"city": "City", metric: METRIC_LABELS.get(metric, metric)})
        io.save_plotly_figure(self._layout(fig), name)

    def scatter_speed_hesitation(self, summary: pl.DataFrame, name: str) -> None:
        """Mean hesitation time against mean crossing speed, one point per city."""
        data = summary.filter(pl.col("speed_index_mean").is_not_null() & pl.col("hesitation_mean_s").is_not_null())
        if data.height == 0:
            return
        fig = px.scatter(data.to_dict(as_series=False), x="speed_index_mean", y="hesitation_mean_s", text="city",
                         size="crossings",
                         labels={"speed_index_mean": METRIC_LABELS["speed_index"],
                                 "hesitation_mean_s": METRIC_LABELS["hesitation_s"]})
        fig.update_traces(textposition="top center")
        io.save_plotly_figure(self._layout(fig), name)

    def bbox_vs_segmentation_speed(self, crossings: pl.DataFrame, name: str) -> None:
        """Per-crossing speed from the whole track against the on-road frames only."""
        data = crossings.filter(pl.col("speed_bbox_index").is_not_null() & pl.col("speed_seg_index").is_not_null())
        if data.height == 0:
            return
        fig = px.scatter(data.to_dict(as_series=False), x="speed_bbox_index", y="speed_seg_index", color="city",
                         hover_data=["video", "track_id"],
                         labels={"speed_bbox_index": METRIC_LABELS["speed_bbox_index"],
                                 "speed_seg_index": METRIC_LABELS["speed_seg_index"], "city": "City"})
        top = max(data["speed_bbox_index"].max(), data["speed_seg_index"].max()) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dot", color="grey"))
        io.save_plotly_figure(self._layout(fig), name)

    def plot_all(self, crossings: pl.DataFrame, summary: pl.DataFrame) -> None:
        """Write every figure to <output>/figures."""
        self.crossings_per_city(summary, "crossings_city")
        self.bar_mean_per_city(summary, "speed_index_mean", "speed_index_std", "crossings_with_speed",
                               METRIC_LABELS["speed_index"], "speed_crossing_city", reference=1.0)
        self.bar_mean_per_city(summary, "hesitation_mean_s", "hesitation_std_s", "crossings_with_hesitation",
                               METRIC_LABELS["hesitation_s"], "hesitation_time_city")
        for metric in ("speed_index", "hesitation_s", "hesitation_bbox_s", "speed_bbox_index", "crossing_time_s"):
            self.box_per_city(crossings, metric, f"box_{metric}")
        self.scatter_speed_hesitation(summary, "scatter_speed-hesitation")
        self.bbox_vs_segmentation_speed(crossings, "scatter_speed_bbox-segmentation")
