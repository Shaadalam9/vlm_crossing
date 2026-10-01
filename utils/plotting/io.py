"""Saving plotly figures. Adapted from utils/plotting/io.py in crowd-dataset/crowd-city."""

import os

import plotly as py

import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


class IO:
    def __init__(self) -> None:
        pass

    @staticmethod
    def figures_dir() -> str:
        """Return <output>/figures, created if missing."""
        path = os.path.join(common.get_output_dir(), "figures")
        os.makedirs(path, exist_ok=True)
        return path

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=1, save_png=True, save_eps=True):
        """
        Save a Plotly figure as HTML, and as PNG and EPS when save_images is true.

        The HTML file is opened in the default browser when open_figures is true.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the PNG and EPS images in pixels.
            height (int, optional): Height of the PNG and EPS images in pixels.
            scale (int, optional): Scaling factor for the PNG image.
        """
        # Raster export goes through kaleido, which can hang on some systems.
        # Setting save_images to false keeps the interactive HTML only.
        if not common.get_configs("save_images"):
            save_png = False
            save_eps = False

        output = self.figures_dir()
        logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(output, filename + ".html"),
                        auto_open=bool(common.get_configs("open_figures")))

        try:
            if save_png:
                logger.info(f"Saving png file for {filename}.")
                fig.write_image(os.path.join(output, filename + ".png"), width=width, height=height, scale=scale)
            if save_eps:
                logger.info(f"Saving eps file for {filename}.")
                fig.write_image(os.path.join(output, filename + ".eps"), width=width, height=height)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Could not export {filename} as image (is kaleido installed?): {e}")
