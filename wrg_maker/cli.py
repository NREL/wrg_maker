# -*- coding: utf-8 -*-
"""
Command Line Interface to Make WRG files
"""

import os

import click
import pandas as pd
from rex.utilities.cli_dtypes import INT, STR
from rex.utilities.loggers import init_logger

from wrg_maker.wrg_maker import WrgMaker


@click.command()
@click.option(
    "--h5_file",
    "-h5",
    type=click.Path(),
    required=True,
    help="Path to H5 file to create wrg file from",
)
@click.option(
    "--resolution",
    "-res",
    type=int,
    required=True,
    help="Pixel resolution in km.",
)
@click.option("--wrg_file", "-wrg", type=click.Path(exists=False), default=None, help="Path to wrg_file")
@click.option(
    "--hub_height",
    "-h",
    default=100,
    type=int,
    show_default=True,
    help="Hub Height in meters to compute wrg statistics at",
)
@click.option("--bin_size", "-bin", default=30, type=float, show_default=True, help="Bin size in degrees")
@click.option(
    "--box_coords",
    "-bc",
    default=None,
    type=(int, int, int, int),
    show_default=True,
    help="Coordinates of bounding box to get pixels to compute WRG statistics for",
)
@click.option(
    "--max_workers",
    "-workers",
    default=None,
    type=INT,
    show_default=True,
    help="Maximum number of workers to use for parallel processing.",
)
@click.option(
    "--buffer",
    "-b",
    default=0.2,
    type=float,
    show_default=True,
    help="Buffer size in %",
)
@click.option("--verbose", "-v", is_flag=True, help="Flag to turn on debug logging.")
def main(h5_file, resolution, wrg_file, hub_height, bin_size, box_coords, max_workers, buffer, verbose):
    """
    WRG Maker command line interface
    """
    if verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    init_logger("wrg_maker", log_level=log_level)

    if box_coords:
        box_coords = ((box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]))

    WrgMaker.run(
        h5_file,
        resolution,
        wrg_file=wrg_file,
        hub_height=hub_height,
        bin_size=bin_size,
        box_coords=box_coords,
        max_workers=max_workers,
        buffer=buffer,
    )


if __name__ == "__main__":
    main()
