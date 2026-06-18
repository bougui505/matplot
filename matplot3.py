#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Mar 31 13:12:22 2025

import math  # Added for arbitrary function plotting
import os
import socket
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # Added for tick formatting
import numpy as np
import scipy.signal  # Added for find_peaks
import scipy.stats  # Added for geometric mean
import typer
from numpy import linalg
from PIL import Image, PngImagePlugin
from PIL.PngImagePlugin import PngInfo
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.manifold import TSNE  # Added for tsne
from sklearn.neighbors import KernelDensity, NearestNeighbors

from draggable_text import DraggableText
from ROC import ROC

console = Console()

# Reading data from a png with large number of points:
# See: https://stackoverflow.com/a/61466412/1679629
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# Define X, Y, INTERACTIVE_LABELS, GLOBAL_C_VALUES as global variables for interactive mode
X, Y = list(), list()
INTERACTIVE_LABELS = list()
GLOBAL_C_VALUES = list()
XTICK_FORMAT = None
YTICK_FORMAT = None

app = typer.Typer(
    no_args_is_help=True,
    # pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
    rich_markup_mode="rich",
)

@app.callback()
def plot_setup(
    xlabel: Annotated[str, typer.Option(help="Label for the x-axis", rich_help_panel="Axis & Grid")] = "x",
    ylabel: Annotated[str, typer.Option(help="Label for the y-axis", rich_help_panel="Axis & Grid")] = "y",
    semilog_x: Annotated[bool, typer.Option(help="Set x-axis to log scale", rich_help_panel="Axis & Grid")] = False,
    semilog_y: Annotated[bool, typer.Option(help="Set y-axis to log scale", rich_help_panel="Axis & Grid")] = False,
    grid: Annotated[bool, typer.Option(help="Add a grid to the plot", rich_help_panel="Axis & Grid")] = False,
    aspect_ratio: Annotated[Optional[str], typer.Option(help="Set the figure size (e.g., '10 5' for 10x5 inches)", rich_help_panel="Layout")] = None,
    subplots: Annotated[str, typer.Option(help="Define subplot grid (e.g., '2 2' for a 2x2 grid)", rich_help_panel="Layout")] = "1 1",
    sharex: Annotated[bool, typer.Option(help="Share the x-axis across subplots", rich_help_panel="Layout")] = False,
    sharey: Annotated[bool, typer.Option(help="Share the y-axis across subplots", rich_help_panel="Layout")] = False,
    titles: Annotated[str, typer.Option(help="Titles for each subplot, separated by spaces. Quoted titles with spaces should be enclosed in double quotes. Example: --titles \"'Title 1' 'Title 2' 'Title with spaces'\"", rich_help_panel="Layout")] = "",
    xtick_format: Annotated[Optional[str], typer.Option(help="Format string for x-axis tick labels (e.g., '%.2f', '%d', '%.1e')", rich_help_panel="Tick Formatting")] = None,
    ytick_format: Annotated[Optional[str], typer.Option(help="Format string for y-axis tick labels (e.g., '%.2f', '%d', '%.1e')", rich_help_panel="Tick Formatting")] = None,
    xtick_fontsize: Annotated[int | None, typer.Option(help="The fontsize of the xtick labels", rich_help_panel="Tick Formatting")] = None,
    debug: Annotated[bool, typer.Option(help="Enable debug mode", rich_help_panel="Debug")] = False,
):
    """
    A new dataset can be defined by separating the data by an empty line.
    """
    global XTICK_FORMAT
    global YTICK_FORMAT
    global XTICK_FONTSIZE
    XTICK_FORMAT = xtick_format
    YTICK_FORMAT = ytick_format
    XTICK_FONTSIZE = xtick_fontsize
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = DEBUG
    if aspect_ratio is not None:
        xaspect, yaspect = aspect_ratio.split()
        plt.figure(figsize=(float(xaspect), float(yaspect)))
    global SUBPLOTS
    SUBPLOTS = [int(e) for e in subplots.strip().split()]
    ax = None
    global TITLES
    # Parse titles properly to handle spaces
    if titles.strip() == "":
        TITLES = []
    else:
        # Split by spaces but preserve quoted strings
        import shlex
        try:
            TITLES = shlex.split(titles)
        except ValueError:
            # Fallback for malformed input
            TITLES = titles.strip().split()
    
    if len(TITLES) < SUBPLOTS[0] * SUBPLOTS[1]:
        TITLES += [""] * (SUBPLOTS[0] * SUBPLOTS[1] - len(TITLES))
    for i in range(SUBPLOTS[0] * SUBPLOTS[1]):
        ax = plt.subplot(
            SUBPLOTS[0],
            SUBPLOTS[1],
            i + 1,
            sharex=None if not sharex else ax,
            sharey=None if not sharey else ax,
        )
        subplot_ij = np.unravel_index(i, SUBPLOTS)
        if sharex and subplot_ij[0] == SUBPLOTS[0] - 1:
            plt.xlabel(xlabel)
        if not sharex:
            plt.xlabel(xlabel)
        if sharey and subplot_ij[1] == 0:
            plt.ylabel(ylabel)
        if not sharey:
            plt.ylabel(ylabel)
        if semilog_x:
            plt.semilogx()
            plt.gca().xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs='all'))
            plt.gca().xaxis.set_major_formatter(mticker.LogFormatterSciNotation(minor_thresholds=(2.0, 0.8)))
        if semilog_y:
            plt.semilogy()
            plt.gca().yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs='all'))
            plt.gca().yaxis.set_major_formatter(mticker.LogFormatterSciNotation(minor_thresholds=(2.0, 0.8)))
        if grid:
            plt.grid()
        if titles != "":
            plt.title(TITLES[i])

        # Tick formatters will be applied after data is plotted

def read_data(delimiter, fields, labels):
    """
    Read data from standard input and organize it into a dictionary.

    Args:
        delimiter (str): The delimiter used to split the data.
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels for the data, separated by spaces.

    Returns:
        tuple: A tuple containing the data dictionary, the data string, and the fields.
    """
    data = OrderedDict()
    datastr = ""
    imax = -1
    field_offset = 0
    lines = sys.stdin.readlines()
    # remove all the last empty lines
    while len(lines) > 0 and lines[-1].strip() == "":
        lines.pop()
    fields_original = fields
    for line in lines:
        line = line.strip().split(delimiter)
        # check if the line is empty
        if len(line) == 0:
            # create new fields
            field_offset = imax + 1
            # duplicate the fields
            fields += f" {fields_original}"
        for i, e in enumerate(line):
            i += field_offset
            # Ensure the key exists in the OrderedDict
            if i not in data:
                data[i] = []
            data[i].append(e)
            datastr += e + ","
            if i > imax:
                imax = i
        datastr += "\n"
    # Printing data as a table
    fields_list = fields.strip().split()
    labels_list = labels.strip().split()
    ndataset = (np.asarray(fields_list) == "y").sum()
    if len(labels_list) == 0:
        labels_list = [""] * ndataset
    titles_list = TITLES
    if len(titles_list) < ndataset:
        titles_list += [""] * (ndataset - len(titles_list))
    assert len(data) == len(fields_list), f"Number of fields ({len(fields_list)}) and data ({len(data)}) does not match"
    assert len(labels_list) == ndataset, f"Number of y fields ({len(labels_list)}) and labels ({ndataset}) does not match"
    table = Table("ids", "lengths", "fields", "labels", "titles")
    j = 0
    for i in data.keys():
        field = fields_list[i]
        if field == "y":
            label = labels_list[j]
            title = titles_list[j]
            j += 1
        else:
            label = ""
            title = ""
        length = len(data[i])
        table.add_row(str(i), str(length), field, label, title)
    console.print(table)
    print(f"{fields=}")
    return data, datastr, fields

def set_limits(xmin=None, xmax=None, ymin=None, ymax=None, equal_aspect: bool = False):
    """
    Set the limits of the plot.

    Args:
        xmin (float): The minimum x value.
        xmax (float): The maximum x value.
        ymin (float): The minimum y value.
        ymax (float): The maximum y value.
        equal_aspect (bool): If True, set the aspect ratio of the plot to equal.
    """
    print(f"set_limits called with ymin={ymin}, ymax={ymax}")
    limits = plt.axis()
    if xmin is None:
        xmin = limits[0]
    if xmax is None:
        xmax = limits[1]
    ax = plt.gca()
    ax.set_xlim(left=float(xmin), right=float(xmax))
    if ymin is None:
        ymin = limits[-2]
    if ymax is None:
        ymax = limits[-1]
    ax.set_ylim(bottom=float(ymin), top=float(ymax))
    for idx, axis in enumerate(plt.gcf().axes):
        print(f"Subplot {idx+1} ylim: {axis.get_ylim()}")
    if equal_aspect:
        ax.set_aspect('equal')

def saveplot(outfilename, datastr, labels=None):
    """
    Save the plot to a file.

    Args:
        outfilename (str): The filename to save the plot to.
        datastr (str): The data string to add to the plot.
        labels (list): The labels for the data.
    """
    plt.savefig(outfilename)
    ext = os.path.splitext(outfilename)[1]
    print(f"{ext=}")
    print(f"{outfilename=}")
    if ext == ".png":
        add_metadata(outfilename, datastr, labels=labels)

def add_metadata(filename, datastr, key="data", labels=None):
    """
    Add metadata to a PNG file.

    Args:
        filename (str): The filename of the PNG file.
        datastr (str): The data string to add to the PNG file.
        key (str): The key to use for the data string.
        labels (list): The labels for the data.
    """
    metadata = PngInfo()
    metadata.add_text(key, datastr, zip=True)
    if labels is not None:
        metadata.add_text(f"labels", " ".join(labels))
    metadata.add_text("cwd", os.getcwd())
    metadata.add_text("hostname", socket.gethostname())
    # if options.subsample is not None:
    #     print("# Adding subsampling metadata")
    #     metadata.add_text("subsampling", "1st-column")
    targetImage = Image.open(filename)
    targetImage.save(filename, pnginfo=metadata)

def set_xtick_labels(fields, data, rotation=45, fontsize=None):
    """
    Set the x-tick labels of the plot.

    Args:
        fields (list): The fields of the data.
        data (dict): The data dictionary.
        rotation (int): The rotation of the x-tick labels in degrees.
        fontsize (int): The fontsize of the x-tick labels.
    """
    if 'xt' in fields:
        xtickslabels = data[fields.index('xt')]
        xval = np.float64(data[fields.index('x')])
        xval, unique_indices = np.unique(xval, return_index=True)
        xtickslabels = np.array(xtickslabels)[unique_indices]
        plt.xticks(xval, xtickslabels.astype(str).tolist())
        ha = "right" if rotation != 0 else "center"
        plt.setp(plt.gca().get_xticklabels(),
                 rotation=rotation,
                 ha=ha,
                 rotation_mode="anchor")
        # Use global fontsize if not specified locally
        if fontsize is None:
            fontsize = XTICK_FONTSIZE
        if fontsize is not None:
            plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize)

def set_ytick_labels(fields, data, rotation=0, fontsize=None):
    """
    Set the y-tick labels of the plot.

    Args:
        fields (list): The fields of the data.
        data (dict): The data dictionary.
        rotation (int): The rotation of the y-tick labels in degrees.
        fontsize (int): The fontsize of the y-tick labels.
    """
    if 'yt' in fields:
        ytickslabels = data[fields.index('yt')]
        yval = np.float64(data[fields.index('y')])
        yval, unique_indices = np.unique(yval, return_index=True)
        ytickslabels = np.array(ytickslabels)[unique_indices]
        plt.yticks(yval, ytickslabels.astype(str).tolist())
        # rotate the labels
        plt.setp(plt.gca().get_yticklabels(),
                 rotation=rotation,
                 ha="right",
                 rotation_mode="anchor")
        if fontsize is not None:
            plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize)

def out(
    save,
    datastr,
    labels,
    colorbar,
    xmin,
    xmax,
    ymin,
    ymax,
    cbar_label=None,
    interactive_plot: bool = True,
    legend: bool = True,
    equal_aspect: bool = False,
):
    """
    Display or save the plot.

    Args:
        save (str): The filename to save the plot to.
        datastr (str): The data string to add to the plot.
        labels (list): The labels for the data.
        colorbar (bool): If True, add a colorbar to the plot.
        xmin (float): The minimum x value.
        xmax (float): The maximum x value.
        ymin (float): The minimum y value.
        ymax (float): The maximum y value.
        cbar_label (str): The label for the colorbar.
        interactive_plot (bool): If True, enable interactive plot.
        legend (bool): If True, display a legend on the plot.
        equal_aspect (bool): If True, set the aspect ratio of the plot to equal.
    """
    set_limits(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, equal_aspect=equal_aspect)
    if colorbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    if labels is not None:
        if len(labels) > 0 and legend:
            plt.legend()
    if save == "":
        # build a kdtree for X, Y
        global NEIGH
        NEIGH = None
        if interactive_plot:
            NEIGH = NearestNeighbors(n_neighbors=1, algorithm='auto')
            NEIGH.fit(np.vstack((X, Y)).T)
            cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    else:
        saveplot(save, datastr, labels)

def onclick(event):
    """
    Handle a click event on the plot.

    Args:
        event: The click event.
    """
    if event.xdata is not None and event.ydata is not None:
        # find the nearest point in the kdtree
        print(event.xdata, event.ydata)
        if NEIGH is not None:
            dist, index = NEIGH.kneighbors(np.asarray([event.xdata, event.ydata]).reshape(1, -1))
            index = index.squeeze()
            dist = dist.squeeze()
            x = X[index]
            y = Y[index]
            if len(INTERACTIVE_LABELS) > 0:
                label = INTERACTIVE_LABELS[index]
            else:
                label = ""
            # Check if index is valid for GLOBAL_C_VALUES
            if index < len(GLOBAL_C_VALUES):
                c_val = GLOBAL_C_VALUES[index]
                print_str = f"Nearest point: {label} x={x}, y={y}"
                if not np.isnan(c_val): # Only display c value if it's not NaN
                    print_str += f", c={format_nbr(c_val)}"
                print_str += f", dist={dist:.2g}"
                print(print_str)
            else:
                print_str = f"Nearest point: {label} x={x}, y={y}"
                print_str += f", dist={dist:.2g}"
                print(print_str)

def toint(x):
    """
    Convert a value to an integer if possible.

    Args:
        x: The value to convert.

    Returns:
        int: The converted value, or the original value if conversion is not possible.
    """
    try:
        x = int(x)
    except ValueError:
        pass
    return x

def _apply_axis_tick_formats(ax, x_data, y_data):
    """
    Applies x and y axis tick formatters based on global settings or data type.
    This function is intended for numerical data. It will not apply formatting
    if the tick labels are already strings (e.g., from categorical data).
    """
    # Check if tick labels are already strings (likely categorical data)
    # If so, do not apply numerical formatting.
    x_tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    y_tick_labels = [t.get_text() for t in ax.get_yticklabels()]

    # Heuristic: if any tick label is not purely numeric, assume it's categorical
    is_x_categorical = any(not t.replace('.', '', 1).replace('-', '', 1).isdigit() for t in x_tick_labels)
    is_y_categorical = any(not t.replace('.', '', 1).replace('-', '', 1).isdigit() for t in y_tick_labels)

    if not is_x_categorical:
        effective_xtick_format = XTICK_FORMAT
        if effective_xtick_format is None and ax.xaxis.get_scale() != 'log' and np.all(x_data == np.round(x_data)):
            effective_xtick_format = "%d"

        if effective_xtick_format:
            if effective_xtick_format == "%d" and ax.xaxis.get_scale() != 'log':
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(effective_xtick_format))
    else:
        # Remove secondary ticks when xticklabels are text
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    if not is_y_categorical:
        effective_ytick_format = YTICK_FORMAT
        if effective_ytick_format is None and ax.yaxis.get_scale() != 'log' and np.all(y_data == np.round(y_data)):
            effective_ytick_format = "%d"

        if effective_ytick_format:
            if effective_ytick_format == "%d" and ax.yaxis.get_scale() != 'log':
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(effective_ytick_format))
    else:
        # Remove secondary ticks when yticklabels are text
        ax.yaxis.set_minor_locator(mticker.NullLocator())

@app.command(help="Plot data from standard input, and optionally an arbitrary function or an average curve.")
def plot(
    # Data input options
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]x[/cyan]: x-axis, [cyan]y[/cyan]: y-axis, [cyan]xt[/cyan]: x-tick labels, [cyan]ts[/cyan]: timestamp (seconds), [cyan]s[/cyan]: std dev.", rich_help_panel="Data Input")] = "x y",
    labels: Annotated[str, typer.Option(help="Space-separated labels for each 'y' field.", rich_help_panel="Data Input")] = "",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    moving_avg: Annotated[int, typer.Option(help="Size of the moving average window", rich_help_panel="Data Input")] = 0,
    # Plot appearance options
    fmt: Annotated[str, typer.Option(help="Format string for the plot (e.g., 'r--', 'bo')", rich_help_panel="Plot Appearance")] = "",
    alpha: Annotated[float, typer.Option(help="Alpha (transparency) value for the plot", rich_help_panel="Plot Appearance")] = 1.0,
    rotation: Annotated[int, typer.Option(help="Rotation of xtick labels in degrees", rich_help_panel="Plot Appearance")] = 45,
    plot_points: Annotated[bool, typer.Option(help="Plot individual data points on the line", rich_help_panel="Plot Appearance")] = False,
    size: Annotated[int, typer.Option(help="Size of markers in the plot", rich_help_panel="Plot Appearance")] = 10,
    # Output and limits options
    save: Annotated[str, typer.Option(help="Filename to save the plot to (e.g., 'plot.png')", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float | None, typer.Option(help="Minimum x value for the plot", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value for the plot", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value for the plot", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value for the plot", rich_help_panel="Output & Limits")] = None,
    shade: Annotated[str | None, typer.Option(help="Shade area under the curve (0 or 1 per y field, e.g., '0 1')", rich_help_panel="Plot Appearance")] = None,
    alpha_shade: Annotated[float, typer.Option(help="Alpha value for the shaded area", rich_help_panel="Plot Appearance")] = 0.2,
    legend: Annotated[bool, typer.Option(help="Display legend on the plot", rich_help_panel="Plot Appearance")] = True,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal", rich_help_panel="Plot Appearance")] = False,
    # Statistical analysis options
    mark_minima: Annotated[bool, typer.Option(help="Mark local minima with coordinates", rich_help_panel="Statistical Analysis")] = False,
    mark_absolute_minima: Annotated[bool, typer.Option(help="Mark absolute minimum with coordinates", rich_help_panel="Statistical Analysis")] = False,
    plot_average: Annotated[bool, typer.Option(help="Plot the average curve of all 'y' datasets", rich_help_panel="Statistical Analysis")] = False,
    average_linestyle: Annotated[str, typer.Option(help="Linestyle for the average curve", rich_help_panel="Statistical Analysis")] = '--',
    plot_median: Annotated[bool, typer.Option(help="Plot the median curve of all 'y' datasets", rich_help_panel="Statistical Analysis")] = False,
    median_linestyle: Annotated[str, typer.Option(help="Linestyle for the median curve", rich_help_panel="Statistical Analysis")] = '-.',
    plot_gmean: Annotated[bool, typer.Option(help="Plot the geometric mean curve of all 'y' datasets", rich_help_panel="Statistical Analysis")] = False,
    gmean_linestyle: Annotated[str, typer.Option(help="Linestyle for the geometric mean curve", rich_help_panel="Statistical Analysis")] = ':',
    mark_average_minima: Annotated[bool, typer.Option(help="Mark global minimum of the average curve", rich_help_panel="Statistical Analysis")] = False,
    mark_median_minima: Annotated[bool, typer.Option(help="Mark global minimum of the median curve", rich_help_panel="Statistical Analysis")] = False,
    mark_gmean_minima: Annotated[bool, typer.Option(help="Mark global minimum of the geometric mean curve", rich_help_panel="Statistical Analysis")] = False,
    # Function plotting options
    function: Annotated[Optional[str], typer.Option(help="Mathematical expression to plot (e.g., 'x**2 + 2*x + 1')", rich_help_panel="Function Plotting")] = None,
    func_label: Annotated[Optional[str], typer.Option(help="Label for the plotted function in the legend", rich_help_panel="Function Plotting")] = None,
    func_linestyle: Annotated[str, typer.Option(help="Linestyle for the plotted function", rich_help_panel="Function Plotting")] = '-',
    func_color: Annotated[str, typer.Option(help="Color for the plotted function", rich_help_panel="Function Plotting")] = 'r',
    # Test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_npts: Annotated[int, typer.Option(help="Number of points to generate for testing", rich_help_panel="Test Data")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="Number of datasets to generate for testing", rich_help_panel="Test Data")] = 2,
):
    """
    Plot data from standard input, and optionally an arbitrary function or an average curve.

    Args:
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels to use for the data, separated by spaces.
        moving_avg (int): The size of the moving average window.
        delimiter (str): The delimiter to use to split the data.
        fmt (str): The format string to use for the plot, separated by spaces.
        alpha (float): The alpha value for the plot.
        rotation (int): The rotation of the xtick labels in degrees.
        save (str): The filename to save the plot to.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
        shade (str): The values to shade the area under the curve, separated by spaces.
        alpha_shade (float): The alpha value for the shaded area.
        test (bool): If True, generate random data for testing.
        test_npts (int): The number of points to generate for testing.
        test_ndata (int): The number of datasets to generate for testing.
    """
    if test:
        data = dict()
        fields = ""
        j = 0
        data[j] = np.arange(test_npts)
        fields += "x "
        j += 1
        for i in range(test_ndata):
            data[j] = np.random.normal(size=test_npts, loc=i, scale=100) + np.arange(test_npts) + i*test_npts
            fields += "y "
            j += 1
        datastr = ""
    else:
        data, datastr, fields = read_data(delimiter, fields, labels)

    fields = fields.strip().split()  #type: ignore
    if shade is not None:
        shade = shade.strip().split()  #type: ignore
        shade = np.bool_(np.int_(shade))  #type: ignore
    else:
        shade = np.zeros(len(fields), dtype=bool)  #type: ignore
    xfmt = None
    if "ts" in fields:
        xfmt = "ts"
        fields = [f if f != "ts" else "x" for f in fields]  #type: ignore
    assert "x" in fields, "x field is required"
    labels_list = labels.strip().split()  #type: ignore
    if fmt != "":
        fmt_list = fmt.strip().split()  #type: ignore
    else:
        fmt_list = [fmt] * len(data)  #type: ignore
    plotid = 0
    xfields = np.where(np.asarray(fields) == "x")[0]
    yfields = np.where(np.asarray(fields) == "y")[0]
    sfields = np.where(np.asarray(fields) == "s")[0]  # New field for standard deviation
    assert len(xfields) == len(yfields) or len(xfields) == 1, "x and y fields must be the same length or x must be a single field"
    if len(xfields) < len(yfields) and len(xfields) == 1:
        xfields = np.ones_like(yfields) * xfields[0]

    all_x_data = [] # To store all x values for function plotting range
    y_arrays_for_average_median = [] # Consolidated list for both average and median
    x_for_avg_med_calculation = None # Consolidated x for average and median

    for xfield, yfield in track(zip(xfields, yfields), total=len(xfields), description="Plotting data..."):
        x_current = np.float64(data[xfield])  #type: ignore
        y_current = np.float64(data[yfield])  #type: ignore
        all_x_data.extend(list(x_current)) # Collect x data for potential function plotting range
        X.extend(list(x_current))  #type: ignore
        Y.extend(list(y_current))  #type: ignore

        # Store x for average/median if not set, assuming all y's share this x
        if x_for_avg_med_calculation is None:
            x_for_avg_med_calculation = x_current

        if moving_avg > 0:
            x_current = np.convolve(x_current, np.ones((moving_avg,))/moving_avg, mode='valid')
            y_current = np.convolve(y_current, np.ones((moving_avg,))/moving_avg, mode='valid')

        y_arrays_for_average_median.append(y_current)

        # Handle standard deviation if provided
        std_current = None
        if len(sfields) > 0:
            sfield = sfields[plotid] if plotid < len(sfields) else sfields[0]  # Use corresponding s field or first one
            std_current = np.float64(data[sfield])  #type: ignore
            if moving_avg > 0:
                std_current = np.convolve(std_current, np.ones((moving_avg,))/moving_avg, mode='valid')

        if len(labels_list) > 0:
            label = labels_list[plotid]
        else:
            label = None
        if len(fmt_list) > 0:
            fmtstr = fmt_list[plotid]
        else:
            fmtstr = ""
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))
        if xfmt == "ts":
            x_current_plot = np.asarray([datetime.fromtimestamp(e) for e in x_current])  #type: ignore
        else:
            x_current_plot = x_current
        plt.plot(x_current_plot, y_current, fmtstr, label=label, alpha=alpha)
        if plot_points:
            plt.scatter(x_current_plot, y_current, s=size, alpha=alpha, color=plt.gca().lines[-1].get_color()) # Plot points with same color as line
        if std_current is not None and len(std_current) == len(y_current):
            # Plot standard deviation as shaded area
            color = plt.gca().lines[-1].get_color()
            plt.fill_between(x_current_plot, y_current - std_current, y_current + std_current, alpha=alpha_shade, color=color, label=f"{label} ± std" if label else None)
        if xfmt == "ts":
            plt.gcf().autofmt_xdate()
        if mark_minima:
            # Find local minima by finding peaks in the inverted y data
            minima_indices, _ = scipy.signal.find_peaks(-y_current)
            if len(minima_indices) > 0:
                min_x = x_current[minima_indices]
                min_y = y_current[minima_indices]
                # Plot markers for minima
                plt.scatter(min_x, min_y, marker='x', color='purple', s=100, zorder=5)
                # Add text labels for minima
                for mx, my in zip(min_x, min_y):
                    plt.annotate(f"({format_nbr(mx)}, {format_nbr(my)})", (mx, my),
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='purple')
        if mark_absolute_minima:
            min_y_abs = np.min(y_current)
            min_x_abs = x_current[np.argmin(y_current)]
            # Plot marker for absolute minimum
            plt.scatter(min_x_abs, min_y_abs, marker='*', color='red', s=200, zorder=6)
            # Add text label for absolute minimum
            plt.annotate(f"({format_nbr(min_x_abs)}, {format_nbr(min_y_abs)})", (min_x_abs, min_y_abs),
                         textcoords="offset points", xytext=(0,-20), ha='center', fontsize=9, color='red',
                         bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7, ec="red", lw=0.5))
        if shade[plotid]:  #type: ignore
            # get the color of the last plot:
            color = plt.gca().lines[-1].get_color()
            plt.fill_between(x_current, y_current, alpha=alpha_shade, color=color)
        set_xtick_labels(fields, data, rotation=rotation)
        _apply_axis_tick_formats(plt.gca(), x_current, y_current) # Apply tick formats after plotting
        plotid += 1

    if (plot_average or plot_median or plot_gmean) and len(y_arrays_for_average_median) > 1:
        # Assuming all y's correspond to `x_for_avg_med_calculation`.
        # More robust solution would involve resampling/interpolation if x-values differ significantly.
        x_avg_med_plot = x_for_avg_med_calculation
        if xfmt == "ts":
            x_avg_med_plot = np.asarray([datetime.fromtimestamp(e) for e in x_for_avg_med_calculation]) #type: ignore

        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], 1) # Plot aggregate statistics on the first subplot

        if plot_average:
            y_average = np.mean(y_arrays_for_average_median, axis=0)
            line_average, = plt.plot(x_avg_med_plot, y_average, average_linestyle, label="Average")
            labels_list.append("Average") # Add average to labels for legend
            if mark_average_minima:
                _mark_global_minimum(x_avg_med_plot, y_average, "Average", color=line_average.get_color())

        if plot_median:
            y_median = np.median(y_arrays_for_average_median, axis=0)
            line_median, = plt.plot(x_avg_med_plot, y_median, median_linestyle, label="Median")
            labels_list.append("Median") # Add median to labels for legend
            if mark_median_minima:
                _mark_global_minimum(x_avg_med_plot, y_median, "Median", color=line_median.get_color())

        if plot_gmean:
            # Note: Geometric mean requires positive numbers. Handle potential zeros/negatives gracefully.
            y_gmean = scipy.stats.gmean(np.array(y_arrays_for_average_median) + np.finfo(float).eps, axis=0) # Add epsilon to avoid log(0)
            line_gmean, = plt.plot(x_avg_med_plot, y_gmean, gmean_linestyle, label="Geometric Mean")
            labels_list.append("Geometric Mean") # Add geometric mean to labels for legend
            if mark_gmean_minima:
                _mark_global_minimum(x_avg_med_plot, y_gmean, "Geometric Mean", color=line_gmean.get_color())

        if xfmt == "ts":
            plt.gcf().autofmt_xdate()

    if function is not None:
        if xmin is None and xmax is None: # Use combined x_data for function range
            # Use the range of the plotted data
            if len(all_x_data) > 0:
                func_xmin = np.min(all_x_data)
                func_xmax = np.max(all_x_data)
            else:
                func_xmin = 0.0
                func_xmax = 1.0 # Default if no data
        else:
            func_xmin = xmin if xmin is not None else 0.0
            func_xmax = xmax if xmax is not None else 1.0

        x_func = np.linspace(func_xmin, func_xmax, 500)
        # Evaluate the function string. Use a limited global scope for security.
        # This allows basic math operations and numpy functions, but nothing else.
        _globals = {"x": x_func, "np": np, "math": math}
        _locals = {}
        try:
            y_func = eval(function, {"__builtins__": {}}, _globals)
        except Exception as e:
            print(f"Error evaluating function '{function}': {e}")
            sys.exit(1)

        # Plot the function on the first subplot
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], 1)
        current_func_label = func_label if func_label else function
        plt.plot(x_func, y_func, color=func_color, linestyle=func_linestyle, label=current_func_label)
        if not labels_list and current_func_label: # If no labels from data and a function label is provided
             labels_list = [current_func_label] # So that out() triggers the legend

    out(save=save, datastr=datastr, labels=labels_list, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, equal_aspect=equal_aspect, legend=legend)

@app.command(help="Create a scatter plot from data in standard input.")
def scatter(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]x[/cyan]: x-axis, [cyan]y[/cyan]: y-axis, [cyan]xt[/cyan]: x-tick labels, [cyan]ts[/cyan]: timestamp, [cyan]t[/cyan]: point labels", rich_help_panel="Data Input")] = "x y",
    labels: Annotated[str, typer.Option(help="Space-separated labels for each 'y' field.", rich_help_panel="Data Input")] = "",
    moving_avg: Annotated[int, typer.Option(help="Size of the moving average window", rich_help_panel="Data Input")] = 0,
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    fmt: Annotated[str, typer.Option(help="Format string for the plot", rich_help_panel="Plot Appearance")] = "",
    alpha: Annotated[float, typer.Option(help="Alpha (transparency) value for the plot", rich_help_panel="Plot Appearance")] = 1.0,
    rotation: Annotated[int, typer.Option(help="Rotation of xtick labels in degrees", rich_help_panel="Plot Appearance")] = 45,
    # output options
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float | None, typer.Option(help="Minimum x value for the plot", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value for the plot", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value for the plot", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value for the plot", rich_help_panel="Output & Limits")] = None,
    shade: Annotated[str | None, typer.Option(help="Shade area under the curve (0 or 1 per y field)", rich_help_panel="Plot Appearance")] = None,
    alpha_shade: Annotated[float, typer.Option(help="Alpha value for the shaded area", rich_help_panel="Plot Appearance")] = 0.2,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_npts: Annotated[int, typer.Option(help="Number of points to generate for testing", rich_help_panel="Test Data")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="Number of datasets to generate for testing", rich_help_panel="Test Data")] = 2,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal", rich_help_panel="Plot Appearance")] = False,
    # New options for scatter plot
    kde: Annotated[bool, typer.Option(help="Use kernel density estimation to color the points", rich_help_panel="Scatter Features")] = False,
    kde_subset: Annotated[int, typer.Option(help="Number of points to use for the KDE", rich_help_panel="Scatter Features")] = 1000,
    kde_normalize: Annotated[bool, typer.Option(help="Normalize the KDE values", rich_help_panel="Scatter Features")] = False,
    cmap: Annotated[str, typer.Option(help="Colormap to use for the plot", rich_help_panel="Plot Appearance")] = "viridis",
    size: Annotated[int, typer.Option(help="Size of markers in the plot", rich_help_panel="Plot Appearance")] = 10,
    pcr: Annotated[bool, typer.Option(help="Perform principal component regression", rich_help_panel="Statistical Analysis")] = False,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot", rich_help_panel="Plot Appearance")] = False,
):
    """
    Create a scatter plot from data in standard input.

    Args:
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels to use for the data, separated by spaces.
        moving_avg (int): The size of the moving average window.
        delimiter (str): The delimiter to use to split the data.
        fmt (str): The format string to use for the plot, separated by spaces.
        alpha (float): The alpha value for the plot.
        rotation (int): The rotation of the xtick labels in degrees.
        save (str): The filename to save the plot to.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
        shade (str): The values to shade the area under the curve, separated by spaces.
        alpha_shade (float): The alpha value for the shaded area.
        test (bool): If True, generate random data for testing.
        test_npts (int): The number of points to generate for testing.
        test_ndata (int): The number of datasets to generate for testing.
    """
    global X, Y, INTERACTIVE_LABELS, GLOBAL_C_VALUES
    X = []
    Y = []
    INTERACTIVE_LABELS = []
    GLOBAL_C_VALUES = []
    kde_c = None
    if test:
        data = dict()
        fields = ""
        j = 0
        for i in range(test_ndata):
            data[j] = np.random.normal(size=test_npts, loc=i*10, scale=1)
            j += 1
            fields += "x "
            data[j] = np.random.normal(size=test_npts, loc=0, scale=1)
            j += 1
            fields += "y "
            data[j] = np.random.normal(size=test_npts, loc=0, scale=1)
            j += 1
            fields += "c "
            data[j] = np.random.normal(size=test_npts, loc=0, scale=50)
            j += 1
            fields += "s "
            data[j] = np.arange(test_npts)
            j += 1
            fields += "il "
        datastr = ""
    else:
        data, datastr, fields = read_data(delimiter, fields, labels)
    fields = fields.strip().split()  # type: ignore
    labels = labels.strip().split()  # type: ignore
    xfields = np.where(np.asarray(fields) == "x")[0]
    yfields = np.where(np.asarray(fields) == "y")[0]
    s_indices = np.where(np.asarray(fields) == "s")[0]
    c_indices = np.where(np.asarray(fields) == "c")[0]
    plotid = 0

    # Collect all x,y data first for KDE calculation if needed
    all_x = []
    all_y = []
    for xfield, yfield in zip(xfields, yfields):
        all_x.extend(list(np.float64(data[xfield]))) # type: ignore
        all_y.extend(list(np.float64(data[yfield]))) # type: ignore

    if kde:
        xy_data = np.vstack([all_x, all_y]).T
        # Apply subsetting if kde_subset is specified and less than total points
        if kde_subset < len(xy_data):
            # Randomly select subset for KDE computation
            subset_indices = np.random.choice(len(xy_data), size=kde_subset, replace=False)
            xy_subset = xy_data[subset_indices]
            # Fit KDE on subset
            kde_model = KernelDensity(kernel='gaussian', bandwidth="scott").fit(xy_subset) # Default bandwidth
            # Score all points using the model fitted on subset
            kde_c = np.exp(kde_model.score_samples(xy_data))
        else:
            # Fit KDE on all data
            kde_model = KernelDensity(kernel='gaussian', bandwidth="scott").fit(xy_data) # Default bandwidth
            # Score samples
            kde_c = np.exp(kde_model.score_samples(xy_data))
        if kde_normalize:
            kde_c -= kde_c.min()
            kde_c /= kde_c.max()

    current_data_idx = 0
    for xfield, yfield in zip(xfields, yfields):
        x = np.float64(data[xfield])  # type: ignore
        y = np.float64(data[yfield])  # type: ignore

        X.extend(list(x))  # type: ignore
        Y.extend(list(y))  # type: ignore

        c_for_subplot = None # Initialize for current subplot

        # Determine colors for current subplot
        if kde:
            c_for_subplot = kde_c[current_data_idx : current_data_idx + len(x)]
            current_data_idx += len(x)
        elif len(c_indices) > plotid:
            # Use plotid to get corresponding c field for current dataset
            c_for_subplot = np.float64(data[c_indices[plotid]])
        elif len(c_indices) == 1:
            # If only one c field but multiple y fields, apply it to all
            c_for_subplot = np.float64(data[c_indices[0]])

        # If no explicit 'c' data or KDE, fill with NaN for interactive display
        if c_for_subplot is None:
            c_for_subplot = np.full_like(x, np.nan)

        GLOBAL_C_VALUES.extend(list(c_for_subplot)) # Add c values for current subplot to global list

        if "il" in fields:
            INTERACTIVE_LABELS.extend(data[fields.index("il")])
        if len(labels) > 0:
            label = labels[0]
        else:
            label = None

        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))
        if "t" in fields:
            texts_to_drag = list()
            for x_, y_, t_ in zip(x, y, data[fields.index("t")]):  # type: ignore
                texts_to_drag.append(plt.annotate(t_, (x_, y_),
                                              zorder=10,
                                              arrowprops=dict(
                                                  arrowstyle='-',    # Crucially, this creates a plain line without an arrowhead
                                                  color='red',       # Sets the color of the line to red
                                                  linewidth=0.8,     # Sets the width of the line to make it thin
                                                  # This is the key: relpos=(0, 0) anchors the line's *start* to the
                                                  # bottom-left of the 'xytext' (text body's) bounding box.
                                                  relpos=(0, 0)
                                                  )
                                              # bbox=dict(facecolor='lightblue', alpha=0.7, pad=7, boxstyle="round,pad=0.5")
                                              )
                                     )
            draggable_text_instances = list()
            for text_to_drag in texts_to_drag:
                draggable_text_instances.append(DraggableText(text_to_drag))
            for draggable_text_instance in draggable_text_instances:
                draggable_text_instance.connect()

        if np.isnan(c_for_subplot).all():
            c_for_subplot = None

        # Determine marker size: use 's' field, then global 'size' option, then Matplotlib default
        effective_size = np.float64(data[s_indices[plotid]]) if len(s_indices) > 0 else size
        plt.scatter(x, y, s=effective_size, c=c_for_subplot, label=label, alpha=alpha, cmap=cmap)
        if pcr:
            do_pcr(x, y)
        set_xtick_labels(fields, data, rotation=rotation) # Use the defined rotation
        set_ytick_labels(fields, data)
        _apply_axis_tick_formats(plt.gca(), x, y) # Apply tick formats after plotting
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, equal_aspect=equal_aspect)

@app.command(help="Create a histogram from data in standard input.")
def hist(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]y[/cyan]: data field to histogram", rich_help_panel="Data Input")] = "y",
    labels: Annotated[str, typer.Option(help="Labels for each dataset", rich_help_panel="Data Input")] = "",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    bins: Annotated[str, typer.Option(help="Number of bins or 'auto'", rich_help_panel="Histogram Styling")] = "auto",
    alpha: Annotated[float, typer.Option(help="Alpha (transparency) value for the plot", rich_help_panel="Histogram Styling")] = 1.0,
    density: Annotated[bool, typer.Option(help="Normalize the histogram to form a probability density", rich_help_panel="Histogram Styling")] = False,
    kde: Annotated[bool, typer.Option(help="Plot a kernel density estimate instead of the histogram", rich_help_panel="Histogram Styling")] = False,
    kde_bandwidth: Annotated[str, typer.Option(help="Bandwidth for KDE ('scott', 'silverman' or float)", rich_help_panel="Histogram Styling")] = "scott",
    cumulative: Annotated[bool, typer.Option(help="Plot a cumulative histogram as a red curve on a secondary y-axis", rich_help_panel="Histogram Styling")] = False,
    # output options
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float | None, typer.Option(help="Minimum x value for the plot", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value for the plot", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value for the plot", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value for the plot", rich_help_panel="Output & Limits")] = None,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_npts: Annotated[int, typer.Option(help="Number of points to generate for testing", rich_help_panel="Test Data")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="Number of datasets to generate for testing", rich_help_panel="Test Data")] = 2,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal", rich_help_panel="Output & Limits")] = False,
):
    """
    Create a histogram from data in standard input.

    Args:
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels to use for the data, separated by spaces.
        delimiter (str): The delimiter to use to split the data.
        bins (str): The number of bins to use for the histogram.
        alpha (float): The alpha value for the plot.
        density (bool): If True, normalize the histogram.
        kde (bool): If True, plot a kernel density estimate instead of the histogram.
        kde_bandwidth (str): The bandwidth for KDE.
        save (str): The filename to save the plot to.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
        test (bool): If True, generate random data for testing.
        test_npts (int): The number of points to generate for testing.
        test_ndata (int): The number of datasets to generate for testing.
    """
    if test:
        data = dict()
        fields = ""
        j = 0
        for i in range(test_ndata):
            data[j] = np.random.normal(size=test_npts, loc=i*10, scale=1)
            j += 1
            fields += "y "
        datastr = ""
    else:
        data, datastr, fields = read_data(delimiter, fields, labels)
    fields = fields.strip().split()  # type: ignore
    labels = labels.strip().split()  # type: ignore
    plotid = 0
    ax = plt.gca()
    ax2 = None
    if cumulative and not kde:
        ax2 = ax.twinx()

    for j, field in enumerate(fields):
        if field == "y":
            y = np.float64(data[j])  # type: ignore
            Y.extend(list(y))  # type: ignore
        else:
            continue
        if len(labels) > 0:
            label = labels[plotid]
        else:
            label = None
        if kde:
            if len(y) > 0:
                ymin_val, ymax_val = np.min(y), np.max(y)
                yrange = ymax_val - ymin_val
                if yrange == 0:
                    yrange = 1.0
                x_plot = np.linspace(ymin_val - 0.2 * yrange, ymax_val + 0.2 * yrange, 1000)
                try:
                    bw = float(kde_bandwidth)
                except ValueError:
                    bw = kde_bandwidth
                kde_model = KernelDensity(kernel='gaussian', bandwidth=bw).fit(y[:, None])
                density_vals = np.exp(kde_model.score_samples(x_plot[:, None]))
                line, = plt.plot(x_plot, density_vals, label=label, alpha=alpha)
                plt.fill_between(x_plot, density_vals, alpha=alpha * 0.3, color=line.get_color())
        else:
            n, bins_edges, patches = ax.hist(y, toint(bins), label=label, alpha=alpha, density=density)
            if cumulative:
                if density:
                    y_curve = np.concatenate(([0], np.cumsum(n * np.diff(bins_edges))))
                else:
                    y_curve = np.concatenate(([0], np.cumsum(n)))
                color = 'red'
                if len(patches) > 0:
                    try:
                        color = patches[0].get_facecolor()
                    except (AttributeError, TypeError, IndexError):
                        pass
                import matplotlib.patheffects as path_effects
                ax2.plot(bins_edges, y_curve, color=color,
                         label=f"{label} (cumulative)" if label else "cumulative",
                         path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])
        plotid += 1

    if cumulative and not kde:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if len(h1) > 0 or len(h2) > 0:
            ax.legend(h1 + h2, l1 + l2)
        out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, interactive_plot=False, equal_aspect=equal_aspect, legend=False)
    else:
        out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, interactive_plot=False, equal_aspect=equal_aspect)

@app.command(help="Create a jitter plot from data in standard input.")
def jitter(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]x[/cyan]: x-axis, [cyan]y[/cyan]: y-axis, [cyan]xt[/cyan]: x-tick labels, [cyan]c[/cyan]: color field, [cyan]il[/cyan]: interactive labels", rich_help_panel="Data Input")] = "x y",
    labels: Annotated[str, typer.Option(help="Labels for each dataset", rich_help_panel="Data Input")] = "",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    xjitter: Annotated[float, typer.Option(help="Amount of jitter to add to the x values", rich_help_panel="Jitter Styling")] = 0.1,
    yjitter: Annotated[float, typer.Option(help="Amount of jitter to add to the y values", rich_help_panel="Jitter Styling")] = 0.0,
    size: Annotated[int, typer.Option(help="Size of markers in the plot", rich_help_panel="Plot Appearance")] = 10,
    alpha: Annotated[float, typer.Option(help="Alpha (transparency) value for the plot", rich_help_panel="Plot Appearance")] = 1.0,
    kde: Annotated[bool, typer.Option(help="Use kernel density estimation to color the points", rich_help_panel="Statistical Analysis")] = False,
    kde_subset: Annotated[int, typer.Option(help="Number of points to use for the KDE", rich_help_panel="Statistical Analysis")] = 1000,
    kde_normalize: Annotated[bool, typer.Option(help="Normalize the KDE values", rich_help_panel="Statistical Analysis")] = False,
    cmap: Annotated[str, typer.Option(help="Colormap to use for the plot", rich_help_panel="Plot Appearance")] = "viridis",
    median: Annotated[bool, typer.Option(help="Plot the median of the data", rich_help_panel="Statistical Analysis")] = False,
    median_size: Annotated[int, typer.Option(help="Size of median markers", rich_help_panel="Statistical Analysis")] = 100,
    median_color: Annotated[str, typer.Option(help="Color of median markers", rich_help_panel="Statistical Analysis")] = "black",
    median_marker: Annotated[str, typer.Option(help="Marker for median values", rich_help_panel="Statistical Analysis")] = "_",
    median_sort: Annotated[bool, typer.Option(help="Sort categories by median values", rich_help_panel="Statistical Analysis")] = False,
    quartiles: Annotated[bool, typer.Option(help="Plot first quartile, median, and third quartile as a box", rich_help_panel="Statistical Analysis")] = False,
    highlight_median: Annotated[str, typer.Option(help="Highlight the boxplot with the highest, lowest, or specific x-axis values (e.g. '0 3 highest')", rich_help_panel="Statistical Analysis")] = "",
    # output options
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float | None, typer.Option(help="Minimum x value for the plot", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value for the plot", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value for the plot", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value for the plot", rich_help_panel="Output & Limits")] = None,
    rotation: Annotated[int, typer.Option(help="Rotation of xtick labels in degrees", rich_help_panel="Plot Appearance")] = 45,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot", rich_help_panel="Plot Appearance")] = False,
    cbar_label: Annotated[str | None, typer.Option(help="Label for the colorbar", rich_help_panel="Plot Appearance")] = None,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_npts: Annotated[int, typer.Option(help="Number of points to generate for testing", rich_help_panel="Test Data")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="Number of datasets to generate for testing", rich_help_panel="Test Data")] = 3,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal", rich_help_panel="Output & Limits")] = False,
    pcr: Annotated[bool, typer.Option(help="Perform principal component regression", rich_help_panel="Statistical Analysis")] = False,
):
    """
    Create a jitter plot from data in standard input.

    Args:
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels to use for the data, separated by spaces.
        delimiter (str): The delimiter to use to split the data.
        xjitter (float): The amount of jitter to add to the x values.
        yjitter (float): The amount of jitter to add to the y values.
        size (int): The size of the markers in the plot.
        alpha (float): The alpha value for the plot.
        kde (bool): If True, use kernel density estimation to color the points.
        kde_subset (int): The number of points to use for the KDE.
        kde_normalize (bool): If True, normalize the KDE values.
        cmap (str): The colormap to use for the plot.
        median (bool): If True, plot the median of the data.
        median_size (int): The size of the median markers in the plot.
        median_color (str): The color of the median markers in the plot.
        median_marker (str): The marker to use for the median markers in the plot.
        median_sort (bool): If True, sort by median values.
        quartiles (bool): If True, plot first quartile, median, and third quartile as a box.
        highlight_median (str): Highlight the boxplots with specific x-axis values or extreme medians (e.g. 'highest', 'lowest', or numeric values like '0 3 highest').
        save (str): The filename to save the plot to.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
        rotation (int): The rotation of the xtick labels in degrees.
        colorbar (bool): If True, add a colorbar to the plot.
        cbar_label (str): The label for the colorbar.
        test (bool): If True, generate random data for testing.
        test_npts (int): The number of points to generate for testing.
        test_ndata (int): The number of datasets to generate for testing.
    """
    global X
    global Y
    global INTERACTIVE_LABELS
    if test:
        data = dict()
        j = 0
        k = 0
        fields = ""
        for i in range(test_ndata*2):
            if i % 2 == 0:
                data[i] = np.ones(test_npts) * j
                fields += "x "
                j += 1
            else:
                data[i] = np.random.normal(size=test_npts, loc=k, scale=1)
                fields += "y "
                k += 1
        datastr = ""
    else:
        data, datastr, fields = read_data(delimiter, fields, labels)
    fields = fields.strip().split()  # type: ignore
    labels = labels.strip().split()  # type: ignore
    if highlight_median:
        quartiles = True
    xfields = np.where(np.asarray(fields) == "x")[0]
    yfields = np.where(np.asarray(fields) == "y")[0]
    cfields = np.where(np.asarray(fields) == "c")[0]

    # Pre-pass: Compute medians for each subplot to identify highlight targets
    subplot_y_values = {}
    subplot_x_to_label = {}
    
    temp_plotid = 0
    for xfield, yfield in zip(xfields, yfields):
        sub_id = min(temp_plotid + 1, SUBPLOTS[0] * SUBPLOTS[1])
        if sub_id not in subplot_y_values:
            subplot_y_values[sub_id] = {}
            subplot_x_to_label[sub_id] = {}
            
        x_data = np.float64(data[xfield])
        y_data = np.float64(data[yfield])
        
        if 'xt' in fields:
            xtickslabels = data[fields.index('xt')]
            for xv, label in zip(x_data, xtickslabels):
                subplot_x_to_label[sub_id][xv] = str(label).strip()
                
        for x_val in np.unique(x_data):
            y_vals = y_data[x_data == x_val]
            if len(y_vals) > 0:
                if x_val not in subplot_y_values[sub_id]:
                    subplot_y_values[sub_id][x_val] = []
                subplot_y_values[sub_id][x_val].extend(y_vals)
        temp_plotid += 1

    subplot_target_x_vals = {}
    for sub_id, x_map in subplot_y_values.items():
        medians = {x_val: np.median(y_list) for x_val, y_list in x_map.items()}
        targets = set()
        explicit_targets = set()
        if len(medians) > 0 and highlight_median:
            tokens = highlight_median.strip().split()
            # First pass: identify explicit targets
            for target_token in tokens:
                token_lower = target_token.lower()
                if token_lower not in ("highest", "lowest"):
                    # Match numeric values
                    try:
                        target_val = float(target_token)
                        for x_val in medians.keys():
                            if np.isclose(x_val, target_val):
                                explicit_targets.add(x_val)
                    except ValueError:
                        pass
                    
                    # Match tick labels
                    for x_val, label in subplot_x_to_label.get(sub_id, {}).items():
                        if label.lower() == token_lower:
                            explicit_targets.add(x_val)
            
            # Add explicit targets to final targets
            targets.update(explicit_targets)
            
            # Second pass: handle keywords with exclusion of explicit targets
            for target_token in tokens:
                token_lower = target_token.lower()
                if token_lower in ("highest", "lowest"):
                    remaining_medians = {k: v for k, v in medians.items() if k not in explicit_targets}
                    if len(remaining_medians) > 0:
                        if token_lower == "highest":
                            target_x_val = max(remaining_medians, key=remaining_medians.get)
                            targets.add(target_x_val)
                        elif token_lower == "lowest":
                            target_x_val = min(remaining_medians, key=remaining_medians.get)
                            targets.add(target_x_val)
        subplot_target_x_vals[sub_id] = targets

    kde_y = None
    plotid = 0
    for xfield, yfield in zip(xfields, yfields):
        sub_id = min(plotid + 1, SUBPLOTS[0] * SUBPLOTS[1])
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], sub_id)
        x = np.float64(data[xfield])  # type: ignore
        y = np.float64(data[yfield])  # type: ignore
        c = np.float64(data[cfields[0]]) if len(cfields) > 0 else None  # type: ignore
        if median:
            x = plot_median(x, y,
                            size=median_size,
                            color=median_color,
                            marker=median_marker,
                            median_sort=median_sort)
            data[xfield] = x  # type: ignore
        if quartiles:
            # Calculate quartiles for each x group
            x_unique = np.unique(x)
            target_x_vals = subplot_target_x_vals.get(sub_id, set())
            
            for x_val in x_unique:
                y_vals = y[x == x_val]
                if len(y_vals) > 0:
                    q1 = np.percentile(y_vals, 25)
                    med = np.median(y_vals)
                    q3 = np.percentile(y_vals, 75)
                    
                    is_highlighted = (x_val in target_x_vals)
                    
                    # Plot quartile box
                    if is_highlighted:
                        plt.boxplot(y_vals, positions=[x_val], widths=xjitter, patch_artist=True,
                                    boxprops=dict(facecolor='#ffd700', alpha=0.8, edgecolor='#b8860b', linewidth=2),
                                    medianprops=dict(color='#d62728', linewidth=3),
                                    whiskerprops=dict(color='black', linewidth=1.5),
                                    capprops=dict(color='black', linewidth=1.5), sym="")
                        
                        # Display the median value on the plot
                        y_range = np.max(y) - np.min(y)
                        offset = y_range * 0.02 if y_range > 0 else 0.1
                        med_text = format_nbr(med, precision='.2g')
                        plt.text(x_val, np.max(y_vals) + offset, f"Median: {med_text}", 
                                 ha='center', va='bottom', fontweight='bold', color='black',
                                 zorder=10,
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#b8860b', boxstyle='round,pad=0.2', linewidth=1))
                        
                        # Print the highlighted median value to stdout
                        print(f"Highlighted {highlight_median} median (x={x_val}): {med}")
                    else:
                        plt.boxplot(y_vals, positions=[x_val], widths=xjitter, patch_artist=True,
                                    boxprops=dict(facecolor='lightblue', alpha=0.5),
                                    medianprops=dict(color='red', linewidth=2),
                                    whiskerprops=dict(color='black'),
                                    capprops=dict(color='black'), sym="")
        set_xtick_labels(fields, data, rotation=rotation)
        if "il" in fields:
            INTERACTIVE_LABELS.extend(data[fields.index("il")])
        if kde:
            c = np.zeros_like(y)
            xunique = np.unique(x)
            for xu in track(xunique, description="KDE..."):
                sel = x == xu
                ysel = y[sel]  # type: ignore
                kde_ins = KernelDensity(kernel="gaussian", bandwidth="scott").fit(np.random.choice(ysel, size=min(kde_subset, len(ysel)))[:, None])  # type: ignore
                kde_y = np.exp(kde_ins.score_samples(ysel[:, None]))
                if kde_normalize:
                    kde_y -= kde_y.min()
                    kde_y /= kde_y.max()
                c[sel] = kde_y
            c = np.asarray(c)
        x += np.random.uniform(size=x.shape, low=-xjitter/2, high=xjitter/2)
        y += np.random.uniform(size=y.shape, low=-yjitter/2, high=yjitter/2)
        X.extend(list(x))
        Y.extend(list(y))
        plt.scatter(x, y, c=c, s=size, alpha=alpha, cmap=cmap)
        if pcr:
            do_pcr(x, y)
        _apply_axis_tick_formats(plt.gca(), x, y) # Apply tick formats after plotting
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, equal_aspect=equal_aspect)

def plot_median(x, y, size=100, color="black", marker="_", median_sort: bool = False):
    """
    Plot the median of the data.

    Args:
        x (array-like): The x values.
        y (array-like): The y values.
        size (int): The size of the median markers.
        color (str): The color of the median markers.
        marker (str): The marker to use for the median markers.
        median_sort (bool): If True, sort by median values.

    Returns:
        array-like: The sorted x values if median_sort is True, otherwise the original x values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    xunique = np.unique(x)
    ymedians = []
    for i in range(len(xunique)):
        xsel = x == xunique[i]
        ysel = y[xsel]
        if len(ysel) > 0:
            ymedian = np.median(ysel)
            ymedians.append(ymedian)
    sorter = None
    if median_sort:
        sorter = np.argsort(ymedians)
        ymedians = np.asarray(ymedians)[sorter]
        # repeat the sorter for the x values
        mapper = dict(zip(sorter, xunique))
        x = np.asarray([mapper[xi] for xi in x])
        x = np.float64(x)
    plt.scatter(xunique, ymedians, color=color, marker=marker, s=size, label="median", zorder=100)
    return x

@app.command(help="Create a ROC curve from data in standard input.")
def roc(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]y[/cyan]: raw values, [cyan]a[/cyan]: active (1) or inactive (0)", rich_help_panel="Data Input")] = "y a",
    labels: Annotated[str, typer.Option(help="Labels for each dataset", rich_help_panel="Data Input")] = "",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float, typer.Option(help="Minimum x value for the plot", rich_help_panel="Output & Limits")] = 0.0,
    xmax: Annotated[float, typer.Option(help="Maximum x value for the plot", rich_help_panel="Output & Limits")] = 1.0,
    ymin: Annotated[float, typer.Option(help="Minimum y value for the plot", rich_help_panel="Output & Limits")] = 0.0,
    ymax: Annotated[float, typer.Option(help="Maximum y value for the plot", rich_help_panel="Output & Limits")] = 1.0,
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_npts: Annotated[int, typer.Option(help="Number of points to generate for testing", rich_help_panel="Test Data")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="Number of datasets to generate for testing", rich_help_panel="Test Data")] = 2,
    highlight_closest: Annotated[bool, typer.Option(help="Highlight the point closest to (0,1) and show its cutoff", rich_help_panel="Statistical Analysis")] = False,
):
    """
    Create a ROC curve from data in standard input.

    Args:
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels to use for the data, separated by spaces.
        delimiter (str): The delimiter to use to split the data.
        test (bool): If True, generate random data for testing.
        save (str): The filename to save the plot to.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
        highlight_closest (bool): If True, highlight the closest point to (0,1) and display its cutoff value.
    """
    global X
    global Y
    global INTERACTIVE_LABELS
    if test:
        data = dict()
        fields = ""
        labels = ""
        j = 0
        for test_dataidx in range(test_ndata):
            data[j] = np.random.normal(size=test_npts, loc=0, scale=1)
            fields += "y "
            j += 1
            data[j] = np.random.randint(0, 2, size=test_npts)
            fields += "a "
            j += 1
            labels += f"DS{test_dataidx} "
        datastr = ""
        print(f"{fields=}")
    else:
        data, datastr, fields = read_data(delimiter=delimiter, fields=fields, labels=labels)
    fields = fields.strip().split()  # type: ignore
    labels = labels.strip().split()  # type: ignore
    yfields = np.where(np.asarray(fields) == "y")[0]
    afields = np.where(np.asarray(fields) == "a")[0]
    for plotid, (yfield, afield) in enumerate(zip(yfields, afields)):
        y = np.float64(data[yfield])  # type: ignore
        a = np.int_(data[afield])  # type: ignore
        active_values = y[a == 1]  # type: ignore
        inactive_values = y[a == 0]  # type: ignore
        x, y, auc, pROC_auc, thresholds = ROC(active_values, inactive_values)
        X.extend(list(x))
        Y.extend(list(y))
        label = labels[plotid] if len(labels) > 0 else None
        if label is None or label == "":
            label = f"AUC={auc:.2f}, pROC={pROC_auc:.2f}"
            labels.append(label)  # type: ignore
        else:
            label += f" (AUC={auc:.2f}, pROC={pROC_auc:.2f})"
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))
        plt.plot(x, y, label=label)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        if highlight_closest:
            # Find the point closest to (0,1)
            x = np.array(x)
            y = np.array(y)
            distances = np.sqrt((x - 0)**2 + (y - 1)**2)
            closest_idx = np.argmin(distances)
            closest_x = x[closest_idx]
            closest_y = y[closest_idx]
            closest_threshold = thresholds[closest_idx]
            
            # Highlight the closest point with a different color and marker
            plt.scatter(closest_x, closest_y, s=100, zorder=5, marker='o', edgecolors='black', linewidth=0.5, color='red')
            
            # Add the cutoff value to the legend by creating a new legend entry
            if label:
                # Create a new label with cutoff information
                cutoff_label = f"{label} (Cutoff: {format_nbr(closest_threshold, precision='.2g')})"
                # Remove the old line and add a new one with the cutoff label
                old_line = plt.gca().lines[-1]
                old_line.set_label(cutoff_label)
        
        if SUBPLOTS[0]*SUBPLOTS[1] > 1:
            set_limits(xmin, xmax, ymin, ymax)
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.title(label)
            plt.plot([xmin, xmax], [ymin, ymax], 'k--', label="Random")
        _apply_axis_tick_formats(plt.gca(), x, y) # Apply tick formats after plotting
    if SUBPLOTS[0]*SUBPLOTS[1] == 1:
        plt.plot([xmin, xmax], [ymin, ymax], 'k--', label="Random")
    else:
        labels = []  # type: ignore
    out(save=save, datastr=datastr, labels=labels, colorbar=None, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cbar_label=None, equal_aspect=True)

@app.command(help="Create a t-SNE plot from data in standard input.")
def tsne(
    perplexity: Annotated[float, typer.Option(help="Perplexity relates to the number of nearest neighbors. Consider values between 5 and 50.", rich_help_panel="t-SNE Parameters")] = 30.0,
    early_exaggeration: Annotated[float, typer.Option(help="Controls how tightly clusters are bunched together.", rich_help_panel="t-SNE Parameters")] = 12.0,
    learning_rate: Annotated[float, typer.Option(help="Learning rate (usually between 10 and 1000).", rich_help_panel="t-SNE Parameters")] = 200.0,
    n_iter: Annotated[int, typer.Option(help="Maximum number of iterations (at least 250).", rich_help_panel="t-SNE Parameters")] = 1000,
    metric: Annotated[str, typer.Option(help="Distance metric (euclidean, cosine, etc.)", rich_help_panel="t-SNE Parameters")] = "euclidean",
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    npy: Annotated[str, typer.Option(help="Load data from a .npy file", rich_help_panel="Data Input")] = "",
    npz: Annotated[str, typer.Option(help="Load data from a .npz file", rich_help_panel="Data Input")] = "",
    data_key: Annotated[str, typer.Option(help="Key for data in .npz file", rich_help_panel="Data Input")] = "data",
    labels_key: Annotated[str, typer.Option(help="Key for labels in .npz file", rich_help_panel="Data Input")] = "",
    ilabels_key: Annotated[str, typer.Option(help="Key for interactive labels in .npz file", rich_help_panel="Data Input")] = "",
    legend: Annotated[bool, typer.Option(help="Add a legend to the plot", rich_help_panel="Plot Appearance")] = True,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot", rich_help_panel="Plot Appearance")] = False,
    cmap: Annotated[str, typer.Option(help="Colormap to use for the plot", rich_help_panel="Plot Appearance")] = "viridis",
    size: Annotated[int, typer.Option(help="Size of markers in the plot", rich_help_panel="Plot Appearance")] = 10,
    alpha: Annotated[float, typer.Option(help="Transparency of markers", rich_help_panel="Plot Appearance")] = 1.0,
    xmin: Annotated[float | None, typer.Option(help="Minimum x value", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value", rich_help_panel="Output & Limits")] = None,
):
    """
    Create a t-SNE plot from data in standard input.

    Args:
        perplexity (float): The perplexity is related to the number of nearest neighbors.
        early_exaggeration (float): Controls how tightly clusters are bunched together.
        learning_rate (float): The learning rate for t-SNE.
        n_iter (int): Maximum number of iterations for the optimization.
        metric (str): The metric to use to compute distance in high dimensional space.
        test (bool): If True, generate random data for testing.
        save (str): The filename to save the plot to.
        npy (str): The filename to load data from a numpy file.
        npz (str): The filename to load data from a numpy file (compressed).
        data_key (str): The key to use to load data from the npz file.
        labels_key (str): The key to use to load labels from the npz file.
        ilabels_key (str): The key to use to load interactive labels from the npz file.
        legend (bool): If True, add a legend to the plot.
        colorbar (bool): If True, add a colorbar to the plot.
        cmap (str): The colormap to use for the plot.
        size (int): The size of the markers in the plot.
        alpha (float): The transparency of the markers in the plot.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
    """
    global X
    global Y
    global INTERACTIVE_LABELS

    labels = None
    if test:
        data = np.random.normal(loc=(0, 0, 0), size=(100, 3))
        data = np.concatenate((data, np.random.normal(loc=(1, 1, 1), size=(100, 3))), axis=0)
    elif npy != "":
        data = np.load(npy)
    elif npz != "":
        dataz = np.load(npz)
        print(f"{dataz.files=}")
        data = dataz[data_key]
        if labels_key != "":
            labels = dataz[labels_key]
        if ilabels_key != "":
            ilabels = dataz[ilabels_key]
    else:
        print("No data provided, use --test, --npy or --npz")
        sys.exit(1)
    print(f"{data.shape=}")
    print(f"{data.min()=}")
    print(f"{data.max()=}")

    # Initialize TSNE model
    model = TSNE(
        n_components=2, # t-SNE typically outputs 2 or 3 dimensions
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        init='pca', # Recommended initialization for t-SNE
        random_state=42 # for reproducibility
    )

    embedding = model.fit_transform(data)

    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=size, cmap=cmap, alpha=alpha) # type: ignore
        X.extend(list(embedding[:, 0])) # type: ignore
        Y.extend(list(embedding[:, 1])) # type: ignore
        if ilabels_key != "" and ilabels is not None:
            INTERACTIVE_LABELS = ilabels
        elif ilabels_key == "":
            INTERACTIVE_LABELS = [] # Clear if not provided
    else:
        for label in np.unique(labels):
            sel = labels == label
            x = embedding[sel, 0] # type: ignore
            y = embedding[sel, 1] # type: ignore
            plt.scatter(x, y, s=size, cmap=cmap, alpha=alpha, label=label)
            X.extend(list(x))
            Y.extend(list(y))
            if ilabels_key != "" and ilabels is not None:
                INTERACTIVE_LABELS.extend(list(ilabels[sel])) # type: ignore
            elif ilabels_key == "":
                INTERACTIVE_LABELS.extend(list(labels[sel]))
    _apply_axis_tick_formats(plt.gca(), embedding[:, 0], embedding[:, 1]) # Apply tick formats
    out(save=save, datastr="", labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, legend=legend)

@app.command(help="Create a UMAP plot from data in standard input.")
def umap(
    n_neighbors: Annotated[int, typer.Option(help="Size of local neighborhood for manifold approximation.", rich_help_panel="UMAP Parameters")] = 15,
    min_dist: Annotated[float, typer.Option(help="Effective minimum distance between embedded points.", rich_help_panel="UMAP Parameters")] = 0.1,
    metric: Annotated[str, typer.Option(help="Distance metric (euclidean, cosine, etc.)", rich_help_panel="UMAP Parameters")] = "euclidean",
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    npy: Annotated[str, typer.Option(help="Load data from a .npy file", rich_help_panel="Data Input")] = "",
    npz: Annotated[str, typer.Option(help="Load data from a .npz file", rich_help_panel="Data Input")] = "",
    data_key: Annotated[str, typer.Option(help="Key for data in .npz file", rich_help_panel="Data Input")] = "data",
    labels_key: Annotated[str, typer.Option(help="Key for labels in .npz file", rich_help_panel="Data Input")] = "",
    ilabels_key: Annotated[str, typer.Option(help="Key for interactive labels in .npz file", rich_help_panel="Data Input")] = "",
    legend: Annotated[bool, typer.Option(help="Add a legend to the plot", rich_help_panel="Plot Appearance")] = True,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot", rich_help_panel="Plot Appearance")] = False,
    cmap: Annotated[str, typer.Option(help="Colormap to use for the plot", rich_help_panel="Plot Appearance")] = "viridis",
    size: Annotated[int, typer.Option(help="Size of markers in the plot", rich_help_panel="Plot Appearance")] = 10,
    alpha: Annotated[float, typer.Option(help="Transparency of markers", rich_help_panel="Plot Appearance")] = 1.0,
    xmin: Annotated[float | None, typer.Option(help="Minimum x value", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value", rich_help_panel="Output & Limits")] = None,
):
    """
    Create a UMAP plot from data in standard input.

    Args:
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
        min_dist (float): The effective minimum distance between embedded points.
        metric (str): The metric to use to compute distance in high dimensional space.
        test (bool): If True, generate random data for testing.
        save (str): The filename to save the plot to.
        npy (str): The filename to load data from a numpy file.
        npz (str): The filename to load data from a numpy file (compressed).
        data_key (str): The key to use to load data from the npz file.
        labels_key (str): The key to use to load labels from the npz file.
        ilabels_key (str): The key to use to load interactive labels from the npz file.
        legend (bool): If True, add a legend to the plot.
        colorbar (bool): If True, add a colorbar to the plot.
        cmap (str): The colormap to use for the plot.
        size (int): The size of the markers in the plot.
        alpha (float): The transparency of the markers in the plot.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
    """
    global X
    global Y
    global INTERACTIVE_LABELS
    import umap

    labels = None
    if test:
        data = np.random.normal(loc=(0, 0, 0), size=(100, 3))
        data = np.concatenate((data, np.random.normal(loc=(1, 1, 1), size=(100, 3))), axis=0)
    elif npy != "":
        data = np.load(npy)
    elif npz != "":
        dataz = np.load(npz)
        print(f"{dataz.files=}")
        data = dataz[data_key]
        if labels_key != "":
            labels = dataz[labels_key]
        if ilabels_key != "":
            ilabels = dataz[ilabels_key]
    else:
        print("No data provided, use --test, --npy or --npz")
        sys.exit(1)
    print(f"{data.shape=}")
    print(f"{data.min()=}")
    print(f"{data.max()=}")
    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = mapper.fit_transform(data)
    # umap.plot.points(mapper, values=r_orig)
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=size, cmap=cmap, alpha=alpha)  # type: ignore
        X.extend(list(embedding[:, 0]))  # type: ignore
        Y.extend(list(embedding[:, 1]))  # type: ignore
        if ilabels_key != "" and ilabels is not None:
            INTERACTIVE_LABELS = ilabels
        elif ilabels_key == "":
            INTERACTIVE_LABELS = [] # Clear if not provided
    else:
        for label in np.unique(labels):
            sel = labels == label
            x = embedding[sel, 0]  # type: ignore
            y = embedding[sel, 1]  # type: ignore
            plt.scatter(x, y, s=size, cmap=cmap, alpha=alpha, label=label)
            X.extend(list(x))
            Y.extend(list(y))
            if ilabels_key != "" and ilabels is not None:
                INTERACTIVE_LABELS.extend(list(ilabels[sel]))  # type: ignore
            elif ilabels_key == "":
                INTERACTIVE_LABELS.extend(list(labels[sel]))
    _apply_axis_tick_formats(plt.gca(), embedding[:, 0], embedding[:, 1]) # Apply tick formats
    out(save=save, datastr="", labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, legend=legend)

@app.command(help="Read metadata from a PNG file.")
def read_metadata(filename: Annotated[str, typer.Option(..., help="PNG filename to read metadata from", rich_help_panel="Data Input")]):
    """
    Read metadata from a PNG file.

    Args:
        filename (str): The filename to read the metadata from.

    Returns:
        str: The metadata string.
    """
    im = Image.open(filename)
    im.load()
    datastr = f'#hostname:{im.info["hostname"]}\n'
    datastr += f'#cwd:{im.info["cwd"]}\n'
    if "subsampling" in im.info:
        datastr += f'#subsampling:{im.info["subsampling"]}\n'
    if "labels" in im.info:
        datastr += f'#labels:{im.info["labels"]}\n'
    datastr += im.info["data"]
    print(datastr)
    return datastr

def do_pcr(x, y):
    """
    Perform principal component regression.

    Args:
        x (array-like): The x values.
        y (array-like): The y values.
    """
    print("######## PCR ########")
    X = np.stack([x, y]).T
    eigenvalues, eigenvectors, center, anglex = pca(X)
    a = eigenvectors[1, 0] / eigenvectors[0, 0]
    print(f"{a=}")
    b = center[1] - a * center[0]
    print(f"{b=}")
    xm = x.min()
    ym = a*xm+b
    xM = x.max()
    yM = a*xM+b
    plt.plot([xm, xM], [ym, yM], zorder=100, color="red")
    v2_x = [center[0], center[0]+eigenvectors[0, 1]*np.sqrt(eigenvalues[1])]
    v2_y = [center[1], center[1]+eigenvectors[1, 1]*np.sqrt(eigenvalues[1])]
    # plt.plot(v2_x, v2_y)
    explained_variance = eigenvalues[0]/eigenvalues.sum()
    print(f"{explained_variance=}")
    pearson = np.sum((x-x.mean())*(y-y.mean())) / (np.sqrt(np.sum((x-x.mean())**2)) * np.sqrt(np.sum((y-y.mean())**2)))
    print(f"{pearson=}")
    spearman = scipy.stats.spearmanr(a=x, b=y).statistic
    print(f"{spearman=}")
    R_squared = 1 - np.sum((y-a*x+b)**2) / np.sum((y-y.mean())**2)
    print(f"{R_squared=}")
    regstr = "y="
    if f"{a:.1g}" == "1":
        regstr+="x"
    else:
        regstr+=f"{a:.1g}x"
    if float(f"{b:.1g}")<0:
        regstr+=f"{b:.1g}"
    elif float(f"{b:.1g}")>0:
        regstr+=f"+{b:.1g}"
    else:
        pass
    plt.title(f"ρ={format_nbr(pearson)}|ρₛ={format_nbr(spearman)}\n{regstr}")
    # annotation=f"{a=:.2g}\n{b=:.2g}\n{explained_variance=:.2g}\n{pearson=:.2g}\n{R_squared=:.2g}"
    # bbox = dict(boxstyle ="round", fc ="0.8")
    # plt.annotate(annotation, (v2_x[1], v2_y[1]), bbox=bbox, fontsize="xx-small")
    print("#####################")

def format_nbr(x, precision='.1f'):
    """
    Format a number.

    Args:
        x (float): The number to format.
        precision (str): The precision to use for formatting.

    Returns:
        str: The formatted number.
    """
    if float(format(x, precision)) == round(x):
        return f'{round(x)}'
    else:
        return format(x, precision)

def _mark_global_minimum(x_data, y_data, label_prefix, color):
    """
    Helper function to mark and annotate the global minimum of a curve.
    """
    min_y_abs = np.min(y_data)
    min_x_abs = x_data[np.argmin(y_data)]

    # Plot marker for absolute minimum
    plt.scatter(min_x_abs, min_y_abs, marker='v', color=color, s=150, zorder=7, edgecolor='black', linewidth=0.8)
    # Add text label for absolute minimum
    plt.annotate(f"{label_prefix} min:\n({format_nbr(min_x_abs)}, {format_nbr(min_y_abs)})",
                 (min_x_abs, min_y_abs),
                 textcoords="offset points", xytext=(0,-35), ha='center', fontsize=8, color=color,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=color, lw=0.5))

def pca(X, outfilename=None):
    """
    Perform principal component analysis.

    Args:
        X (array-like): The data to analyze.
        outfilename (str): The filename to save the results to.

    Returns:
        tuple: A tuple containing the eigenvalues, eigenvectors, center, and angle of the first eigenvector.
    """
    center = X.mean(axis=0)
    cov = (X - center).T.dot(X - center) / X.shape[0]
    eigenvalues, eigenvectors = linalg.eigh(cov)
    sorter = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[sorter], eigenvectors[:, sorter]
    print(f"{eigenvectors.shape=}")
    dotprod1 = eigenvectors[:, 0].dot(np.asarray([1, 0]))
    dotprod2 = eigenvectors[:, 1].dot(np.asarray([0, 1]))
    eigenvectors[:, 0] *= np.sign(dotprod1)
    eigenvectors[:, 1] *= np.sign(dotprod2)
    anglesign = np.sign(eigenvectors[:, 0].dot(np.asarray([0, 1])))
    anglex = anglesign * np.rad2deg(
        np.arccos(eigenvectors[:, 0].dot(np.asarray([1, 0]))))
    print(f"{anglex=:.4g}")
    # angley = np.rad2deg(np.arccos(eigenvectors[:, 1].dot(np.asarray([0, 1]))))
    # print(f"{angley=:.4g}")
    if outfilename is not None:
        np.savez(outfilename,
                 eigenvalues=eigenvalues,
                 eigenvectors=eigenvectors)
    return eigenvalues, eigenvectors, center, anglex

@app.command(help="Create a chord diagram from data in standard input.")
def chord_diagram(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]d[/cyan]: matrix values, [cyan]r[/cyan]: row labels, [cyan]c[/cyan]: column labels", rich_help_panel="Data Input")] = "d r c",
    labels: Annotated[str, typer.Option(help="Labels for the data", rich_help_panel="Data Input")] = "",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
 ):
    """
    Create a chord diagram from data in standard input.

    Args:
        fields (str): Specifies the columns for data components ('d'), row labels ('r'), and column labels ('c').
        labels (str): The labels to use for the data.
        delimiter (str): The delimiter to use to split the data.
        test (bool): If True, generate random data for testing.
        save (str): The filename to save the plot to.
    """
    try:
        from pycirclize import Circos
    except ImportError:
        print("You need to install pyCirclize")
        print("see: https://github.com/moshi4/pyCirclize")
        print("You can run:")
        print("make pycirclize")
        sys.exit(2)
    import pandas as pd
    if test:
        # Create matrix dataframe (3 x 6)
        row_names = ["F1", "F2", "F3"]
        col_names = ["T1", "T2", "T3", "T4", "T5", "T6"]
        matrix_data = [
            [10, 16, 7, 7, 10, 8],
            [4, 9, 10, 12, 12, 7],
            [17, 13, 7, 4, 20, 4],
        ]
        matrix_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)

        # Initialize Circos instance for chord diagram plot
        circos = Circos.chord_diagram(
            matrix_df,
            space=5,
            cmap="tab10",
            label_kws=dict(size=12),
            link_kws=dict(ec="black", lw=0.5, direction=1),
        )
        fig = circos.plotfig()
        if save == "":
            plt.show()
        else:
            fig.savefig(save)
            ext = os.path.splitext(save)[1]
            print(f"{ext=}")
            print(f"{save=}")
            if ext == ".png":
                add_metadata(save, "", labels=None)
    else:
        data, datastr, fields = read_data(delimiter, fields, labels)
        fields = fields.strip().split()  # type: ignore
        labels = labels.strip().split()  # type: ignore
        data_fields = np.where(np.asarray(fields) == "d")[0]
        row_names = data[fields.index('r')]
        col_names = data[fields.index('c')]
        rows = np.unique(row_names)
        cols = np.unique(col_names)
        nrows = len(rows)
        ncols = len(cols)
        rows_mapping = dict(zip(rows, range(nrows)))
        cols_mapping = dict(zip(cols, range(ncols)))
        matrix_data = np.zeros((nrows, ncols))
        datavalues = data[data_fields[0]]
        for i in range(len(datavalues)):
            v = datavalues[i]
            row_name = row_names[i]
            col_name = col_names[i]
            i = rows_mapping[row_name]
            j = cols_mapping[col_name]
            matrix_data[i, j] = v
        matrix_df = pd.DataFrame(matrix_data, index=rows, columns=cols)
        print(matrix_df)
        # Initialize Circos instance for chord diagram plot
        circos = Circos.chord_diagram(
            matrix_df,
            space=5,
            cmap="tab10",
            label_kws=dict(size=12),
            link_kws=dict(ec="black", lw=0.5, direction=1),
        )
        fig = circos.plotfig()
        out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=None, xmax=None, ymin=None, ymax=None, interactive_plot=False)

@app.command(help="Create a Venn diagram from data in standard input or generated test data.")
def venn_diagram(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]d[/cyan]: set components, [cyan]l[/cyan]: set labels", rich_help_panel="Data Input")] = "d l",
    labels_fill: Annotated[str, typer.Option(help="Comma-separated fill options: 'number', 'logic', 'percent', 'elements'", rich_help_panel="Venn Styling")] = "number",
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_ndata: Annotated[int, typer.Option(help="Number of sets to generate for testing (2 to 6)", rich_help_panel="Test Data")] = 3,
    test_npts: Annotated[int, typer.Option(help="Number of points in each test set", rich_help_panel="Test Data")] = 10,
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    colors: Annotated[str, typer.Option(help="Comma-separated list of colors", rich_help_panel="Venn Styling")] = "",
    figsize: Annotated[str, typer.Option(help="Figure size in inches (e.g., '9 7')", rich_help_panel="Venn Styling")] = "",
    dpi: Annotated[int, typer.Option(help="Resolution of the figure in DPI", rich_help_panel="Venn Styling")] = 96,
    fontsize: Annotated[int, typer.Option(help="Font size for labels", rich_help_panel="Venn Styling")] = 14,
    sortkey: Annotated[int, typer.Option(help="Key to sort elements in 'elements' fill option", rich_help_panel="Venn Styling")] = 0,
):
    """
    Create a Venn diagram from data in standard input or generated test data.

    Args:
        fields (str): Specifies the columns for data components ('d') and set labels ('l').
        labels_fill (str): Comma-separated options for filling Venn diagram labels (e.g., 'number', 'logic', 'percent', 'elements').
        save (str): The filename to save the plot to.
        test (bool): If True, generate random data for testing.
        test_ndata (int): The number of sets to generate for testing (2 to 6).
        test_npts (int): The number of points in each test set for testing.
        delimiter (str): The delimiter to use to split the data.
        colors (str): Comma-separated list of colors for the sets.
        figsize (str): Figure size in inches (e.g., '9 7').
        dpi (int): Resolution of the figure in dots per inch.
        fontsize (int): Font size for labels.
    """
    try:
        import venn
    except ImportError:
        print("You need to install pyvenn")
        print("See: https://github.com/tctianchi/pyvenn")
        print("You can run:")
        print("pip install pyvenn")
        sys.exit(1)

    def print_labels_dict(labels_dict, names_for_venn):
        """
        Prints the content of the labels dictionary for the Venn diagram.

        Args:
            labels_dict (dict): Dictionary where keys are logic strings (e.g., '100')
                                and values are formatted labels (e.g., "Set A: 5 elements").
            names_for_venn (list): List of set names, used for initial mapping.
        """
        nset = len(names_for_venn)
        for i in range(nset):
            logic = ['0'] * nset
            logic[i] = '1'
            logic = ''.join(logic)
            print(f"{logic}={names_for_venn[i]}")
        print("--")
        for logic in labels_dict:
            value = labels_dict[logic].replace("\n", ",").strip()
            if logic+":" in value:
                value = value.replace(logic+": ", "")
            print(f"{logic}={value}")
            print("--")

    venn_options = {}
    if colors:
        venn_options['colors'] = colors.split(',')
    if figsize:
        xaspect, yaspect = figsize.split()
        venn_options['figsize'] = (float(xaspect), float(yaspect))
    venn_options['dpi'] = dpi
    venn_options['fontsize'] = fontsize
    labels_fill_list = labels_fill.split(',')

    if test:
        if not (2 <= test_ndata <= 6):
            print(f"Warning: test_ndata must be between 2 and 6. Using 3 sets for test.")
            test_ndata = 3

        sets_data = []
        names_for_venn = []
        for i in range(test_ndata):
            start = i * (test_npts // 2)
            end = start + test_npts
            sets_data.append(set(range(start, end)))
            names_for_venn.append(f'Set {chr(65 + i)}')

        labels_dict = venn.get_labels(sets_data, fill=labels_fill_list, sortkey=sortkey)
        venn_func = getattr(venn, f"venn{test_ndata}")
        fig, ax = venn_func(labels_dict, names=names_for_venn, **venn_options)
        out(save=save, datastr="", labels=names_for_venn, colorbar=False, xmin=None, xmax=None, ymin=None, ymax=None, interactive_plot=False, legend=False)
    else:
        data_dict, datastr, parsed_fields_str = read_data(delimiter, fields, "")
        parsed_fields = parsed_fields_str.strip().split()

        d_field_indices = np.where(np.asarray(parsed_fields) == "d")[0]
        l_field_indices = np.where(np.asarray(parsed_fields) == "l")[0]

        if len(d_field_indices) == 0 or len(l_field_indices) == 0:
            print("Error: 'd' (data component) and 'l' (set label) fields are required for venn_diagram.")
            sys.exit(1)

        set_components = [str(x) for x in data_dict[d_field_indices[0]]]
        set_labels_raw = [str(x) for x in data_dict[l_field_indices[0]]]

        sets_grouped_by_name = defaultdict(list)
        for component, label in zip(set_components, set_labels_raw):
            sets_grouped_by_name[label].append(component)

        names_for_venn = sorted(sets_grouped_by_name.keys())
        sets_data = [set(sets_grouped_by_name[name]) for name in names_for_venn]

        num_sets = len(sets_data)

        if not (2 <= num_sets <= 6):
            print(f"Error: Venn diagrams currently support 2 to 6 sets. Found {num_sets} sets.")
            sys.exit(1)

        labels_dict = venn.get_labels(sets_data, fill=labels_fill_list, sortkey=sortkey)
        print_labels_dict(labels_dict, names_for_venn)

        venn_func = getattr(venn, f"venn{num_sets}")
        fig, ax = venn_func(labels_dict, names=names_for_venn, **venn_options)

        out(save=save, datastr=datastr, labels=names_for_venn, colorbar=False, xmin=None, xmax=None, ymin=None, ymax=None, interactive_plot=False, legend=False)

@app.command(help="Create a heatmap from data in standard input.")
def heatmap(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]v[/cyan]: value, [cyan]r[/cyan]: row label, [cyan]c[/cyan]: column label", rich_help_panel="Data Input")] = "v r c",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    cmap: Annotated[str, typer.Option(help="Colormap to use for the heatmap", rich_help_panel="Heatmap Styling")] = "viridis",
    cbar_label: Annotated[str | None, typer.Option(help="Label for the colorbar", rich_help_panel="Heatmap Styling")] = None,
    rotation: Annotated[int, typer.Option(help="Rotation for x-tick labels", rich_help_panel="Heatmap Styling")] = 90,
    # output options
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float | None, typer.Option(help="Minimum x value (ignored if not applicable)", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value (ignored if not applicable)", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value (ignored if not applicable)", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value (ignored if not applicable)", rich_help_panel="Output & Limits")] = None,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal", rich_help_panel="Heatmap Styling")] = False,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_rows: Annotated[int, typer.Option(help="Number of rows for test data", rich_help_panel="Test Data")] = 5,
    test_cols: Annotated[int, typer.Option(help="Number of columns for test data", rich_help_panel="Test Data")] = 7,
    matrix_order: Annotated[bool, typer.Option(help="Plot in matrix order (rows and columns instead of x,y)", rich_help_panel="Heatmap Styling")] = False,
    fontsize: Annotated[int, typer.Option(help="Font size for tick labels", rich_help_panel="Heatmap Styling")] = 10,
    display_values: Annotated[bool, typer.Option(help="Display values on the heatmap cells", rich_help_panel="Heatmap Styling")] = False,
):
    """
    Create a heatmap from data in standard input.

    The input data should contain a value, a row label, and a column label for each cell.
    Example input:
    1.2 R1 C1
    3.4 R1 C2
    ...
    5.6 R2 C1
    """
    if test:
        num_rows = test_rows
        num_cols = test_cols
        values = np.arange(num_rows * num_cols)
        print(f"Random labels are defined from R1 to R{num_rows} and C1 to C{num_cols} for rows and columns respectively")
        row_labels = np.random.choice([f"R{i+1}" for i in range(num_rows)], replace=False, size=num_rows)
        col_labels = np.random.choice([f"C{j+1}" for j in range(num_cols)], replace=False, size=num_cols)
        row_labels = [i for i in row_labels for _ in range(num_cols)]
        col_labels = [j for _ in range(num_rows) for j in col_labels]

        data_dict = {
            0: values.astype(str).tolist(),
            1: row_labels,
            2: col_labels
        }
        datastr = ""
        fields_str = "v r c"
    else:
        data_dict, datastr, fields_str = read_data(delimiter, fields, "")

    parsed_fields = fields_str.strip().split()

    v_field_idx = np.where(np.asarray(parsed_fields) == "v")[0]
    r_field_idx = np.where(np.asarray(parsed_fields) == "r")[0]
    c_field_idx = np.where(np.asarray(parsed_fields) == "c")[0]

    if len(v_field_idx) == 0:
        print("Error: 'v' (value) field is required for heatmap.")
        sys.exit(1)
    if len(r_field_idx) == 0:
        print("Error: 'r' (row label) field is required for heatmap.")
        sys.exit(1)
    if len(c_field_idx) == 0:
        print("Error: 'c' (column label) field is required for heatmap.")
        sys.exit(1)

    values = np.float64(data_dict[v_field_idx[0]])
    row_labels_raw = data_dict[r_field_idx[0]]
    col_labels_raw = data_dict[c_field_idx[0]]

    # unique_row_labels = sorted(list(np.unique(row_labels_raw)))
    # unique_col_labels = sorted(list(np.unique(col_labels_raw)))
    unique_row_labels = list()
    for e in row_labels_raw:
        if e not in unique_row_labels:
            unique_row_labels.append(e)
    unique_col_labels = list()
    for e in col_labels_raw:
        if e not in unique_col_labels:
            unique_col_labels.append(e)

    row_to_idx = {label: i for i, label in enumerate(unique_row_labels)}
    col_to_idx = {label: i for i, label in enumerate(unique_col_labels)}

    nrows = len(unique_row_labels)
    ncols = len(unique_col_labels)

    heatmap_matrix = np.full((nrows, ncols), np.nan)
    count_matrix = np.zeros((nrows, ncols), dtype=int)

    for i in track(range(len(values)), description="Building heatmap matrix..."):
        r_label = row_labels_raw[i]
        c_label = col_labels_raw[i]
        val = values[i]

        row_idx = row_to_idx[r_label]
        col_idx = col_to_idx[c_label]
        if np.isnan(heatmap_matrix[row_idx, col_idx]):
            heatmap_matrix[row_idx, col_idx] = val
        else:
            heatmap_matrix[row_idx, col_idx] += val
        count_matrix[row_idx, col_idx] += 1

    heatmap_matrix/=count_matrix
    print(f"Ratio of averaged elements in the matrix: {(count_matrix>1).sum()/count_matrix.size}")
    print(f"Mean number of values averaged in the matrix: {count_matrix.mean()}")
    print(f"Standard deviation of the number of values averaged in the matrix {count_matrix.std()}")

    # Check if the matrix is symmetric:
    if heatmap_matrix.shape == heatmap_matrix.T.shape:
        if np.isclose(heatmap_matrix, heatmap_matrix.T).all():
            print("The matrix is SYMMETRIC")
        else:
            print("The matrix is NOT symmetric")
    ha = 'right' if rotation != 0 else 'center'
    if matrix_order:
        im = plt.imshow(heatmap_matrix, cmap=cmap, origin='upper', aspect='auto')
        plt.xticks(np.arange(ncols), unique_col_labels, rotation=rotation, ha=ha, rotation_mode='anchor', fontsize=fontsize)
        plt.yticks(np.arange(nrows), unique_row_labels, fontsize=fontsize)
    else:
        im = plt.imshow(heatmap_matrix.T, cmap=cmap, origin='lower', aspect='auto')
        plt.xticks(np.arange(nrows), unique_row_labels, rotation=rotation, ha=ha, rotation_mode='anchor', fontsize=fontsize)
        plt.yticks(np.arange(ncols), unique_col_labels, fontsize=fontsize)

    if display_values:
        # Loop over data dimensions and create text annotations.
        for i in range(heatmap_matrix.shape[0]):
            for j in range(heatmap_matrix.shape[1]):
                val = heatmap_matrix[i, j]
                if not np.isnan(val):
                    # Determine text color based on background brightness
                    # This is a simple heuristic and might need adjustment
                    bg_brightness = im.cmap(im.norm(val))[:3] # Get RGB values
                    text_color = "white" if sum(bg_brightness) < 1.5 else "black" # Adjust threshold as needed

                    if matrix_order:
                        plt.text(j, i, format_nbr(val, precision='.2g'),
                                 ha="center", va="center", color=text_color, fontsize=fontsize*0.8)
                    else:
                        plt.text(i, j, format_nbr(val, precision='.2g'),
                                 ha="center", va="center", color=text_color, fontsize=fontsize*0.8)

    # Remove axis lines (spines)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove only secondary ticks (keep main ticks)
    ax.tick_params(axis='both', which='minor', length=0)

    # Apply tick formats after setting labels.
    # This function is intended for numerical data. If string labels are used,
    # it will not apply numerical formatting, which is the desired behavior here.
    # _apply_axis_tick_formats(plt.gca(), np.arange(ncols), np.arange(nrows))
    plt.xlabel("")  # Remove x label
    plt.ylabel("")  # Remove y label
    out(save=save, datastr=datastr, labels=[], colorbar=True, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cbar_label=cbar_label, interactive_plot=False, equal_aspect=equal_aspect)

@app.command(help="Create a bar plot from data in standard input.")
def bar(
    fields: Annotated[str, typer.Option(help="[bold]Fields to read[/bold]: [cyan]x[/cyan]: x-axis, [cyan]y[/cyan]: y-axis, [cyan]xt[/cyan]: x-tick labels", rich_help_panel="Data Input")] = "x y",
    labels: Annotated[str, typer.Option(help="Space-separated labels for each 'y' field", rich_help_panel="Data Input")] = "",
    delimiter: Annotated[str | None, typer.Option(help="Delimiter used to split the data", rich_help_panel="Data Input")] = None,
    alpha: Annotated[float, typer.Option(help="Alpha (transparency) value for the plot", rich_help_panel="Bar Styling")] = 1.0,
    rotation: Annotated[int, typer.Option(help="Rotation of xtick labels in degrees", rich_help_panel="Bar Styling")] = 45,
    # output options
    save: Annotated[str, typer.Option(help="Filename to save the plot to", rich_help_panel="Output & Limits")] = "",
    xmin: Annotated[float | None, typer.Option(help="Minimum x value for the plot", rich_help_panel="Output & Limits")] = None,
    xmax: Annotated[float | None, typer.Option(help="Maximum x value for the plot", rich_help_panel="Output & Limits")] = None,
    ymin: Annotated[float | None, typer.Option(help="Minimum y value for the plot", rich_help_panel="Output & Limits")] = None,
    ymax: Annotated[float | None, typer.Option(help="Maximum y value for the plot", rich_help_panel="Output & Limits")] = None,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing", rich_help_panel="Test Data")] = False,
    test_npts: Annotated[int, typer.Option(help="Number of points to generate for testing", rich_help_panel="Test Data")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="Number of datasets to generate for testing", rich_help_panel="Test Data")] = 2,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal", rich_help_panel="Bar Styling")] = False,
    # Bar plot specific options
    bar_width: Annotated[float, typer.Option(help="Width of the bars", rich_help_panel="Bar Styling")] = 0.8,
    color: Annotated[str, typer.Option(help="Color of the bars", rich_help_panel="Bar Styling")] = None,
    edgecolor: Annotated[str, typer.Option(help="Color of the bar edges", rich_help_panel="Bar Styling")] = None,
    linewidth: Annotated[float, typer.Option(help="Width of the bar edges", rich_help_panel="Bar Styling")] = 0.0,
    yline: Annotated[float | None, typer.Option(help="Plot a horizontal line at the given y value", rich_help_panel="Bar Styling")] = None,
    yline_color: Annotated[str, typer.Option(help="Color of the horizontal line", rich_help_panel="Bar Styling")] = "red",
    yline_linestyle: Annotated[str, typer.Option(help="Line style of the horizontal line", rich_help_panel="Bar Styling")] = "--",
    yline_linewidth: Annotated[float, typer.Option(help="Line width of the horizontal line", rich_help_panel="Bar Styling")] = 1.0,
    display_values: Annotated[bool, typer.Option(help="Display the actual data values on top of the bars", rich_help_panel="Bar Styling")] = False,
):
    """
    Create a bar plot from data in standard input.

    Args:
        fields (str): The fields to read, separated by spaces.
        labels (str): The labels to use for the data, separated by spaces.
        delimiter (str): The delimiter to use to split the data.
        alpha (float): The alpha value for the plot.
        rotation (int): The rotation of the xtick labels in degrees.
        save (str): The filename to save the plot to.
        xmin (float): The minimum x value for the plot.
        xmax (float): The maximum x value for the plot.
        ymin (float): The minimum y value for the plot.
        ymax (float): The maximum y value for the plot.
        test (bool): If True, generate random data for testing.
        test_npts (int): The number of points to generate for testing.
        test_ndata (int): The number of datasets to generate for testing.
        equal_aspect (bool): If True, set the aspect ratio of the plot to equal.
        bar_width (float): Width of the bars.
        color (str): Color of the bars.
        edgecolor (str): Color of the bar edges.
        linewidth (float): Width of the bar edges.
        yline (float): Plot a horizontal line at the given y value.
        yline_color (str): Color of the horizontal line.
        yline_linestyle (str): Line style of the horizontal line.
        yline_linewidth (float): Line width of the horizontal line.
        display_values (bool): If True, display the actual data values on top of the bars.
    """
    if test:
        data = dict()
        fields = ""
        j = 0
        data[j] = np.arange(test_npts)
        fields += "x "
        j += 1
        for i in range(test_ndata):
            data[j] = np.random.normal(size=test_npts, loc=i, scale=100) + np.arange(test_npts) + i*test_npts
            fields += "y "
            j += 1
        datastr = ""
    else:
        data, datastr, fields = read_data(delimiter, fields, labels)

    fields = fields.strip().split()  # type: ignore
    labels_list = labels.strip().split()  # type: ignore
    xfields = np.where(np.asarray(fields) == "x")[0]
    yfields = np.where(np.asarray(fields) == "y")[0]
    assert len(xfields) == len(yfields) or len(xfields) == 1, "x and y fields must be the same length or x must be a single field"
    if len(xfields) < len(yfields) and len(xfields) == 1:
        xfields = np.ones_like(yfields) * xfields[0]

    plotid = 0
    for xfield, yfield in track(zip(xfields, yfields), total=len(xfields), description="Plotting bar data..."):
        x_current = np.float64(data[xfield])  # type: ignore
        y_current = np.float64(data[yfield])  # type: ignore

        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))
        
        # Create bar plot
        if color is not None:
            bars = plt.bar(x_current, y_current, width=bar_width, alpha=alpha, color=color, 
                   edgecolor=edgecolor, linewidth=linewidth)
        else:
            bars = plt.bar(x_current, y_current, width=bar_width, alpha=alpha, 
                   edgecolor=edgecolor, linewidth=linewidth)
        
        # Display values on top of bars if requested
        if display_values:
            for bar, value in zip(bars, y_current):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (plt.ylim()[1] - plt.ylim()[0]) * 0.005,
                         format_nbr(value, precision='.2g'),
                         ha='center', va='bottom', fontsize=8)
        
        # Set labels if provided
        if len(labels_list) > 0 and plotid < len(labels_list):
            plt.gca().get_legend_handles_labels()[1][0] = labels_list[plotid]
        
        set_xtick_labels(fields, data, rotation=rotation)
        _apply_axis_tick_formats(plt.gca(), x_current, y_current) # Apply tick formats after plotting
        plotid += 1

    # Plot horizontal line if specified
    if yline is not None:
        plt.axhline(y=yline, color=yline_color, linestyle=yline_linestyle, linewidth=yline_linewidth)

    out(save=save, datastr=datastr, labels=labels_list, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, equal_aspect=equal_aspect, legend=False, interactive_plot=False)

if __name__ == "__main__":
    app()
