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
)

@app.callback()
def plot_setup(
    xlabel: Annotated[str, typer.Option(help="Label for the x-axis")] = "x",
    ylabel: Annotated[str, typer.Option(help="Label for the y-axis")] = "y",
    semilog_x: Annotated[bool, typer.Option(help="Set x-axis to log scale")] = False,
    semilog_y: Annotated[bool, typer.Option(help="Set y-axis to log scale")] = False,
    grid: Annotated[bool, typer.Option(help="Add a grid to the plot")] = False,
    aspect_ratio: Annotated[Optional[str], typer.Option(help="Set the figure size (e.g., '10 5' for 10x5 inches)")] = None,
    subplots: Annotated[str, typer.Option(help="Define subplot grid (e.g., '2 2' for a 2x2 grid)")] = "1 1",
    sharex: Annotated[bool, typer.Option(help="Share the x-axis across subplots")] = False,
    sharey: Annotated[bool, typer.Option(help="Share the y-axis across subplots")] = False,
    titles: Annotated[str, typer.Option(help="Titles for each subplot, separated by spaces")] = "",
    xtick_format: Annotated[Optional[str], typer.Option(help="Format string for x-axis tick labels (e.g., '%.2f', '%d', '%.1e')")] = None,
    ytick_format: Annotated[Optional[str], typer.Option(help="Format string for y-axis tick labels (e.g., '%.2f', '%d', '%.1e')")] = None,
    debug: Annotated[bool, typer.Option(help="Enable debug mode")] = False,
):
    """
    A new dataset can be defined by separating the data by an empty line.
    """
    global XTICK_FORMAT
    global YTICK_FORMAT
    XTICK_FORMAT = xtick_format
    YTICK_FORMAT = ytick_format
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
        if semilog_y:
            plt.semilogy()
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
    # AI! how to make this defaultdict an ordered dict (keeping original data ordering)?
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

def set_xtick_labels(fields, data, rotation=45):
    """
    Set the x-tick labels of the plot.

    Args:
        fields (list): The fields of the data.
        data (dict): The data dictionary.
        rotation (int): The rotation of the x-tick labels in degrees.
    """
    if 'xt' in fields:
        xtickslabels = data[fields.index('xt')]
        xval = np.float64(data[fields.index('x')])
        xval, unique_indices = np.unique(xval, return_index=True)
        xtickslabels = np.array(xtickslabels)[unique_indices]
        plt.xticks(xval, xtickslabels.astype(str).tolist())
        # rotate the labels
        plt.setp(plt.gca().get_xticklabels(),
                 rotation=rotation,
                 ha="right",
                 rotation_mode="anchor")

def set_ytick_labels(fields, data, rotation=0):
    """
    Set the y-tick labels of the plot.

    Args:
        fields (list): The fields of the data.
        data (dict): The data dictionary.
        rotation (int): The rotation of the y-tick labels in degrees.
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
            c_val = GLOBAL_C_VALUES[index]
            print_str = f"Nearest point: {label} x={x}, y={y}"
            if not np.isnan(c_val): # Only display c value if it's not NaN
                print_str += f", c={format_nbr(c_val)}"
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
        if effective_xtick_format is None and np.all(x_data == np.round(x_data)):
            effective_xtick_format = "%d"

        if effective_xtick_format:
            if effective_xtick_format == "%d":
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(effective_xtick_format))

    if not is_y_categorical:
        effective_ytick_format = YTICK_FORMAT
        if effective_ytick_format is None and np.all(y_data == np.round(y_data)):
            effective_ytick_format = "%d"

        if effective_ytick_format:
            if effective_ytick_format == "%d":
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(effective_ytick_format))

@app.command()
def plot(
    fields: Annotated[str, typer.Option(help="x: The x field, y: The y field, xt: The xtick labels field, ts: The x field is a timestamp (in seconds since epoch)")] = "x y",
    labels: Annotated[str, typer.Option(help="Space-separated labels for each 'y' field. E.g., if --fields 'x y y' then labels 'Series1 Series2'")] = "",
    moving_avg: Annotated[int, typer.Option(help="The size of the moving average window")] = 0,
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    fmt: Annotated[str, typer.Option(help="The format string to use for the plot")] = "",
    alpha: Annotated[float, typer.Option(help="The alpha value for the plot")] = 1.0,
    rotation: Annotated[int, typer.Option(help="The rotation of the xtick labels in degrees")] = 45,
    # output options
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot")] = None,
    shade: Annotated[str | None, typer.Option(help="Give 0 (no shade) or 1 (shade) to shade the area under the curve. Give 1 value per y field. e.g. if --fields x y y, shade can be 0 1 to only shade the area under the second y field")] = None,
    alpha_shade: Annotated[float, typer.Option(help="The alpha value for the shaded area")] = 0.2,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_npts: Annotated[int, typer.Option(help="The number of points to generate for testing")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="The number of datasets to generate for testing")] = 2,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal")] = False,
    function: Annotated[Optional[str], typer.Option(help="Mathematical expression to plot (e.g., 'x**2 + 2*x + 1'). Use 'x' as the variable.")] = None,
    func_label: Annotated[Optional[str], typer.Option(help="Label for the plotted function in the legend.")] = None,
    func_linestyle: Annotated[str, typer.Option(help="Linestyle for the plotted function (e.g., '-', '--', '-.', ':').")] = '-',
    func_color: Annotated[str, typer.Option(help="Color for the plotted function (e.g., 'red', 'blue', '#FF00FF').")] = 'r',
    legend: Annotated[bool, typer.Option(help="Display legend on the plot")] = True,
    mark_minima: Annotated[bool, typer.Option(help="Mark the position of local minima with their x and y coordinates.")] = False,
    mark_absolute_minima: Annotated[bool, typer.Option(help="Mark the position of the absolute minimum with its x and y coordinates.")] = False,
    plot_average: Annotated[bool, typer.Option(help="Plot the average curve of all 'y' datasets.")] = False,
    average_linestyle: Annotated[str, typer.Option(help="Linestyle for the average curve (e.g., '--', ':', '-.').")] = '--',
    plot_median: Annotated[bool, typer.Option(help="Plot the median curve of all 'y' datasets.")] = False,
    median_linestyle: Annotated[str, typer.Option(help="Linestyle for the median curve (e.g., '-', ':', '-.').")] = '-.',
    plot_gmean: Annotated[bool, typer.Option(help="Plot the geometric mean curve of all 'y' datasets.")] = False,
    gmean_linestyle: Annotated[str, typer.Option(help="Linestyle for the geometric mean curve (e.g., '-', ':', '-.').")] = ':',
    mark_average_minima: Annotated[bool, typer.Option(help="Mark the global minimum of the average curve.")] = False,
    mark_median_minima: Annotated[bool, typer.Option(help="Mark the global minimum of the median curve.")] = False,
    mark_gmean_minima: Annotated[bool, typer.Option(help="Mark the global minimum of the geometric mean curve.")] = False,
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

@app.command()
def scatter(
    fields: Annotated[str, typer.Option(help="x: The x field, y: The y field, xt: The xtick labels field, ts: The x field is a timestamp (in seconds since epoch)")] = "x y",
    labels: Annotated[str, typer.Option(help="Space-separated labels for each 'y' field. E.g., if --fields 'x y y' then labels 'Series1 Series2'")] = "",
    moving_avg: Annotated[int, typer.Option(help="The size of the moving average window")] = 0,
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    fmt: Annotated[str, typer.Option(help="The format string to use for the plot")] = "",
    alpha: Annotated[float, typer.Option(help="The alpha value for the plot")] = 1.0,
    rotation: Annotated[int, typer.Option(help="The rotation of the xtick labels in degrees")] = 45,
    # output options
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot")] = None,
    shade: Annotated[str | None, typer.Option(help="Give 0 (no shade) or 1 (shade) to shade the area under the curve. Give 1 value per y field. e.g. if --fields x y y, shade can be 0 1 to only shade the area under the second y field")] = None,
    alpha_shade: Annotated[float, typer.Option(help="The alpha value for the shaded area")] = 0.2,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_npts: Annotated[int, typer.Option(help="The number of points to generate for testing")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="The number of datasets to generate for testing")] = 2,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal")] = False,
    # New options for scatter plot
    kde: Annotated[bool, typer.Option(help="Use kernel density estimation to color the points")] = False,
    kde_normalize: Annotated[bool, typer.Option(help="Normalize the KDE values")] = False,
    cmap: Annotated[str, typer.Option(help="The colormap to use for the plot")] = "viridis",
    size: Annotated[int, typer.Option(help="The size of the markers in the plot")] = 10,
    pcr: Annotated[bool, typer.Option(help="Perform principal component regression")] = False,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot")] = False,
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
        # Fit KDE
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(xy_data) # Default bandwidth
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

        # Determine marker size: use 's' field, then global 'size' option, then Matplotlib default
        effective_size = np.float64(data[s_indices[0]]) if len(s_indices) > 0 else size

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
        plt.scatter(x, y, s=effective_size, c=c_for_subplot, label=label, alpha=alpha, cmap=cmap)
        if pcr:
            do_pcr(x, y)
        set_xtick_labels(fields, data, rotation=rotation) # Use the defined rotation
        set_ytick_labels(fields, data)
        _apply_axis_tick_formats(plt.gca(), x, y) # Apply tick formats after plotting
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, equal_aspect=equal_aspect)

@app.command()
def hist(
    fields: Annotated[str, typer.Option(help="The fields to read")] = "y",
    labels: Annotated[str, typer.Option(help="The labels to use for the data")] = "",
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    bins: Annotated[str, typer.Option(help="The number of bins to use for the histogram")] = "auto",
    alpha: Annotated[float, typer.Option(help="The alpha value for the plot")] = 1.0,
    density: Annotated[bool, typer.Option(help="Normalize the histogram")] = False,
    # output options
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot")] = None,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_npts: Annotated[int, typer.Option(help="The number of points to generate for testing")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="The number of datasets to generate for testing")] = 2,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal")] = False,
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
        plt.hist(y, toint(bins), label=label, alpha=alpha, density=density)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, interactive_plot=False, equal_aspect=equal_aspect)

@app.command()
def jitter(
    fields: Annotated[str, typer.Option(help="x: The x field, y: The y field, xt: The xtick labels field, c: The color field, il: The interactive labels field")] = "x y",
    labels: Annotated[str, typer.Option(help="The labels to use for the data")] = "",
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    xjitter: Annotated[float, typer.Option(help="The amount of jitter to add to the x values")] = 0.1,
    yjitter: Annotated[float, typer.Option(help="The amount of jitter to add to the y values")] = 0.0,
    size: Annotated[int, typer.Option(help="The size of the markers in the plot")] = 10,
    alpha: Annotated[float, typer.Option(help="The alpha value for the plot")] = 1.0,
    kde: Annotated[bool, typer.Option(help="Use kernel density estimation to color the points")] = False,
    kde_subset: Annotated[int, typer.Option(help="The number of points to use for the KDE")] = 1000,
    kde_normalize: Annotated[bool, typer.Option(help="Normalize the KDE values")] = False,
    cmap: Annotated[str, typer.Option(help="The colormap to use for the plot")] = "viridis",
    median: Annotated[bool, typer.Option(help="Plot the median of the data")] = False,
    median_size: Annotated[int, typer.Option(help="The size of the median markers in the plot")] = 100,
    median_color: Annotated[str, typer.Option(help="The color of the median markers in the plot")] = "black",
    median_marker: Annotated[str, typer.Option(help="The marker to use for the median markers in the plot")] = "_",
    median_sort: Annotated[bool, typer.Option(help="Sort by median values")] = False,
    # output options
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot")] = None,
    rotation: Annotated[int, typer.Option(help="The rotation of the xtick labels in degrees")] = 45,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot")] = False,
    cbar_label: Annotated[str | None, typer.Option(help="The label for the colorbar")] = None,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_npts: Annotated[int, typer.Option(help="The number of points to generate for testing")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="The number of datasets to generate for testing")] = 3,
    equal_aspect: Annotated[bool, typer.Option(help="Set the aspect ratio of the plot to equal")] = False,
    pcr: Annotated[bool, typer.Option(help="Perform principal component regression")] = False,
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
    xfields = np.where(np.asarray(fields) == "x")[0]
    yfields = np.where(np.asarray(fields) == "y")[0]
    cfields = np.where(np.asarray(fields) == "c")[0]
    kde_y = None
    plotid = 0
    for xfield, yfield in zip(xfields, yfields):
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))
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

@app.command()
def roc(
    fields: Annotated[str, typer.Option(help="y: The value (the lower the better by default), a: 1 for active, 0 for inactive")] = "y a",
    labels: Annotated[str, typer.Option(help="The labels to use for the data")] = "",
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    xmin: Annotated[float, typer.Option(help="The minimum x value for the plot")] = 0.0,
    xmax: Annotated[float, typer.Option(help="The maximum x value for the plot")] = 1.0,
    ymin: Annotated[float, typer.Option(help="The minimum y value for the plot")] = 0.0,
    ymax: Annotated[float, typer.Option(help="The maximum y value for the plot")] = 1.0,
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_npts: Annotated[int, typer.Option(help="The number of points to generate for testing")] = 1000,
    test_ndata: Annotated[int, typer.Option(help="The number of datasets to generate for testing")] = 2,
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

@app.command()
def tsne(
    perplexity: Annotated[float, typer.Option(help="The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.")] = 30.0,
    early_exaggeration: Annotated[float, typer.Option(help="Controls how tightly clusters in the original space are bunched together in the embedded space. Larger values increase the separation between clusters.")] = 12.0,
    learning_rate: Annotated[float, typer.Option(help="The learning rate for t-SNE. Typically between 10.0 and 1000.0. If the learning rate is too high, the data will look like a 'ball' with any point approximately at the same distance from its nearest neighbours. If the learning rate is too low, most points will look compressed in a dense cloud with few outliers. Some suggestions are between 100 and 1000.")] = 200.0,
    n_iter: Annotated[int, typer.Option(help="Maximum number of iterations for the optimization. Should be at least 250.")] = 1000,
    metric: Annotated[str, typer.Option(help="The metric to use to compute distance in high dimensional space (default: euclidean, precomputed, cosine, manhattan, hamming, etc.)")] = "euclidean",
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    npy: Annotated[str, typer.Option(help="Load data from a numpy file")] = "",
    npz: Annotated[str, typer.Option(help="Load data from a numpy file (compressed)")] = "",
    data_key: Annotated[str, typer.Option(help="The key to use to load data from the npz file")] = "data",
    labels_key: Annotated[str, typer.Option(help="The key to use to load labels from the npz file")] = "",
    ilabels_key: Annotated[str, typer.Option(help="The key to use to load interactive labels from the npz file")] = "",
    legend: Annotated[bool, typer.Option(help="Add a legend to the plot")] = True,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot")] = False,
    cmap: Annotated[str, typer.Option(help="The colormap to use for the plot")] = "viridis",
    size: Annotated[int, typer.Option(help="The size of the markers in the plot")] = 10,
    alpha: Annotated[float, typer.Option(help="The transparency of the markers in the plot")] = 1.0,
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot")] = None,
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

@app.command()
def umap(
    n_neighbors: Annotated[int, typer.Option(help="The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation")] = 15,
    min_dist: Annotated[float, typer.Option(help="The effective minimum distance between embedded points")] = 0.1,
    metric: Annotated[str, typer.Option(help="The metric to use to compute distance in high dimensional space (default: euclidean, precomputed, cosine, manhattan, hamming, etc.)")] = "euclidean",
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    npy: Annotated[str, typer.Option(help="Load data from a numpy file")] = "",
    npz: Annotated[str, typer.Option(help="Load data from a numpy file (compressed)")] = "",
    data_key: Annotated[str, typer.Option(help="The key to use to load data from the npz file")] = "data",
    labels_key: Annotated[str, typer.Option(help="The key to use to load labels from the npz file")] = "",
    ilabels_key: Annotated[str, typer.Option(help="The key to use to load interactive labels from the npz file")] = "",
    legend: Annotated[bool, typer.Option(help="Add a legend to the plot")] = True,
    colorbar: Annotated[bool, typer.Option(help="Add a colorbar to the plot")] = False,
    cmap: Annotated[str, typer.Option(help="The colormap to use for the plot")] = "viridis",
    size: Annotated[int, typer.Option(help="The size of the markers in the plot")] = 10,
    alpha: Annotated[float, typer.Option(help="The transparency of the markers in the plot")] = 1.0,
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot")] = None,
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

@app.command()
def read_metadata(filename: Annotated[str, typer.Option(..., help="The filename to read the metadata from")]):
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

@app.command()
def chord_diagram(
    fields: Annotated[str, typer.Option(help="d: The data field (matrix values), r: The row labels field, c: The column labels field")] = "d r c",
    labels: Annotated[str, typer.Option(help="The labels to use for the data")] = "",
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
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

@app.command()
def venn_diagram(
    fields: Annotated[str, typer.Option(help="d: The data field (set components), l: The set label field (Unique label for each set, maximum 6 labels, 6 sets)")] = "d l",
    labels_fill: Annotated[str, typer.Option(help="Comma-separated options for filling labels: 'number', 'logic', 'percent', 'elements'. E.g., 'number,percent'")] = "number",
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_ndata: Annotated[int, typer.Option(help="The number of sets to generate for testing (2 to 6)")] = 3,
    test_npts: Annotated[int, typer.Option(help="The number of points in each test set")] = 10,
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    colors: Annotated[str, typer.Option(help="Comma-separated list of colors (e.g., 'red,blue,green'). Uses default colors if not specified.")] = "",
    figsize: Annotated[str, typer.Option(help="Figure size in inches (e.g., '9 7'). Uses default if not specified.")] = "",
    dpi: Annotated[int, typer.Option(help="Resolution of the figure in dots per inch.")] = 96,
    fontsize: Annotated[int, typer.Option(help="Font size for labels.")] = 14,
    sortkey: Annotated[int, typer.Option(help="Key to sort elements in 'elements' fill option. Defaults to 0 for string slicing.")] = 0,
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

@app.command()
def heatmap(
    fields: Annotated[str, typer.Option(help="v: The value field, r: The row label field, c: The column label field")] = "v r c",
    delimiter: Annotated[str | None, typer.Option(help="The delimiter to use to split the data")] = None,
    cmap: Annotated[str, typer.Option(help="The colormap to use for the heatmap")] = "viridis",
    cbar_label: Annotated[str | None, typer.Option(help="Label for the colorbar")] = None,
    rotation: Annotated[int, typer.Option(help="Rotation for x-tick labels")] = 90,
    # output options
    save: Annotated[str, typer.Option(help="The filename to save the plot to")] = "",
    xmin: Annotated[float | None, typer.Option(help="The minimum x value for the plot (will be ignored if not applicable for heatmap)")] = None,
    xmax: Annotated[float | None, typer.Option(help="The maximum x value for the plot (will be ignored if not applicable for heatmap)")] = None,
    ymin: Annotated[float | None, typer.Option(help="The minimum y value for the plot (will be ignored if not applicable for heatmap)")] = None,
    ymax: Annotated[float | None, typer.Option(help="The maximum y value for the plot (will be ignored if not applicable for heatmap)")] = None,
    # test options
    test: Annotated[bool, typer.Option(help="Generate random data for testing")] = False,
    test_rows: Annotated[int, typer.Option(help="Number of rows for test data")] = 5,
    test_cols: Annotated[int, typer.Option(help="Number of columns for test data")] = 7,
    matrix_order: Annotated[bool, typer.Option(help="If True, plot the heatmap in matrix order (rows and columns instead of x,y)")] = False,
    fontsize: Annotated[int, typer.Option(help="Font size for tick labels")] = 10,
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

    for i in track(range(len(values)), description="Building heatmap matrix..."):
        r_label = row_labels_raw[i]
        c_label = col_labels_raw[i]
        val = values[i]

        row_idx = row_to_idx[r_label]
        col_idx = col_to_idx[c_label]
        heatmap_matrix[row_idx, col_idx] = val

    if matrix_order:
        plt.imshow(heatmap_matrix, cmap=cmap, origin='upper', aspect='auto')
        plt.xticks(np.arange(ncols), unique_col_labels, rotation=rotation, ha='right', rotation_mode='anchor', fontsize=fontsize)
        plt.yticks(np.arange(nrows), unique_row_labels, fontsize=fontsize)
    else:
        plt.imshow(heatmap_matrix.T, cmap=cmap, origin='lower', aspect='auto')
        plt.xticks(np.arange(nrows), unique_row_labels, rotation=rotation, ha='right', rotation_mode='anchor', fontsize=fontsize)
        plt.yticks(np.arange(ncols), unique_col_labels, fontsize=fontsize)

    # Apply tick formats after setting labels.
    # This function is intended for numerical data. If string labels are used,
    # it will not apply numerical formatting, which is the desired behavior here.
    # _apply_axis_tick_formats(plt.gca(), np.arange(ncols), np.arange(nrows))
    plt.xlabel("")  # Remove x label
    plt.ylabel("")  # Remove y label
    out(save=save, datastr=datastr, labels=[], colorbar=True, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cbar_label=cbar_label, interactive_plot=False)

if __name__ == "__main__":
    app()
