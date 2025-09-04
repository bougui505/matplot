#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Mar 31 13:12:22 2025

import os
import socket
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy
import typer
from numpy import linalg
from PIL import Image, PngImagePlugin
from PIL.PngImagePlugin import PngInfo
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.neighbors import KernelDensity, NearestNeighbors
from typing_extensions import Annotated

from draggable_text import DraggableText
from ROC import ROC

console = Console()

# Reading data from a png with large number of points:
# See: https://stackoverflow.com/a/61466412/1679629
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# Define X and Y as global variables
X, Y = list(), list()
INTERACTIVE_LABELS = list()

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.callback()
def plot_setup(
    xlabel:str="x",
    ylabel:str="y",
    semilog_x:bool=False,
    semilog_y:bool=False,
    grid:bool=False,
    aspect_ratio:str=None,  # type:ignore
    subplots:str="1 1",
    sharex:bool=False,
    sharey:bool=False,
    titles:str="",
    debug:bool=False,
):
    """
    Read data from stdin and plot them.

    If an empty line is found, a new dataset is created.

    Setup the plot with the given parameters

    --aspect_ratio: "16 9", set the aspect ratio of the plot
    """
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = DEBUG
    if aspect_ratio is not None:
        xaspect, yaspect = aspect_ratio.split()
        plt.figure(figsize=(float(xaspect), float(yaspect)))
    global SUBPLOTS
    SUBPLOTS = [int(e) for e in subplots.strip().split()]  # type:ignore
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

def read_data(delimiter, fields, labels):
    """
    Read data from stdin and return a dictionary of data
    if an empty line is found, new fields are created

    :param delimiter: The delimiter to use to split the data
    :param fields: The fields to read
    :param labels: The labels to use for the data
    :return: A dictionary of data, a string of data, and the fields
    """
    data = defaultdict(list)
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
            data[i].append(e)
            datastr += e + ","
            if i > imax:
                imax = i
        datastr += "\n"
    # Printing data as a table
    fields_list = fields.strip().split()
    labels_list = labels.strip().split()
    ndataset = (np.asarray(fields_list)=="y").sum()
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

def set_limits(xmin=None, xmax=None, ymin=None, ymax=None, equal_aspect:bool=False):
    limits = plt.axis()
    if xmin is None:
        xmin = limits[0]
    if xmax is None:
        xmax = limits[1]
    ax = plt.gca()
    ax.set_xlim([float(xmin), float(xmax)])
    if ymin is None:
        ymin = limits[-2]
    if ymax is None:
        ymax = limits[-1]
    ax.set_ylim([float(ymin), float(ymax)])
    if equal_aspect:
        ax.set_aspect('equal')

def saveplot(outfilename, datastr, labels=None):
    plt.savefig(outfilename)
    ext = os.path.splitext(outfilename)[1]
    print(f"{ext=}")
    print(f"{outfilename=}")
    if ext == ".png":
        add_metadata(outfilename, datastr, labels=labels)

def add_metadata(filename, datastr, key="data", labels=None):
    """
    Add metadata to a png file
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
    Set the xtick labels for the plot
    """
    xticks = []
    xticklabels = []
    if 'xt' in fields:
        xtickslabels = data[fields.index('xt')]
        xval = np.float64(data[fields.index('x')])
        xval, unique_indices = np.unique(xval, return_index=True)
        xtickslabels = np.array(xtickslabels)[unique_indices]
        plt.xticks(xval, xtickslabels)
        # rotate the labels
        plt.setp(plt.gca().get_xticklabels(),
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
    interactive_plot:bool=True,
    legend:bool=True,
    equal_aspect:bool=False,
):
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
    Print the x, y coordinates of the mouse click on the plot
    """
    if event.xdata is not None and event.ydata is not None:
        # find the nearest point in the kdtree
        print(event.xdata, event.ydata)
        dist, index = NEIGH.kneighbors(np.asarray([event.xdata, event.ydata]).reshape(1, -1))
        index = index.squeeze()
        dist = dist.squeeze()
        x = X[index]
        y = Y[index]
        if len(INTERACTIVE_LABELS) > 0:
            label = INTERACTIVE_LABELS[index]
        else:
            label = ""
        print(f"Nearest point: {label} x={x}, y={y}, dist={dist:.2g}")
        # print(f"x={event.xdata:.2f}, y={event.ydata:.2f}")

def toint(x):
    try:
        x = int(x)
    except ValueError:
        pass
    return x

@app.command()
def plot(
    fields:Annotated[str, typer.Option(help="x: The x field, y: The y field, xt: The xtick labels field, ts: The x field is a timestamp (in seconds since epoch)")]="x y",
    labels:Annotated[str, typer.Option(help="The labels to use for the data")]="",
    moving_avg:Annotated[int, typer.Option(help="The size of the moving average window")]=0,
    delimiter:Annotated[str, typer.Option(help="The delimiter to use to split the data")]=None,
    fmt:Annotated[str, typer.Option(help="The format string to use for the plot")]="",
    alpha:Annotated[float, typer.Option(help="The alpha value for the plot")]=1.0,
    rotation:Annotated[int, typer.Option(help="The rotation of the xtick labels in degrees")]=45,
    # output options
    save:Annotated[str, typer.Option(help="The filename to save the plot to")]="",
    xmin:Annotated[float, typer.Option(help="The minimum x value for the plot")]=None,  # type:ignore
    xmax:Annotated[float, typer.Option(help="The maximum x value for the plot")]=None,  # type:ignore
    ymin:Annotated[float, typer.Option(help="The minimum y value for the plot")]=None,  # type:ignore
    ymax:Annotated[float, typer.Option(help="The maximum y value for the plot")]=None,  # type:ignore
    shade:Annotated[str, typer.Option(help="Give 0 (no shade) or 1 (shade) to shade the area under the curve. Give 1 value per y field. e.g. if --fields x y y, shade can be 0 1 to only shade the area under the second y field")]=None,  # type: ignore
    alpha_shade:Annotated[float, typer.Option(help="The alpha value for the shaded area")]=0.2,
    # test options
    test:Annotated[bool, typer.Option(help="Generate random data for testing")]=False,
    test_npts:Annotated[int, typer.Option(help="The number of points to generate for testing")]=1000,
    test_ndata:Annotated[int, typer.Option(help="The number of datasets to generate for testing")]=2,
):
    """
    Plot y versus x as lines and/or markers, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    --fields:\n
    x: The x field\n
    y: The y field\n
    xt: The xtick labels field\n
    ts: The x field is a timestamp (in seconds since epoch)\n
    --shade:
    give 0 (no shade) or 1 (shade) to shade the area under the curve\n
    Give 1 value per y field\n
    e.g. if --fields x y y, shade can be 0 1 to only shade the area under the second y field\n
    --alpha-shade: The alpha value for the shaded area (default: 0.2)\n
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
    fields = fields.strip().split()
    if shade is not None:
        shade = shade.strip().split()  # type:ignore
        shade = np.bool_(np.int_(shade))  # type:ignore
    else:
        shade = np.zeros(len(fields), dtype=bool)
    xfmt = None
    if "ts" in fields:
        xfmt = "ts"
        fields = [f if f != "ts" else "x" for f in fields]
    assert "x" in fields, "x field is required"
    labels = labels.strip().split()
    if fmt != "":
        fmt = fmt.strip().split()
    else:
        fmt = [fmt] * len(data)
    plotid = 0
    xfields = np.where(np.asarray(fields)=="x")[0]
    yfields = np.where(np.asarray(fields)=="y")[0]
    assert len(xfields) == len(yfields) or len(xfields) == 1, "x and y fields must be the same length or x must be a single field"
    if len(xfields) < len(yfields) and len(xfields) == 1:
        xfields = np.ones_like(yfields) * xfields[0]
    for xfield, yfield in track(zip(xfields, yfields), total=len(xfields), description="Plotting..."):
        x = np.float64(data[xfield])  # type: ignore
        y = np.float64(data[yfield])  # type: ignore
        X.extend(list(x))  # type:ignore
        Y.extend(list(y))  # type:ignore
        if moving_avg > 0:
            x = np.convolve(x, np.ones((moving_avg,))/moving_avg, mode='valid')
            y = np.convolve(y, np.ones((moving_avg,))/moving_avg, mode='valid')
        if len(labels) > 0:
            label = labels[plotid]
        else:
            label = None
        if len(fmt) > 0:
            fmtstr = fmt[plotid]
        else:
            fmtstr = ""
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))  # type:ignore
        if xfmt == "ts":
            x = np.asarray([datetime.fromtimestamp(e) for e in x])
        plt.plot(x, y, fmtstr, label=label, alpha=alpha)
        if xfmt == "ts":
            plt.gcf().autofmt_xdate()
        if shade[plotid]:
            # get the color of the last plot:
            color = plt.gca().lines[-1].get_color()
            plt.fill_between(x, y, alpha=alpha_shade, color=color)
        set_xtick_labels(fields, data, rotation=rotation)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

@app.command()
def scatter(
    fields:Annotated[str, typer.Option(help="x: The x field, y: The y field, c: A sequence of numbers to be mapped to colors using cmap (see: --cmap), s: The marker size in points**2, il: a particular field with labels to display for interactive mode, t: a field with text labels to display on the plot")]="x y",
    labels:Annotated[str, typer.Option(help="The labels to use for the data")]="",
    delimiter:Annotated[str, typer.Option(help="The delimiter to use to split the data")]=None,
    alpha:Annotated[float, typer.Option(help="The alpha value for the plot")]=1.0,
    cmap:Annotated[str, typer.Option(help="The colormap to use for the plot")]="viridis",
    pcr:Annotated[bool, typer.Option(help="Principal component regression (see: https://en.wikipedia.org/wiki/Principal_component_regression)")]=False,
    # output options
    save:Annotated[str, typer.Option(help="The filename to save the plot to")]="",
    xmin:Annotated[float, typer.Option(help="The minimum x value for the plot")]=None,  # type:ignore
    xmax:Annotated[float, typer.Option(help="The maximum x value for the plot")]=None,  # type:ignore
    ymin:Annotated[float, typer.Option(help="The minimum y value for the plot")]=None,  # type:ignore
    ymax:Annotated[float, typer.Option(help="The maximum y value for the plot")]=None,  # type:ignore
    colorbar:Annotated[bool, typer.Option(help="Add a colorbar to the plot")]=False,
    # test options
    test:Annotated[bool, typer.Option(help="Generate random data for testing")]=False,
    test_npts:Annotated[int, typer.Option(help="The number of points to generate for testing")]=1000,
    test_ndata:Annotated[int, typer.Option(help="The number of datasets to generate for testing")]=2,
):
    """
    A scatter plot of y vs. x with varying marker size and/or color, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html\n
    --fields: x y c s (c: A sequence of numbers to be mapped to colors using cmap (see: --cmap), s: The marker size in points**2)\n
              il: a particular field with labels to display for interactive mode\n
              t: a field with text labels to display on the plot\n
                 text is draggable with the mouse to avoid overlaps\n
    --pcr: principal component regression (see: https://en.wikipedia.org/wiki/Principal_component_regression)\n
    --cmap: see: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps\n
    """
    global X
    global Y
    global INTERACTIVE_LABELS
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
    fields = fields.strip().split()
    labels = labels.strip().split()
    xfields = np.where(np.asarray(fields)=="x")[0]
    yfields = np.where(np.asarray(fields)=="y")[0]
    s_indices = np.where(np.asarray(fields)=="s")[0]
    c_indices = np.where(np.asarray(fields)=="c")[0]
    plotid = 0
    for xfield, yfield in zip(xfields, yfields):
        x = np.float64(data[xfield])  # type: ignore
        y = np.float64(data[yfield])  # type: ignore
        X.extend(list(x))  # type:ignore
        Y.extend(list(y))  # type:ignore
        if "il" in fields:
            INTERACTIVE_LABELS.extend(data[fields.index("il")])
        if len(labels) > 0:
            label = labels[0]
        else:
            label = None
        if len(s_indices) > 0:
            s = np.float64(data[s_indices[0]])  # type: ignore
        else:
            s = None
        if len(c_indices) > 0:
            c = np.float64(data[c_indices[0]])  # type: ignore
        else:
            c = None
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))  # type:ignore
        if "t" in fields:
            texts_to_drag = list()
            for x_,y_,t_ in zip(x,y,data[fields.index("t")]):
                texts_to_drag.append(plt.annotate(t_, (x_,y_),
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
                # text_to_drag = plt.text(0.5, 0.5, 'Hello Draggable!', # Source
                #             fontsize=22, ha='center', va='center', color='darkblue',
                #             bbox=dict(facecolor='lightblue', alpha=0.7, pad=7, boxstyle="round,pad=0.5"),
                #             zorder=10 # Ensures text is drawn on top [6]
                #             )
            draggable_text_instances = list()
            for text_to_drag in texts_to_drag:
                draggable_text_instances.append(DraggableText(text_to_drag))
            for draggable_text_instance in draggable_text_instances:
                draggable_text_instance.connect()

        plt.scatter(x, y, s=s, c=c, label=label, alpha=alpha, cmap=cmap)
        if pcr:
            do_pcr(x,y)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

@app.command()
def hist(
    fields:Annotated[str, typer.Option(help="The fields to read")]="y",
    labels:Annotated[str, typer.Option(help="The labels to use for the data")]="",
    delimiter:Annotated[str, typer.Option(help="The delimiter to use to split the data")]=None,
    bins:Annotated[str, typer.Option(help="The number of bins to use for the histogram")]="auto",
    alpha:Annotated[float, typer.Option(help="The alpha value for the plot")]=1.0,
    density:Annotated[bool, typer.Option(help="Normalize the histogram")]=False,
    # output options
    save:Annotated[str, typer.Option(help="The filename to save the plot to")]="",
    xmin:Annotated[float, typer.Option(help="The minimum x value for the plot")]=None, # type:ignore
    xmax:Annotated[float, typer.Option(help="The maximum x value for the plot")]=None, # type:ignore
    ymin:Annotated[float, typer.Option(help="The minimum y value for the plot")]=None, # type:ignore
    ymax:Annotated[float, typer.Option(help="The maximum y value for the plot")]=None, # type:ignore
    # test options
    test:Annotated[bool, typer.Option(help="Generate random data for testing")]=False,
    test_npts:Annotated[int, typer.Option(help="The number of points to generate for testing")]=1000,
    test_ndata:Annotated[int, typer.Option(help="The number of datasets to generate for testing")]=2,
):
    """
    Compute and plot an histogram, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
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
    fields = fields.strip().split()
    labels = labels.strip().split()
    plotid = 0
    for j, field in enumerate(fields):
        if field == "y":
            y = np.float64(data[j])  # type: ignore
            Y.extend(list(y))  # type:ignore
        else:
            continue
        if len(labels) > 0:
            label = labels[plotid]
        else:
            label = None
        plt.hist(y, toint(bins), label=label, alpha=alpha, density=density)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, interactive_plot=False)

@app.command()
def jitter(
    fields:Annotated[str, typer.Option(help="x: The x field, y: The y field, xt: The xtick labels field, c: The color field, il: The interactive labels field")]="x y",
    labels:Annotated[str, typer.Option(help="The labels to use for the data")]="",
    delimiter:Annotated[str, typer.Option(help="The delimiter to use to split the data")]=None,
    xjitter:Annotated[float, typer.Option(help="The amount of jitter to add to the x values")]=0.1,
    yjitter:Annotated[float, typer.Option(help="The amount of jitter to add to the y values")]=0.0,
    size:Annotated[int, typer.Option(help="The size of the markers in the plot")]=10,
    alpha:Annotated[float, typer.Option(help="The alpha value for the plot")]=1.0,
    kde:Annotated[bool, typer.Option(help="Use kernel density estimation to color the points")]=False,
    kde_subset:Annotated[int, typer.Option(help="The number of points to use for the KDE")]=1000,
    kde_normalize:Annotated[bool, typer.Option(help="Normalize the KDE values")]=False,
    cmap:Annotated[str, typer.Option(help="The colormap to use for the plot")]="viridis",
    median:Annotated[bool, typer.Option(help="Plot the median of the data")]=False,
    median_size:Annotated[int, typer.Option(help="The size of the median markers in the plot")]=100,
    median_color:Annotated[str, typer.Option(help="The color of the median markers in the plot")]="black",
    median_marker:Annotated[str, typer.Option(help="The marker to use for the median markers in the plot")]="_",
    median_sort:Annotated[bool, typer.Option(help="Sort by median values")]=False,
    # output options
    save:Annotated[str, typer.Option(help="The filename to save the plot to")]="",
    xmin:Annotated[float, typer.Option(help="The minimum x value for the plot")]=None,  # type:ignore
    xmax:Annotated[float, typer.Option(help="The maximum x value for the plot")]=None,  # type:ignore
    ymin:Annotated[float, typer.Option(help="The minimum y value for the plot")]=None,  # type:ignore
    ymax:Annotated[float, typer.Option(help="The maximum y value for the plot")]=None,  # type:ignore
    rotation:Annotated[int, typer.Option(help="The rotation of the xtick labels in degrees")]=45,
    colorbar:Annotated[bool, typer.Option(help="Add a colorbar to the plot")]=False,
    cbar_label:Annotated[str, typer.Option(help="The label for the colorbar")]=None,  # type:ignore
    # test options
    test:Annotated[bool, typer.Option(help="Generate random data for testing")]=False,
    test_npts:Annotated[int, typer.Option(help="The number of points to generate for testing")]=1000,
    test_ndata:Annotated[int, typer.Option(help="The number of datasets to generate for testing")]=3,
):
    """
    Jitter plot

    --fields:\n
        x: The x field\n
        y: The y field\n
        xt: The xtick labels field\n
        c: The color field\n
        il: The interactive labels field\n
    --rotation: The rotation of the xtick labels in degrees (default: 45)\n
    --median: Plot the median of the data\n
    --median_sort: Sort by median values\n
    --kde_normalize: Normalize the KDE values\n
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
    fields = fields.strip().split()
    labels = labels.strip().split()
    xfields = np.where(np.asarray(fields)=="x")[0]
    yfields = np.where(np.asarray(fields)=="y")[0]
    cfields = np.where(np.asarray(fields)=="c")[0]
    kde_y = None
    plotid = 0
    for xfield, yfield in zip(xfields, yfields):
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))  # type:ignore
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
                kde_y = np.exp(kde_ins.score_samples(ysel[:, None]))  # type: ignore
                if kde_normalize:
                    kde_y -= kde_y.min()
                    kde_y /= kde_y.max()
                c[sel] = kde_y
            c = np.asarray(c)
        x += np.random.uniform(size=x.shape, low=-xjitter/2, high=xjitter/2)
        y += np.random.uniform(size=y.shape, low=-yjitter/2, high=yjitter/2)
        X.extend(list(x))  # type:ignore
        Y.extend(list(y))  # type:ignore
        plt.scatter(x, y, c=c, s=size, alpha=alpha, cmap=cmap)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cbar_label=cbar_label)

def plot_median(x, y, size=100, color="black", marker="_", median_sort:bool=False):
    """
    Plot the median of the data
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
    fields:Annotated[str, typer.Option(help="y: The value (the lower the better by default), a: 1 for active, 0 for inactive")]="y a",
    labels:Annotated[str, typer.Option(help="The labels to use for the data")]="",
    delimiter:Annotated[str, typer.Option(help="The delimiter to use to split the data")]=None,
    test:Annotated[bool, typer.Option(help="Generate random data for testing")]=False,
    save:Annotated[str, typer.Option(help="The filename to save the plot to")]="",
    xmin:Annotated[float, typer.Option(help="The minimum x value for the plot")]=0.0,
    xmax:Annotated[float, typer.Option(help="The maximum x value for the plot")]=1.0,
    ymin:Annotated[float, typer.Option(help="The minimum y value for the plot")]=0.0,
    ymax:Annotated[float, typer.Option(help="The maximum y value for the plot")]=1.0,

    ):
    """
    Plot a ROC curve (Receiver Operating Characteristic curve) for binary classification.\n

    --fields: y a (y: The value (the lower the better by default), a: 1 for active, 0 for inactive)\n
    """
    global X
    global Y
    global INTERACTIVE_LABELS
    data, datastr, fields = read_data(delimiter=delimiter, fields=fields, labels=labels)
    fields = fields.strip().split()
    labels = labels.strip().split()  # type:ignore
    yfields = np.where(np.asarray(fields)=="y")[0]
    afields = np.where(np.asarray(fields)=="a")[0]
    for plotid, (yfield, afield) in enumerate(zip(yfields, afields)):
        y = np.float64(data[yfield])  # type: ignore
        a = np.int_(data[afield])  # type: ignore
        active_values = y[a == 1]  # type: ignore
        inactive_values = y[a == 0]  # type: ignore
        x, y, auc, pROC_auc, thresholds = ROC(active_values, inactive_values)
        X.extend(list(x))  # type:ignore
        Y.extend(list(y))  # type:ignore
        label = labels[plotid] if len(labels) > 0 else None
        if label is None or label == "":
            label = f"AUC={auc:.2f}, pROC={pROC_auc:.2f}"
            labels.append(label)  # type: ignore
        else:
            label += f" (AUC={auc:.2f}, pROC={pROC_auc:.2f})"
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))  # type:ignore
        plt.plot(x, y, label=label)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        if SUBPLOTS[0]*SUBPLOTS[1] > 1:
            set_limits(xmin, xmax, ymin, ymax)
            ax = plt.gca()
            ax.set_aspect('equal')  # type: ignore
            plt.title(label)
            plt.plot([xmin, xmax], [ymin, ymax], 'k--', label="Random")  # type:ignore
    if SUBPLOTS[0]*SUBPLOTS[1] == 1:
        plt.plot([xmin, xmax], [ymin, ymax], 'k--', label="Random")  # type:ignore
    else:
        labels = []  # type: ignore
    out(save=save, datastr=datastr, labels=labels, colorbar=None, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cbar_label=None, equal_aspect=True)

@app.command()
def umap(
    n_neighbors:Annotated[int, typer.Option(help="The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation")]=15,
    min_dist:Annotated[float, typer.Option(help="The effective minimum distance between embedded points")]=0.1,
    metric:Annotated[str, typer.Option(help="The metric to use to compute distance in high dimensional space (default: euclidean, precomputed, cosine, manhattan, hamming, etc.)")]="euclidean",
    test:Annotated[bool, typer.Option(help="Generate random data for testing")]=False,
    save:Annotated[str, typer.Option(help="The filename to save the plot to")]="",
    npy:Annotated[str, typer.Option(help="Load data from a numpy file")]="",
    npz:Annotated[str, typer.Option(help="Load data from a numpy file (compressed)")]="",
    data_key:Annotated[str, typer.Option(help="The key to use to load data from the npz file")]="data",
    labels_key:Annotated[str, typer.Option(help="The key to use to load labels from the npz file")]="",
    ilabels_key:Annotated[str, typer.Option(help="The key to use to load interactive labels from the npz file")]="",
    legend:Annotated[bool, typer.Option(help="Add a legend to the plot")]=True,
    colorbar:Annotated[bool, typer.Option(help="Add a colorbar to the plot")]=False,
    cmap:Annotated[str, typer.Option(help="The colormap to use for the plot")]="viridis",
    size:Annotated[int, typer.Option(help="The size of the markers in the plot")]=10,
    alpha:Annotated[float, typer.Option(help="The transparency of the markers in the plot")]=1.0,
    xmin:Annotated[float, typer.Option(help="The minimum x value for the plot")]=None,  # type:ignore
    xmax:Annotated[float, typer.Option(help="The maximum x value for the plot")]=None,  # type:ignore
    ymin:Annotated[float, typer.Option(help="The minimum y value for the plot")]=None,  # type:ignore
    ymax:Annotated[float, typer.Option(help="The maximum y value for the plot")]=None,  # type:ignore
):
    """
    UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction technique.\n
    See: https://umap-learn.readthedocs.io/en/latest/\n
    n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.\n
    min_dist: The effective minimum distance between embedded points.\n
    metric: The metric to use to compute distance in high dimensional space (default: euclidean, precomputed, cosine, manhattan, hamming, etc.)\n
    test: Generate random data for testing\n
    save: Save the plot to a file\n
    npy: Load data from a numpy file\n
    npz: Load data from a numpy file (compressed)\n
    data_key: The key to use to load data from the npz file\n
    labels_key: The key to use to load labels from the npz file\n
    ilabels_key: The key to use to load interactive labels from the npz file\n
    colorbar: Add a colorbar to the plot\n
    cmap: The colormap to use for the plot\n
    size: The size of the markers in the plot\n
    alpha: The transparency of the markers in the plot\n
    xmin: The minimum x value for the plot\n
    xmax: The maximum x value for the plot\n
    ymin: The minimum y value for the plot\n
    ymax: The maximum y value for the plot\n
    """
    global X
    global Y
    global INTERACTIVE_LABELS
    import umap

    if test:
        data = np.random.normal(loc=(0,0,0), size=(100, 3))
        data = np.concatenate((data, np.random.normal(loc=(1,1,1), size=(100, 3))), axis=0)
    if npy != "":
        data = np.load(npy)
    labels = None
    if npz != "":
        dataz = np.load(npz)
        print(f"{dataz.files=}")
        data = dataz[data_key]
        if labels_key != "":
            labels = dataz[labels_key]
        if ilabels_key != "":
            ilabels = dataz[ilabels_key]
    print(f"{data.shape=}")  # type:ignore
    print(f"{data.min()=}")  # type:ignore
    print(f"{data.max()=}")  # type:ignore
    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = mapper.fit_transform(data)  # type:ignore
    # umap.plot.points(mapper, values=r_orig)
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=size, cmap=cmap, alpha=alpha)  # type:ignore
        X.extend(list(embedding[:, 0]))  # type:ignore
        Y.extend(list(embedding[:, 1]))  # type:ignore
    else:
        for label in np.unique(labels):
            sel = labels == label
            x = embedding[sel, 0]  # type:ignore
            y = embedding[sel, 1]  # type:ignore
            plt.scatter(x, y, s=size, cmap=cmap, alpha=alpha, label=label)  # type:ignore
            X.extend(list(x))
            Y.extend(list(y))
            if ilabels_key == "":
                INTERACTIVE_LABELS.extend(list(labels[sel]))
            else:
                INTERACTIVE_LABELS.extend(list(ilabels[sel]))  # type:ignore
    out(save=save, datastr="", labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, legend=legend)

@app.command()
def read_metadata(filename:Annotated[str, typer.Option(help="The filename to read the metadata from")]):
    """
    Read metadata at least for a png file
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
    Principal Component Regression
    See: https://en.wikipedia.org/wiki/Principal_component_regression
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
    if float(format(x, precision))==round(x):
        return f'{round(x)}'
    else:
        return format(x, precision)

def pca(X, outfilename=None):
    """
    >>> X = np.random.normal(size=(10, 512))
    >>> proj = compute_pca(X)
    >>> proj.shape
    (10, 2)
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

if __name__ == "__main__":
    import doctest
    import sys

    @app.command()
    def test():
        """
        Test the code
        """
        doctest.testmod(
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF
        )

    @app.command()
    def test_func(func:Annotated[str, typer.Option(help="The function to test")]):
        """
        Test the given function
        """
        print(f"Testing {func}")
        f = getattr(sys.modules[__name__], func)
        doctest.run_docstring_examples(
            f,
            globals(),
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF,
        )
    app()
