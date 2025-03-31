#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Mar 31 13:12:22 2025

import gzip
import os
import socket
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image, PngImagePlugin
from PIL.PngImagePlugin import PngInfo

# Reading data from a png with large number of points:
# See: https://stackoverflow.com/a/61466412/1679629
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

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
):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if semilog_x:
        plt.semilogx()
    if semilog_y:
        plt.semilogy()
    if grid:
        plt.grid()

def read_data(delimiter):
    data = defaultdict(list)
    datastr = ""
    for line in sys.stdin.readlines():
        line = line.strip().split(delimiter)
        for i, e in enumerate(line):
            data[i].append(e)
            datastr += e + ","
        datastr += "\n"
    return data, datastr

def set_limits(xmin=None, xmax=None, ymin=None, ymax=None):
    limits = plt.axis()
    if xmin is None:
        xmin = limits[0]
    if xmax is None:
        xmax = limits[1]
    plt.xlim([float(xmin), float(xmax)])
    if ymin is None:
        ymin = limits[-2]
    if ymax is None:
        ymax = limits[-1]
    plt.ylim([float(ymin), float(ymax)])

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

def out(
    save,
    xmin,
    xmax,
    ymin,
    ymax,
    datastr,
    labels,
):
    set_limits(xmin, xmax, ymin, ymax)
    if len(labels) > 0:
        plt.legend()
    if save == "":
        plt.show()
    else:
        saveplot(save, datastr, labels)

def toint(x):
    try:
        x = int(x)
    except ValueError:
        pass
    return x

@app.command()
def plot(
    fields="x y",
    labels="",
    moving_avg:int=0,
    delimiter=None,
    fmt="",
    alpha:float=1.0,
    # output options
    save:str="",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
):
    """
    Plot y versus x as lines and/or markers, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    """
    data, datastr = read_data(delimiter)
    fields = fields.strip().split()
    labels = labels.strip().split()
    if fmt != "":
        fmt = fmt.strip().split()
    else:
        fmt = [fmt] * len(data)
    plotid = 0
    for i, f1 in enumerate(fields):
        if f1 == "x":
            x = np.float_(data[i])  # type: ignore
            if moving_avg > 0:
                x = np.convolve(x, np.ones((moving_avg,))/moving_avg, mode='valid')
        else:
            continue
        for j, f2 in enumerate(fields):
            if f2 == "y":
                y = np.float_(data[j])  # type: ignore
                if moving_avg > 0:
                    y = np.convolve(y, np.ones((moving_avg,))/moving_avg, mode='valid')
            else:
                continue
            if len(labels) > 0:
                label = labels[plotid]
            else:
                label = None
            plt.plot(x, y, fmt[plotid], label=label, alpha=alpha)
            plotid += 1
    out(save=save, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, datastr=datastr, labels=labels)

@app.command()
def hist(
    fields="y y",
    labels="",
    delimiter=None,
    bins="auto",
    alpha:float=1.0,
    # output options
    save:str="",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
):
    """
    Compute and plot a histogram, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    """
    fields = fields.strip().split()
    labels = labels.strip().split()
    data, datastr = read_data(delimiter)
    plotid = 0
    for j, f2 in enumerate(fields):
        if f2 == "y":
            y = np.float_(data[j])  # type: ignore
        else:
            continue
        if len(labels) > 0:
            label = labels[plotid]
        else:
            label = None
        plt.hist(y, toint(bins), label=label, alpha=1.0 - alpha)
        plotid += 1
    out(save=save, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, datastr=datastr, labels=labels)

@app.command()
def read_metadata(filename):
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
    def test_func(func:str):
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
