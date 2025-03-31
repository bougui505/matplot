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
from PIL import Image
from PIL.PngImagePlugin import PngInfo

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
)

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

@app.command()
def plot(
    fields:str="x y",
    delimiter:str=None,  # type: ignore
    save:str="",  # type: ignore
    xlabel:str="x",
    ylabel:str="y",
    labels:str="",
    semilog_x:bool=False,
    semilog_y:bool=False,
):
    """"""
    fields = fields.strip().split()  # type: ignore
    labels = labels.strip().split()  # type: ignore
    data, datastr = read_data(delimiter)
    plotid = 0
    for i, f1 in enumerate(fields):
        if f1 == "x":
            x = np.float_(data[i])  # type: ignore
        else:
            continue
        for j, f2 in enumerate(fields):
            if f2 == "y":
                y = np.float_(data[j])  # type: ignore
            else:
                continue
            if len(labels) > 0:
                label = labels[plotid]
            else:
                label = None
            plt.plot(x, y, label=label)
            plotid += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if semilog_x:
        plt.semilogx()
    if semilog_y:
        plt.semilogy()
    if len(labels) > 0:
        plt.legend()
    if save == "":
        plt.show()
    else:
        saveplot(save, datastr)

def saveplot(outfilename, datastr):
    plt.savefig(outfilename)
    ext = os.path.splitext(outfilename)[1]
    print(f"{ext=}")
    print(f"{outfilename=}")
    if ext == ".png":
        add_metadata(outfilename, datastr)

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
