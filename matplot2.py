#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import socket
from sliding import Sliding_op
from numpy import linalg


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def renumber_fields(fields):
    outfields = []
    for field in fields:
        i = 0
        while f"{field}{i}" in outfields:
            i += 1
        outfields.append(f"{field}{i}")
    ndataset = i + 1
    print(f"{ndataset=}")
    return outfields, ndataset


def read_data(fields, delimiter):
    print("######## read_data ########")
    inp = np.genfromtxt(
        sys.stdin,
        dtype=str,
        delimiter=delimiter,
    )

    # Add the x data if not present
    if "x" not in fields:
        inp = np.vstack((np.arange(len(inp)), inp)).T
        fields = ["x"] + fields

    fields, ndataset = renumber_fields(fields)
    print(f"{fields=}")
    print(f"{inp.ndim=}")
    print(f"{len(inp)=}")
    data = dict()
    print(f"{inp.shape=}")
    for i, field in enumerate(fields):
        data[field] = inp[:, i]
        print(f"data['{field}']={data[field][:10]}...")
    print("###########################")
    return data, ndataset


def add_metadata(filename, datastr, key="data"):
    """
    Add metadata to a png file
    """
    metadata = PngInfo()
    metadata.add_text(key, datastr, zip=True)
    metadata.add_text("cwd", os.getcwd())
    metadata.add_text("hostname", socket.gethostname())
    # if options.subsample is not None:
    #     print("# Adding subsampling metadata")
    #     metadata.add_text("subsampling", "1st-column")
    # if options.labels is not None:
    #     metadata.add_text("labels", options.labels)
    targetImage = Image.open(filename)
    targetImage.save(filename, pnginfo=metadata)


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
    return datastr


def tofloat(arr):
    out = arr.copy()
    out = np.asarray([float(e) for e in out if e != ""])
    return out


def plot(data, ndataset):
    """
    Simple plot
    """
    for i in range(ndataset):
        x = data[f"x{i}"]
        y = data[f"y{i}"]
        x = tofloat(x)
        y = tofloat(y)
        plt.plot(x, y)


def scatter(data, ndataset):
    """
    Scatter plot
    """
    print("######## scatter ########")
    for dataset in range(ndataset):
        print(f"{dataset=}")
        x = data[f"x{dataset}"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        print(f"{x.shape=}")
        y = tofloat(y)
        print(f"{y.shape=}")
        if f"z{dataset}" in data:
            z = data[f"z{dataset}"]
            z = tofloat(z)
            print(f"{z.shape=}")
        else:
            z = None
        plt.scatter(x, y, c=z)
    print("#########################")


def moving_average(data, ndataset, window_size):
    print("######## moving_average ########")
    for dataset in range(ndataset):
        print(f"{dataset=}")
        x = data[f"x{dataset}"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        print(f"{x.shape=}")
        y = tofloat(y)
        print(f"{y.shape=}")
        plt.plot(x, y, color="gray", alpha=0.25)
        slmean = Sliding_op(y, window_size, np.mean, padding=True)
        ma = slmean.transform()
        plt.plot(x, ma)
    print("#########################")


def get_ellipse(center, width, height, angle):
    """
    see: https://stackoverflow.com/a/48409811/1679629
    center: x, y center of the ellipse
    width: first radius
    height: second radius
    angle: angle of the ellipse in degree
    """
    angle = np.deg2rad(angle)
    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([width * np.cos(t), height * np.sin(t)])
    R_rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    Ell = (center[0] + Ell_rot[0, :], center[1] + Ell_rot[1, :])
    return Ell


def plot_pca(data, ndataset):
    """
    Compute the pca for each dataset
    """
    print("######## plot_pca ########")
    for dataset in range(ndataset):
        print(f"{dataset=}")
        x = data[f"x{dataset}"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        print(f"{x.shape=}")
        y = tofloat(y)
        print(f"{y.shape=}")
        if f"z{dataset}" in data:
            z = data[f"z{dataset}"]
            z = tofloat(z)
            print(f"{z.shape=}")
        else:
            z = None
        scatter_obj = plt.scatter(x, y, c=z)
        if z is None:
            zlist = np.zeros(x.shape[0], dtype=int)
        else:
            zlist = z
        for zval in np.unique(zlist):
            print(f"{zval=}")
            sel = zlist == zval
            X = np.vstack((x[sel], y[sel])).T
            print(f"{X.shape=}")
            eigenvalues, eigenvectors, center, anglex = pca(X)
            width = 2 * np.sqrt(eigenvalues[0])
            height = 2 * np.sqrt(eigenvalues[1])
            print(f"{width=}")
            print(f"{height=}")
            ellipse = get_ellipse(center, width, height, anglex)
            if z is None:
                # color of the last scatter
                color = scatter_obj.get_facecolor()[0]
            else:
                cmap = plt.get_cmap(plt.get_cmap().name)
                color = cmap(zval)
            plt.scatter(
                center[0], center[1], marker="P", s=100, c=color, edgecolors="w"
            )
            plt.plot(
                ellipse[0],
                ellipse[1],
                color=color,
                path_effects=[pe.Stroke(linewidth=5, foreground="w"), pe.Normal()],
            )
            # see: https://stackoverflow.com/a/35762000/1679629
    print("##########################")


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
    anglex = np.rad2deg(np.arccos(eigenvectors[:, 0].dot(np.asarray([1, 0]))))
    print(f"{anglex=:.4g}")
    # angley = np.rad2deg(np.arccos(eigenvectors[:, 1].dot(np.asarray([0, 1]))))
    # print(f"{angley=:.4g}")
    if outfilename is not None:
        np.savez(outfilename, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    return eigenvalues, eigenvectors, center, anglex


def get_datastr(data):
    n = len(data["x0"])
    keys = data.keys()
    outstr = ""
    for k in keys:
        outstr += f"#{k} "
    outstr += "\n"
    for i in range(n):
        for k in keys:
            outstr += f"{data[k][i]} "
        outstr += "\n"
    print(f"{outstr[:80]=}...")
    return outstr


def save(outfilename, datastr):
    plt.savefig(outfilename)
    ext = os.path.splitext(outfilename)[1]
    print(f"{ext=}")
    if ext == ".png":
        add_metadata(outfilename, datastr)


if __name__ == "__main__":
    import doctest
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-f",
        "--fields",
        help="Fields for the data; e.g. 'x y x y'.\
        By default the first column is for x data and the other for y data.\
        If a 'z' field is given, this field is used to color the scatter dots.\
        If a 'e' field is given it is used to plot the error.\
        If a 'l' field is given this are the labels for the xticks -- xticklabels --.\
        If a 'w' field is given it is used as weights for weighted hostogram plotting (see: -H).\
        If a 't' field is given plot the given text at the given position (with x and y fields) using matplotlib.pyplot.text.\
        If a 'm' field is given, use it as markers (see: https://matplotlib.org/stable/api/markers_api.html).\
        If a 's' field is given, use it as a list of sizes for the markers.\
        If a 'c' field is given, use it as a list of colors (text color: r, g, b, y, ...) for the markers.\
        If --fields='*' is given all the columns are considered as y values.",
        default=["y"],
        nargs="+",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Scatter plot of the (x,y) data",
    )
    parser.add_argument(
        "--xlabel", dest="xlabel", default=None, type=str, help="x axis label"
    )
    parser.add_argument(
        "--ylabel", dest="ylabel", default=None, type=str, help="y axis label"
    )
    parser.add_argument(
        "--cmap",
        help="colormap to use. See: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html",
    )
    parser.add_argument(
        "-d", "--delimiter", help="Delimiter to use to read the data", default=None
    )
    parser.add_argument(
        "--moving_average",
        type=int,
        help="Plot a moving average on the data with the given window size",
    )
    parser.add_argument(
        "--pca",
        help="Compute and plot the Principal Component Analysis for each dataset and/or each z",
        action="store_true",
    )
    parser.add_argument("--save", help="Save the file", type=str)
    parser.add_argument(
        "--read_data",
        help="Read plot data from the given png saved image using the --save option",
    )
    args = parser.parse_args()

    if args.read_data is not None:
        DATASTR = read_metadata(args.read_data)
        print(DATASTR)
    if (
        not sys.stdin.isatty()
    ):  # stdin is not empty (see: https://stackoverflow.com/a/17735803/1679629)
        if args.cmap is not None:
            print(f"{args.cmap=}")
            plt.set_cmap(args.cmap)
        DATA, NDATASET = read_data(args.fields, delimiter=args.delimiter)
        DATASTR = get_datastr(DATA)
        if args.scatter:
            scatter(DATA, NDATASET)
        elif args.moving_average is not None:
            moving_average(DATA, NDATASET, window_size=args.moving_average)
        elif args.pca:
            plot_pca(DATA, NDATASET)
        else:
            plot(DATA, NDATASET)

        if args.xlabel is not None:
            plt.xlabel(args.xlabel)
        if args.ylabel is not None:
            plt.ylabel(args.ylabel)
        if args.save is None:
            plt.show()
        else:
            save(args.save, DATASTR)
