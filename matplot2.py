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
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh


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
    print(f"{inp.shape=}")

    # Add the x data if not present
    if "x" not in fields:
        # inp = np.vstack((np.arange(len(inp))[None, :], inp)).T
        inp = np.c_[np.arange(len(inp)), inp]
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
        x = data[f"x{i}"] if f"x{i}" in data else data[f"x{0}"]
        y = data[f"y{i}"]
        x = tofloat(x)
        y = tofloat(y)
        plt.plot(x, y)


def scatter(data, ndataset, size=20):
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
        if f"m{dataset}" not in data:
            plt.scatter(x, y, c=z, s=size)
        else:
            markers = data[f"m{dataset}"]
            scatter_markers(x=x, y=y, z=z, markers=markers, size=size)
    print("#########################")


def scatter_markers(x, y, z=None, markers=None, size=None, color=None):
    markers_unique = np.unique(markers)
    size_ori = size
    for marker in markers_unique:
        sel = markers == marker
        if len(marker) > 1:
            color = marker[0]
            marker = marker[1]
        if z is not None:
            c = z[sel]
        else:
            c = None
        if color is not None:
            c = None
        if marker != "o":
            zorder = 100
            edgecolors = "w"
            size = 100
        else:
            zorder = None
            edgecolors = None
            size = size_ori
        out = plt.scatter(
            x[sel],
            y[sel],
            c=c,
            s=size,
            marker=marker,
            color=color,
            zorder=zorder,
            edgecolors=edgecolors,
        )
    return out


def moving_average(data, ndataset, window_size, labels, extremas):
    print("######## moving_average ########")
    for dataset in range(ndataset):
        print(f"{dataset=}")
        x = data[f"x{dataset}"] if f"x{dataset}" in data else data["x0"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        print(f"{x.shape=}")
        y = tofloat(y)
        print(f"{y.shape=}")
        plt.plot(x, y, color="gray", alpha=0.25)
        slmean = Sliding_op(y, window_size, np.mean, padding=True)
        ma = slmean.transform()
        label = labels[dataset] if labels is not None else None
        print(f"{label=}")
        pltobj = plt.plot(x, ma, label=label)
        extrema = extremas[dataset] if extremas is not None else None
        if extrema == "min":
            v = ma.min()
        else:
            v = ma.max()
        if extrema is not None:
            # color of the last plot
            color = pltobj[0].get_color()
            plt.axhline(
                y=v,
                color=color,
                linestyle="--",
                linewidth=1.0,
                label=f"{extrema}={v:.2g}",
            )
        print("--")
    if labels is not None:
        plt.legend()
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


def ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B, tau=1.0):
    """
    Test if two ellipses intersect
    Adapted from: https://math.stackexchange.com/a/3678498/192193
    """
    lambdas, Phi = eigh(Sigma_A, b=Sigma_B)
    v_squared = np.dot(Phi.T, mu_A - mu_B) ** 2
    res = minimize_scalar(
        K_function, bracket=[0.0, 0.5, 1.0], args=(lambdas, v_squared, tau)
    )
    # print(f"{res.fun=}")
    return res.fun >= 0


def batch_ellipsoid_intersection_test(Sigma_A, Sigma_B_list, mu_A, mu_B_list):
    n = len(Sigma_B_list)
    for i in range(n):
        Sigma_B = Sigma_B_list[i]
        mu_B = mu_B_list[i]
        intersect = ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B)
        if intersect:
            return True
    return False


def K_function(s, lambdas, v_squared, tau):
    return 1.0 - (1.0 / tau**2) * np.sum(
        v_squared * ((s * (1.0 - s)) / (1.0 + s * (lambdas - 1.0)))
    )


def get_ellipse_sigma_mat(u1u2, sigma1, sigma2):
    """
    Get the sigma matrix from the ellipse
    See: https://math.stackexchange.com/a/3678498/192193
    """
    S = np.eye(2)
    S[0, 0] = 1.0 / sigma1**2
    S[1, 1] = 1.0 / sigma2**2
    A = u1u2.dot(S).dot(u1u2.T)
    Sigma_A = linalg.inv(A)
    return Sigma_A


def plot_pca(data, ndataset, plot_overlap=True, scale=1.0, size=20.0):
    """
    Compute the pca for each dataset
    scale: scale of the ellipses
    """
    print("######## plot_pca ########")
    Sigma_Alist = []
    centerlist = []
    cmap = plt.get_cmap(plt.get_cmap().name)
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
        # scatter_obj = plt.scatter(x, y, c=z)
        if z is None:
            zlist = np.zeros(x.shape[0], dtype=int)
        else:
            zlist = z
        zmax = zlist.max()
        if zmax == 0:
            zmax = 1
        for zval in np.unique(zlist):
            print(f"{zval=}")
            sel = zlist == zval
            X = np.vstack((x[sel], y[sel])).T
            print(f"{X.shape=}")
            eigenvalues, eigenvectors, center, anglex = pca(X)
            print(f"{eigenvalues=}")
            width = 2 * np.sqrt(eigenvalues[0]) * scale
            height = 2 * np.sqrt(eigenvalues[1]) * scale
            ax1 = np.squeeze(eigenvectors[:, 0] * width)
            ax2 = np.squeeze(eigenvectors[:, 1] * height)
            ellipse = get_ellipse(center, width, height, anglex)
            print(f"{width=}")
            print(f"{height=}")
            Sigma_A = get_ellipse_sigma_mat(eigenvectors, width, height)
            intersect = batch_ellipsoid_intersection_test(
                Sigma_A=Sigma_A,
                Sigma_B_list=Sigma_Alist,
                mu_A=center,
                mu_B_list=centerlist,
            )
            print(f"{intersect=}")
            if not plot_overlap:
                if intersect:
                    plt.scatter(
                        x[sel], y[sel], color="gray", alpha=0.125, zorder=-1, s=size
                    )
                    continue
            if z is None:
                color = None
            else:
                color = cmap(zval / zmax)
            if f"m{dataset}" not in data:
                scatter_obj = plt.scatter(x[sel], y[sel], color=color, s=size)
            else:
                markers = data[f"m{dataset}"][sel]
                scatter_obj = scatter_markers(
                    x=x[sel], y=y[sel], markers=markers, color=color, size=size
                )
            Sigma_Alist.append(Sigma_A)
            centerlist.append(center)
            if not plot_overlap:
                if intersect:
                    continue
            if z is None:
                # color of the last scatter
                color = scatter_obj.get_facecolor()[0]
            else:
                color = cmap(zval / zmax)
            plot_ellipse(ellipse, color, center=center, ax1=ax1, ax2=ax2)
            print("--")
    print("##########################")


def plot_ellipse(ellipse, color, center, ax1=None, ax2=None):
    plt.scatter(
        center[0],
        center[1],
        marker="P",
        s=100,
        color=color,
        edgecolors="w",
        zorder=99,
    )
    plt.plot(
        ellipse[0],
        ellipse[1],
        color=color,
        path_effects=[pe.Stroke(linewidth=5, foreground="w"), pe.Normal()],
    )
    # see: https://stackoverflow.com/a/35762000/1679629
    # Print axis (bug? FIXME)
    if center is not None and ax1 is not None:
        xc, yc = center
        x1, y1 = center + ax1
        plt.plot([xc, x1], [yc, y1], color="k", lw=1)
    if center is not None and ax2 is not None:
        xc, yc = center
        x2, y2 = center + ax2
        plt.plot([xc, x2], [yc, y2], color="k", lw=1)


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
        np.arccos(eigenvectors[:, 0].dot(np.asarray([1, 0])))
    )
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
        For the 'm' field 2 characters could be given. The first one is the color, the second the marker. E.g. 'r*' to plot a red star marker\
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
        "-s",
        "--size",
        default=20.0,
        type=float,
        help="size of the dots for the scatter plot (default: 20.0)",
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
    parser.add_argument(
        "--no_overlap",
        help="Compute and plot the Principal Component Analysis for each dataset and/or each z and do not plot overlapping data (from z or datasets)",
        action="store_true",
    )
    parser.add_argument(
        "--scale",
        help="scale for the ellipses in the pca plot (--pca) and the --no_overlap plot",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--aspect_ratio",
        help="Change the aspect ratio of the figure",
        nargs=2,
        type=int,
    )
    parser.add_argument(
        "--orthonormal", help="Set an orthonormal basis", action="store_true"
    )
    parser.add_argument("--title", help="Title of the plot", type=str)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="List of labels for each dataset defined with the --fields option",
    )
    parser.add_argument(
        "--extrema",
        help="List of keyword 'min' 'max' for each dataset to plot an horizontal line for minima or maxima respectively",
        nargs="+",
        choices=["min", "max"],
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
    if args.aspect_ratio is not None:
        print(f"{args.aspect_ratio=}")
        plt.figure(figsize=(args.aspect_ratio[0], args.aspect_ratio[1]))
    if args.orthonormal:
        plt.axis("equal")
    if (
        not sys.stdin.isatty()
    ):  # stdin is not empty (see: https://stackoverflow.com/a/17735803/1679629)
        if args.cmap is not None:
            print(f"{args.cmap=}")
            plt.set_cmap(args.cmap)
        if args.title is not None:
            print(f"{args.title=}")
            plt.title(args.title)
        DATA, NDATASET = read_data(args.fields, delimiter=args.delimiter)
        DATASTR = get_datastr(DATA)
        if args.scatter:
            scatter(DATA, NDATASET, size=args.size)
        elif args.moving_average is not None:
            moving_average(
                DATA,
                NDATASET,
                window_size=args.moving_average,
                labels=args.labels,
                extremas=args.extrema,
            )
        elif args.pca:
            plot_pca(
                DATA, NDATASET, plot_overlap=True, scale=args.scale, size=args.size
            )
        elif args.no_overlap:
            plot_pca(
                DATA, NDATASET, plot_overlap=False, scale=args.scale, size=args.size
            )
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
