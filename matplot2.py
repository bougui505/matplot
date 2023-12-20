#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

import os
import socket
import sys

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from sliding import Sliding_op


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
    if args.labels is not None:
        metadata.add_text(f"labels", " ".join(args.labels))
    metadata.add_text("cwd", os.getcwd())
    metadata.add_text("hostname", socket.gethostname())
    # if options.subsample is not None:
    #     print("# Adding subsampling metadata")
    #     metadata.add_text("subsampling", "1st-column")
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
    # out = arr.copy()
    # out = np.asarray([float(e) for e in out if e != ""])
    out = []
    for e in arr:
        try:
            v = float(e)
        except ValueError:
            v = np.nan
        out.append(v)
    out = np.asarray(out)
    return out


def plot(
    data,
    ndataset,
    labels=None,
    extremas=None,
    subplots=None,
    subplots_assignment=None,
    xlabels=None,
    ylabels=None,
    semilog=None,
    ymin=None,
    ymax=None,
    xmin=None,
    xmax=None,
    title=None,
):
    """
    Simple plot
    """
    print("######## plot ########")
    if subplots is not None:
        print(f"{subplots=}")
    ymin = _broadcast_(ymin, ndataset)
    ymax = _broadcast_(ymax, ndataset)
    xmin = _broadcast_(xmin, ndataset)
    xmax = _broadcast_(xmax, ndataset)
    if subplots_assignment is None:
        subplots_assignment = range(ndataset)
    for dataset in range(ndataset):
        if subplots is not None:
            _setup_subplot_(subplots,
                            subplots_assignment[dataset],
                            title=title,
                            xlabels=xlabels,
                            ylabels=ylabels)
        x = data[f"x{dataset}"] if f"x{dataset}" in data else data[f"x{0}"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        y = tofloat(y)
        print(f"{x=}")
        print(f"{y=}")
        label = labels[dataset] if labels is not None else None
        print(f"{label=}")
        pltobj = plt.plot(x, y, label=label)
        plot_extremas(extremas, dataset, y, pltobj)
        if labels is not None:
            plt.legend()
        if subplots is not None:
            if semilog is not None:
                if "x" in semilog:
                    plt.xscale("log")
                if "y" in semilog:
                    plt.yscale("log")
            set_x_lim(xmin[dataset], xmax[dataset])
            set_y_lim(ymin[dataset], ymax[dataset])
    print("######################")


def apply_repulsion(
    repulsion,
    x,
    ndims=2,
    niter=10000,
    device="cpu",
    min_delta=1e-6,
    return_np=True,
    verbose=True,
):
    import torch

    class Mover(torch.nn.Module):

        def __init__(self, npts, ndims=2):
            super().__init__()
            self.delta = torch.nn.Parameter(torch.randn(size=(npts, ndims)))

        def forward(self, x):
            return x + self.delta

    def lossfunc(x, repulsion):
        xmat = torch.cdist(x, x)
        repulsive_mask = xmat < repulsion
        loss_rep = torch.mean((xmat[repulsive_mask] - repulsion)**2)
        return loss_rep

    npts = x.shape[0]
    mover = Mover(npts=npts, ndims=ndims).to(device)
    optimizer = torch.optim.Adam(mover.parameters(), amsgrad=False, lr=0.01)
    y = x
    loss = torch.inf
    loss_prev = torch.inf
    for i in range(niter):
        optimizer.zero_grad()
        y = mover(x)
        loss = lossfunc(y, repulsion=repulsion)
        loss.backward()
        optimizer.step()
        progress = (i + 1) / niter
        delta_loss = torch.abs(loss - loss_prev)
        loss_prev = loss
        if verbose:
            print(f"{i=}")
            print(f"{progress=:.2%}")
            print(f"{loss=:.5g}")
            print(f"{delta_loss=:.5g}")
            print("--")
        if delta_loss <= min_delta:
            break
    if return_np:
        return y.detach().cpu().numpy(), loss
    else:
        return y.detach(), loss


def add_repulsion(x, y, repulsion):
    import scipy.spatial.distance as scidist
    import torch

    coords = np.c_[x, y]
    coords = torch.from_numpy(coords)
    min_delta = 1e-6
    coords, loss = apply_repulsion(x=coords,
                                   repulsion=repulsion,
                                   min_delta=min_delta,
                                   niter=1000)
    return coords[:, 0], coords[:, 1]


def graph(data, ndataset, size=20, labels=None, fontsize="medium"):
    print("######## GRAPH ########")
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
            scatter_markers(x=x,
                            y=y,
                            z=z,
                            markers=markers,
                            size=size,
                            labels=labels)
        plot_texts(data, dataset, fontsize=fontsize)
        edge_labels = np.unique(data[f"ed{dataset}"])
        for edge_label in edge_labels:
            sel = (data[f"ed{dataset}"] == edge_label)
            plt.plot(x[sel], y[sel], 'k-')
    print("#######################")

def scatter(data,
            ndataset,
            size=20,
            labels=None,
            fontsize="medium",
            repulsion=0.):
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
        if repulsion > 0:
            x, y = add_repulsion(x, y, repulsion=repulsion)
            data[f"x{dataset}"] = x
            data[f"y{dataset}"] = y
        if f"z{dataset}" in data:
            z = data[f"z{dataset}"]
            z = tofloat(z)
            print(f"{z.shape=}")
        else:
            z = None
        plot_texts(data, dataset, fontsize=fontsize)
        if f"m{dataset}" not in data:
            plt.scatter(x, y, c=z, s=size)
        else:
            markers = data[f"m{dataset}"]
            scatter_markers(x=x,
                            y=y,
                            z=z,
                            markers=markers,
                            size=size,
                            labels=labels)
    if labels is not None:
        plt.legend()
    print("#########################")


def histogram(data, ndataset, labels=None, alpha=1.0, bins=None, normed=False):
    print("######## histogram ########")
    if bins is None:
        bins = "auto"
    print(f"{bins=}")
    for dataset in range(ndataset):
        print(f"{dataset=}")
        y = data[f"y{dataset}"]
        y = tofloat(y)
        y = y[~np.isinf(y)]
        print(f"{y.shape=}")
        if labels is not None:
            label = labels[dataset]
        else:
            label = None
        plt.hist(y, label=label, alpha=alpha, bins=bins, density=normed)
    if labels is not None:
        plt.legend()
    print("#########################")


def plot_texts(data, dataset, fontsize):
    if f"t{dataset}" in data:
        texts = data[f"t{dataset}"]
        x = data[f"x{dataset}"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        y = tofloat(y)
        for xval, yval, t in zip(x, y, texts):
            if t != "-":
                plttext = plt.text(x=xval,
                                   y=yval,
                                   s=t,
                                   fontsize=fontsize,
                                   zorder=101)
                # Add white line around text
                plttext.set_path_effects(
                    [pe.withStroke(linewidth=2, foreground="w")])


KNOWN_LABELS = set()


def scatter_markers(x,
                    y,
                    z=None,
                    markers=None,
                    size=None,
                    color=None,
                    labels=None):
    markers_unique = np.unique(markers)
    size_ori = size
    for i, marker in enumerate(markers_unique):
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
            # edgecolors = "w"
            # size = 100
        else:
            zorder = None
            edgecolors = None
            # size = size_ori
        if labels is not None:
            label = labels[i]
        else:
            label = None
        if label in KNOWN_LABELS:
            label = None
        KNOWN_LABELS.add(label)
        try:
            s = size[sel]
        except TypeError:
            s = size
        out = plt.scatter(
            x[sel],
            y[sel],
            c=c,
            s=s,
            marker=marker,
            color=color,
            zorder=zorder,
            edgecolors=edgecolors,
            label=label,
        )
    return out


def _broadcast_(inp, n):
    if not isinstance(inp, list):
        return [inp] * n
    if len(inp) == 1 and isinstance(inp, list):
        return inp * n
    return inp


def _setup_subplot_(subplots, dataset, title=None, xlabels=None, ylabels=None):
    if title is not None:
        plt.title(title)
    subplot = subplots + [(dataset + 1)]
    print(f"{subplot=}")
    plt.subplot(*subplot)
    if xlabels is not None:
        plt.xlabel(xlabels[dataset])
    if ylabels is not None:
        plt.ylabel(ylabels[dataset])


def plot_extremas(extremas, dataset, ydata, pltobj):
    if extremas is not None:
        assert dataset < len(
            extremas
        ), f"Number of extrema keyword given to option --extrema ({len(args.extrema)}) does not match the number of dataset for current dataset ({dataset+1})"
    extrema = extremas[dataset] if extremas is not None else None
    if extrema == "min":
        v = np.nanmin(ydata)
        xv = np.nanargmin(ydata)
    else:
        v = np.nanmax(ydata)
        xv = np.nanargmax(ydata)
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
        plt.axvline(x=xv,
                    color=color,
                    linestyle="dotted",
                    linewidth=1.0,
                    label=xv)


def moving_average(
    data,
    ndataset,
    window_size,
    labels,
    extremas,
    subplots=None,
    subplots_assignment=None,
    xlabels=None,
    ylabels=None,
    semilog=None,
    ymin=None,
    ymax=None,
    xmin=None,
    xmax=None,
    title=None,
):
    print("######## moving_average ########")

    if subplots is not None:
        print(f"{subplots=}")
    ymin = _broadcast_(ymin, ndataset)
    ymax = _broadcast_(ymax, ndataset)
    xmin = _broadcast_(xmin, ndataset)
    xmax = _broadcast_(xmax, ndataset)
    if subplots_assignment is None:
        subplots_assignment = range(ndataset)
    for dataset in range(ndataset):
        if subplots is not None:
            _setup_subplot_(subplots,
                            subplots_assignment[dataset],
                            title=title,
                            xlabels=xlabels,
                            ylabels=ylabels)
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
        plot_extremas(extremas, dataset, ma, pltobj)
        if labels is not None:
            plt.legend()
        if subplots is not None:
            if semilog is not None:
                if "x" in semilog:
                    plt.xscale("log")
                if "y" in semilog:
                    plt.yscale("log")
            set_x_lim(xmin[dataset], xmax[dataset])
            set_y_lim(ymin[dataset], ymax[dataset])
        print("--")
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
    R_rot = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
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
    v_squared = np.dot(Phi.T, mu_A - mu_B)**2
    res = minimize_scalar(K_function,
                          bracket=[0.0, 0.5, 1.0],
                          args=(lambdas, v_squared, tau))
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
    return 1.0 - (1.0 / tau**2) * np.sum(v_squared * ((s * (1.0 - s)) /
                                                      (1.0 + s *
                                                       (lambdas - 1.0))))


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


def order_z_per_variance(data, ndataset):
    zorders = []
    for dataset in range(ndataset):
        x = data[f"x{dataset}"]
        y = data[f"y{dataset}"]
        x = tofloat(x)
        y = tofloat(y)
        z = data[f"z{dataset}"]
        z = tofloat(z)
        eigenvalues_list = []
        for zval in np.unique(z):
            sel = z == zval
            X = np.vstack((x[sel], y[sel])).T
            eigenvalues, eigenvectors, center, anglex = pca(X)
            eigenvalues_list.append(eigenvalues[0])
        zsorter = np.argsort(eigenvalues_list)
        zorder = np.unique(z)[zsorter]
        print(f"{zorder=}")
        zorders.append(zorder)
    return zorders


def plot_pca(data,
             ndataset,
             plot_overlap=True,
             scale=1.0,
             size=20.0,
             labels=None):
    """
    Compute the pca for each dataset
    scale: scale of the ellipses
    """
    print("######## plot_pca ########")
    Sigma_Alist = []
    centerlist = []
    cmap = plt.get_cmap(plt.get_cmap().name)
    if "z0" in data:
        zorders = order_z_per_variance(data, ndataset)
    else:
        zorders = None
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
        if zorders is not None:
            zunique = zorders[dataset]
        else:
            zunique = np.unique(zlist)
        for i, zval in enumerate(zunique):
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
                    plt.scatter(x[sel],
                                y[sel],
                                color="gray",
                                alpha=0.125,
                                zorder=-1,
                                s=size)
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
                    x=x[sel],
                    y=y[sel],
                    markers=markers,
                    color=color,
                    size=size,
                    labels=labels,
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
    if labels is not None:
        plt.legend()
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
        path_effects=[pe.Stroke(linewidth=5, foreground="w"),
                      pe.Normal()],
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
        np.arccos(eigenvectors[:, 0].dot(np.asarray([1, 0]))))
    print(f"{anglex=:.4g}")
    # angley = np.rad2deg(np.arccos(eigenvectors[:, 1].dot(np.asarray([0, 1]))))
    # print(f"{angley=:.4g}")
    if outfilename is not None:
        np.savez(outfilename,
                 eigenvalues=eigenvalues,
                 eigenvectors=eigenvectors)
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


def set_x_lim(xmin, xmax):
    axes = plt.gca()
    limits = plt.axis()
    if xmin is None:
        xmin = limits[0]
    if xmax is None:
        xmax = limits[1]
    axes.set_xlim([xmin, xmax])


def set_y_lim(ymin: float, ymax: float):
    axes = plt.gca()
    limits = plt.axis()
    if ymin is None:
        ymin = limits[-2]
    if ymax is None:
        ymax = limits[-1]
    axes.set_ylim([ymin, ymax])


if __name__ == "__main__":
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
        If a 'ed' field is given this is the edge label for the graph see --graph option \
        If --fields='*' is given all the columns are considered as y values.",
        default=["y"],
        choices=["x", "y", "z", "e", "l", "w", "t", "m", "s", "c", "ed", "*"],
        nargs="+",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Scatter plot of the (x,y) data",
    )
    parser.add_argument(
        "--repulsion",
        help="Add a repulsion parameter between points in scatter plot to \
        avoid overlap. Repulsion is a distance between points. \
        Warnings: dependencies to pytorch",
        default=0.0,
        type=float)
    parser.add_argument(
        "-s",
        "--size",
        default=20.0,
        type=float,
        help="size of the dots for the scatter plot (default: 20.0)",
    )
    parser.add_argument(
        "--xlabel",
        dest="xlabel",
        default=None,
        type=str,
        help="x axis label. Possibly one for each subplot",
        nargs="+",
    )
    parser.add_argument(
        "--ylabel",
        dest="ylabel",
        default=None,
        type=str,
        help="y axis label. Possibly one for each subplot",
        nargs="+",
    )
    parser.add_argument(
        "--cmap",
        help="colormap to use. See: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html",
    )
    parser.add_argument("-d",
                        "--delimiter",
                        help="Delimiter to use to read the data",
                        default=None)
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
    parser.add_argument("--orthonormal",
                        help="Set an orthonormal basis",
                        action="store_true")
    parser.add_argument("--title", help="Title of the plot", type=str)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="List of labels for each dataset defined with the --fields option. For scatter plots with different markers one label per marker can be given.",
    )
    parser.add_argument(
        "--extrema",
        help="List of keyword 'min' 'max' for each dataset to plot an horizontal line for minima or maxima respectively",
        nargs="+",
        choices=["min", "max"],
    )
    parser.add_argument(
        "--subplots",
        nargs="+",
        help="Print each dataset in a different subplot. Give the number of rows and columns for subplot layout.",
        type=int,
    )
    parser.add_argument(
        "-sp",
        "--sp_assignment",
        nargs="+",
        help="Assign each dataset to a specific subplot. By default, 1 dataset per subplot.",
        type=int)
    parser.add_argument(
        "--semilog",
        help="Log scale for the given axis (x and/or y)",
        nargs="+",
        choices=["x", "y"],
        default=[None, None],
    )
    parser.add_argument(
        "-H",
        "--histogram",
        action="store_true",
        help="Compute and plot histogram from data",
    )
    parser.add_argument(
        "--normed",
        help="If True, the first element of the return tuple will be the counts normalized to form a probability density",
        action="store_true"
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        help="Number of bins in the histogram",
    )
    parser.add_argument("--graph",
                        help="Plot a graph. The input data are x y e where e is the edge index. Points with the same edge index are linked by a line",
                        action="store_true")
    parser.add_argument("--alpha",
                        type=float,
                        default=1.0,
                        help="Transparency")
    parser.add_argument(
        "--ymin",
        type=float,
        help="Lower limit for y-axis. If subplots are on, give one value per subplot",
        nargs="+",
        default=[None],
    )
    parser.add_argument(
        "--ymax",
        type=float,
        help="Upper limit for y-axis. If subplots are on, give one value per subplot",
        nargs="+",
        default=[None],
    )
    parser.add_argument(
        "--xmin",
        type=float,
        help="Lower limit for x-axis. If subplots are on, give one value per subplot",
        nargs="+",
        default=[None],
    )
    parser.add_argument(
        "--xmax",
        type=float,
        help="Upper limit for x-axis. If subplots are on, give one value per subplot",
        nargs="+",
        default=[None],
    )
    parser.add_argument(
        "--fontsize",
        default="medium",
        type=str,
        help="The font size for the legend and the text plot. If the value is numeric the size will be the absolute font size in points. String values are relative to the current default font size. int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. Default: 'medium'",
    )
    parser.add_argument("--colorbar",
                        help="Display the colorbar",
                        action="store_true")
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
    if "x" in args.semilog:
        plt.xscale("log")
    if "y" in args.semilog:
        plt.yscale("log")
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
        if "s0" in DATA:
            args.size = np.float_(DATA["s0"])
        if args.scatter:
            scatter(DATA,
                    NDATASET,
                    size=args.size,
                    labels=args.labels,
                    fontsize=args.fontsize,
                    repulsion=args.repulsion)
        elif args.moving_average is not None:
            moving_average(
                DATA,
                NDATASET,
                window_size=args.moving_average,
                labels=args.labels,
                extremas=args.extrema,
                subplots=args.subplots,
                subplots_assignment=args.sp_assignment,
                xlabels=args.xlabel,
                ylabels=args.ylabel,
                semilog=args.semilog,
                ymin=args.ymin,
                ymax=args.ymax,
                xmin=args.xmin,
                xmax=args.xmax,
                title=args.title,
            )
        elif args.pca:
            plot_pca(DATA,
                     NDATASET,
                     plot_overlap=True,
                     scale=args.scale,
                     size=args.size)
        elif args.no_overlap:
            plot_pca(
                DATA,
                NDATASET,
                plot_overlap=False,
                scale=args.scale,
                size=args.size,
                labels=args.labels,
            )
        elif args.histogram:
            histogram(DATA,
                      NDATASET,
                      labels=args.labels,
                      alpha=args.alpha,
                      bins=args.bins,
                      normed=args.normed)
        elif args.graph:
            graph(DATA,
                  NDATASET,
                  size=args.size,
                  labels=args.labels,
                  fontsize=args.fontsize
                  )
        else:
            plot(
                DATA,
                NDATASET,
                labels=args.labels,
                extremas=args.extrema,
                subplots=args.subplots,
                subplots_assignment=args.sp_assignment,
                xlabels=args.xlabel,
                ylabels=args.ylabel,
                semilog=args.semilog,
                ymin=args.ymin,
                ymax=args.ymax,
                xmin=args.xmin,
                xmax=args.xmax,
                title=args.title,
            )

        if args.xlabel is not None:
            plt.xlabel(args.xlabel[-1])
        if args.ylabel is not None:
            plt.ylabel(args.ylabel[-1])
        if (args.ymin is not None
                or args.ymax is not None) and args.subplots is None:
            set_y_lim(args.ymin[0], args.ymax[0])
        if (args.xmin is not None
                or args.xmax is not None) and args.subplots is None:
            set_x_lim(args.xmin[0], args.xmax[0])
        if args.colorbar:
            plt.colorbar()
        if args.save is None:
            plt.show()
        else:
            save(args.save, DATASTR)
