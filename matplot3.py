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
from sklearn.neighbors import KernelDensity

console = Console()

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
    datastr,
    labels,
    colorbar,
    xmin,
    xmax,
    ymin,
    ymax,
):
    set_limits(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    if colorbar:
        plt.colorbar()
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
    xmin:float=None,  # type:ignore
    xmax:float=None,  # type:ignore
    ymin:float=None,  # type:ignore
    ymax:float=None,  # type:ignore
    # test options
    test:bool=False,
    test_npts:int=1000,
    test_ndata:int=2,
):
    """
    Plot y versus x as lines and/or markers, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
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
        x = np.float_(data[xfield])  # type: ignore
        y = np.float_(data[yfield])  # type: ignore
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
        plt.plot(x, y, fmtstr, label=label, alpha=alpha)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

@app.command()
def scatter(
    fields="x y",
    labels="",
    delimiter=None,
    alpha:float=1.0,
    cmap:str="viridis",
    pcr:bool=False,
    # output options
    save:str="",
    xmin:float=None,  # type:ignore
    xmax:float=None,  # type:ignore
    ymin:float=None,  # type:ignore
    ymax:float=None,  # type:ignore
    colorbar:bool=False,
    # test options
    test:bool=False,
    test_npts:int=1000,
    test_ndata:int=2,
):
    """
    A scatter plot of y vs. x with varying marker size and/or color, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    --fields: x y c s (c: A sequence of numbers to be mapped to colors using cmap (see: --cmap), s: The marker size in points**2)

    --pcr: principal component regression (see: https://en.wikipedia.org/wiki/Principal_component_regression)

    --cmap: see: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps
    """
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
        x = np.float_(data[xfield])  # type: ignore
        y = np.float_(data[yfield])  # type: ignore
        if len(labels) > 0:
            label = labels[0]
        else:
            label = None
        if len(s_indices) > 0:
            s = np.float_(data[s_indices[0]])  # type: ignore
        else:
            s = None
        if len(c_indices) > 0:
            c = np.float_(data[c_indices[0]])  # type: ignore
        else:
            c = None
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))  # type:ignore
        plt.scatter(x, y, s=s, c=c, label=label, alpha=alpha, cmap=cmap)
        if pcr:
            do_pcr(x,y)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

@app.command()
def hist(
    fields="y",
    labels="",
    delimiter=None,
    bins="auto",
    alpha:float=1.0,
    # output options
    save:str="",
    xmin:float=None, # type:ignore
    xmax:float=None, # type:ignore
    ymin:float=None, # type:ignore
    ymax:float=None, # type:ignore
    # test options
    test:bool=False,
    test_npts:int=1000,
    test_ndata:int=2,
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
            y = np.float_(data[j])  # type: ignore
        else:
            continue
        if len(labels) > 0:
            label = labels[plotid]
        else:
            label = None
        plt.hist(y, toint(bins), label=label, alpha=alpha)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

@app.command()
def jitter(
    fields="x y",
    labels="",
    delimiter=None,
    xjitter:float=0.1,
    yjitter:float=0.0,
    size:int=10,
    alpha:float=1.0,
    kde:bool=False,
    kde_subset:int=1000,
    cmap:str="viridis",
    # output options
    save:str="",
    xmin:float=None,  # type:ignore
    xmax:float=None,  # type:ignore
    ymin:float=None,  # type:ignore
    ymax:float=None,  # type:ignore
    # test options
    test:bool=False,
    test_npts:int=1000,
    test_ndata:int=3,
):
    """
    Jitter plot
    """
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
    kde_y = None
    plotid = 0
    for xfield, yfield in track(zip(xfields, yfields), total=len(xfields), description="Jittering..."):
        x = np.float_(data[xfield])  # type: ignore
        y = np.float_(data[yfield])  # type: ignore
        if kde:
            kde_ins = KernelDensity(kernel="gaussian", bandwidth="scott").fit(np.random.choice(y, size=min(kde_subset, len(y)))[:, None])  # type: ignore
            # kde_ins = kde_ins.fit(y[:, None])  # type: ignore
            kde_y = np.exp(kde_ins.score_samples(y[:, None]))  # type: ignore
        x += np.random.normal(size=x.shape, loc=0, scale=xjitter)
        y += np.random.normal(size=y.shape, loc=0, scale=yjitter)
        plt.subplot(SUBPLOTS[0], SUBPLOTS[1], min(plotid+1, SUBPLOTS[0]*SUBPLOTS[1]))  # type:ignore
        plt.scatter(x, y, c=kde_y, s=size, alpha=alpha, cmap=cmap)
        plotid += 1
    out(save=save, datastr=datastr, labels=labels, colorbar=False, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

@app.command()
def umap(
    n_neighbors:int=15,
    min_dist:float=0.1,
    metric:str="euclidean",
    test:bool=False,
    save:str="",
    npy:str="",
    npz:str="",
    data_key:str="data",
    labels_key:str="",
    colorbar:bool=False,
    cmap:str="viridis",
    size:int=10,
    alpha:float=1.0,
    xmin:float=None,  # type:ignore
    xmax:float=None,  # type:ignore
    ymin:float=None,  # type:ignore
    ymax:float=None,  # type:ignore
):
    """
    UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction technique.
    See: https://umap-learn.readthedocs.io/en/latest/

    n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.

    min_dist: The effective minimum distance between embedded points.

    metric: The metric to use to compute distance in high dimensional space (default: euclidean, precomputed, cosine, manhattan, hamming, etc.)

    test: Generate random data for testing

    save: Save the plot to a file

    npy: Load data from a numpy file

    npz: Load data from a numpy file (compressed)

    data_key: The key to use to load data from the npz file

    labels_key: The key to use to load labels from the npz file

    colorbar: Add a colorbar to the plot

    cmap: The colormap to use for the plot

    size: The size of the markers in the plot

    alpha: The transparency of the markers in the plot

    xmin: The minimum x value for the plot

    xmax: The maximum x value for the plot

    ymin: The minimum y value for the plot

    ymax: The maximum y value for the plot
    """
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
    print(f"{data.shape=}")
    print(f"{data.min()=}")
    print(f"{data.max()=}")
    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = mapper.fit_transform(data)
    # umap.plot.points(mapper, values=r_orig)
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=size, cmap=cmap, alpha=alpha)
    else:
        for label in np.unique(labels):
            sel = labels == label
            plt.scatter(embedding[sel, 0], embedding[sel, 1], s=size, cmap=cmap, alpha=alpha, label=label)
    out(save=save, datastr="", labels=labels, colorbar=colorbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


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
