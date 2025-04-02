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
from rich.progress import track
from sklearn.neighbors import KernelDensity

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
):
    """
    Read data from stdin and plot them.

    If an empty line is found, a new dataset is created.

    Setup the plot with the given parameters

    --aspect_ratio: "16 9", set the aspect ratio of the plot
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if semilog_x:
        plt.semilogx()
    if semilog_y:
        plt.semilogy()
    if grid:
        plt.grid()
    if aspect_ratio is not None:
        xaspect, yaspect = aspect_ratio.split()
        plt.figure(figsize=(float(xaspect), float(yaspect)))

def read_data(delimiter, fields):
    """
    Read data from stdin and return a dictionary of data
    if an empty line is found, new fields are created
    """
    data = defaultdict(list)
    datastr = ""
    imax = -1
    field_offset = 0
    for line in sys.stdin.readlines():
        line = line.strip().split(delimiter)
        # check if the line is empty
        if len(line) == 0:
            # create new fields
            field_offset = imax + 1
            # duplicate the fields
            fields += f" {fields}"
        for i, e in enumerate(line):
            i += field_offset
            data[i].append(e)
            datastr += e + ","
            if i > imax:
                imax = i
        datastr += "\n"
    print(f"{data=}")
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
    xmin,
    xmax,
    ymin,
    ymax,
    datastr,
    labels,
    colorbar,):
    set_limits(xmin, xmax, ymin, ymax)
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
    subplots:str="1 1",
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
        print(f"{fields=}")
    else:
        data, datastr, fields = read_data(delimiter, fields)
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
    subplots = [int(e) for e in subplots.strip().split()]  # type:ignore
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
        plt.subplot(subplots[0], subplots[1], min(plotid+1, subplots[0]*subplots[1]))  # type:ignore
        plt.plot(x, y, fmtstr, label=label, alpha=alpha)
        plotid += 1
    out(save=save, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, datastr=datastr, labels=labels, colorbar=False)

@app.command()
def scatter(
    fields="x y",
    labels="",
    delimiter=None,
    alpha:float=1.0,
    cmap:str="viridis",
    pcr:bool=False,
    subplots:str="1 1",
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
        print(f"{fields=}")
    else:
        data, datastr, fields = read_data(delimiter, fields)
    fields = fields.strip().split()
    labels = labels.strip().split()
    xfields = np.where(np.asarray(fields)=="x")[0]
    yfields = np.where(np.asarray(fields)=="y")[0]
    s_indices = np.where(np.asarray(fields)=="s")[0]
    c_indices = np.where(np.asarray(fields)=="c")[0]
    subplots = [int(e) for e in subplots.strip().split()]  # type:ignore
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
        plt.subplot(subplots[0], subplots[1], min(plotid+1, subplots[0]*subplots[1]))  # type:ignore
        plt.scatter(x, y, s=s, c=c, label=label, alpha=alpha, cmap=cmap)
        if pcr:
            do_pcr(x,y)
        plotid += 1
    out(save=save, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, datastr=datastr, labels=labels, colorbar=colorbar)

@app.command()
def hist(
    fields="y y",
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
):
    """
    Compute and plot a histogram, see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    """
    fields = fields.strip().split()
    labels = labels.strip().split()
    data, datastr, fields = read_data(delimiter, fields)
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
    out(save=save, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, datastr=datastr, labels=labels, colorbar=False)

@app.command()
def jitter(
    fields="x y",
    labels="",
    delimiter=None,
    xjitter:float=0.1,
    yjitter:float=0.0,
    alpha:float=1.0,
    kde:bool=False,
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
        data, datastr, fields = read_data(delimiter, fields)
    fields = fields.strip().split()
    labels = labels.strip().split()
    xfields = np.where(np.asarray(fields)=="x")[0]
    yfields = np.where(np.asarray(fields)=="y")[0]
    if kde:
        kde_ins = KernelDensity(kernel="gaussian", bandwidth="scott")  # type: ignore
    kde_y = None
    for xfield, yfield in track(zip(xfields, yfields), total=len(xfields), description="Jittering..."):
        x = np.float_(data[xfield])  # type: ignore
        y = np.float_(data[yfield])  # type: ignore
        if kde:
            kde_ins = kde_ins.fit(y[:, None])  # type: ignore
            kde_y = np.exp(kde_ins.score_samples(y[:, None]))  # type: ignore
        x += np.random.normal(size=x.shape, loc=0, scale=xjitter)
        y += np.random.normal(size=y.shape, loc=0, scale=yjitter)
        plt.scatter(x, y, c=kde_y, alpha=alpha, cmap=cmap)
    out(save=save, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, datastr=datastr, labels=labels, colorbar=False)

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
