#!/usr/bin/env python3
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2016 09 26
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys
from inspect import currentframe, getframeinfo
from collections.abc import Iterable
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL import PngImagePlugin

# Allow to print unicode text (see: http://stackoverflow.com/a/21190382/1679629)
# reload(sys)
# sys.setdefaultencoding('utf8')
##############################
import sliding
import ROC
from violin import Violin
from tsne import tsne_embed
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.cm as cm

try:
    import seaborn as sns
#     sns.set_context('paper')
except ImportError:
    print("seaborn is not installed")
try:
    import pandas as pd
except ImportError:
    print("pandas is not installed")
import numpy
from optparse import OptionParser
from optparse import OptionGroup

try:
    from sklearn import mixture

    is_sklearn = True
except ImportError:
    print("sklearn is not installed you cannot use the Gaussian Mixture Model option")
    is_sklearn = False
import numexpr as ne
import os
import socket
from scipy.cluster import hierarchy

# To read large metadata from a png image file
# See: https://stackoverflow.com/a/61466412/1679629
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

print(f"# >>> parsing options {getframeinfo(currentframe()).lineno}")
parser = OptionParser()
parser.add_option("--save", help="Save the file", type=str, dest="outfilename")
parser.add_option(
    "--subsample",
    help="Subsample randomly the point. The given number between 0 and 1 give the ratio of points to keep",
    type=float,
    default=None,
)
parser.add_option(
    "--read_data",
    help="Read plot data from the given png saved image using the --save option",
)
parser.add_option(
    "--aspect_ratio", help="Change the aspect ratio of the figure", nargs=2, type=int
)
parser.add_option("--title", help="Title of the plot", type=str)
parser.add_option("--grid", help="Display a grid on the plot", action="store_true")
parser.add_option(
    "--fix_overlap",
    help="Change the linewidth of curves to set a larger linewidth in curves below an other one. Useful to be able to see overlapping curves",
    action="store_true",
)
parser.add_option(
    "--heatmap",
    help="Plot an heatmap from the given matrix. To give the first column as yticklabels use the option --fields 'l*'",
    action="store_true",
)
parser.add_option(
    "--header",
    help="Read the first line as xticklabels with option --heatmap",
    action="store_true",
)
parser.add_option(
    "--linkage",
    help="Sort the heatmap (see --heatmap) with hierarchical clustering using the given method (one of: single, complete, average, weighted, centroid, median, ward, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)",
    type=str,
)
parser.add_option(
    "-d", "--delimiter", help="Delimiter to use to read the data", default=None
)
parser.add_option(
    "--xlabel", dest="xlabel", default=None, type="str", help="x axis label"
)
parser.add_option(
    "--ylabel", dest="ylabel", default=None, type="str", help="y axis label"
)
parser.add_option(
    "--ylabel2",
    dest="ylabel2",
    default=None,
    type="str",
    help="y axis label for second y-axis (see --dax)",
)
parser.add_option(
    "--xmin", dest="xmin", default=None, type="float", help="Minimum x-value"
)
parser.add_option(
    "--xmax", dest="xmax", default=None, type="float", help="Maximum x-value"
)
parser.add_option(
    "--ymin", dest="ymin", default=None, type="float", help="Lower limit for y-axis"
)
parser.add_option(
    "--ymax", dest="ymax", default=None, type="float", help="Upper limit for y-axis"
)
parser.add_option(
    "--ymin1",
    dest="ymin1",
    default=None,
    type="float",
    help="Lower limit for first y-axis see --dax",
)
parser.add_option(
    "--ymax1",
    dest="ymax1",
    default=None,
    type="float",
    help="Upper limit for first y-axis see --dax",
)
parser.add_option(
    "--ymin2",
    dest="ymin2",
    default=None,
    type="float",
    help="Lower limit for second y-axis see --dax",
)
parser.add_option(
    "--ymax2",
    dest="ymax2",
    default=None,
    type="float",
    help="Upper limit for second y-axis see --dax",
)
parser.add_option(
    "--yticklabelformat",
    dest="yticklabelformat",
    default=None,
    type="str",
    help="Format of the y-ticks labels. E.g.: '{x:.2f}'",
)
parser.add_option(
    "--polyfit",
    dest="polyfit",
    default=None,
    type="int",
    help="Least squares polynomial fit, with the given degree.",
)
parser.add_option(
    "--tsne",
    action="store_true",
    help="Embed the multidimensional data in 2D using TSNE",
)
parser.add_option(
    "--perplexity",
    help="perplexity parameter for the TSNE (default=30)",
    type=int,
    default=30,
)
parser.add_option(
    "--bw", help="Plot in black and white using grey shade", action="store_true"
)
parser.add_option(
    "--roc",
    help="Plot a ROC curve from the given data. Two columns separated by a comma must be given. The first column gives the negative values, the second the positive values.",
    action="store_true",
)
parser.add_option(
    "--vline",
    help="Draw a vertical line at the given x. --vline option can be given multiple time",
    type=float,
    default=None,
    action="append",
)
parser.add_option(
    "--vlabel",
    help="Optional labels for the vertical lines. --vlabel option can be given multiple time",
    default=None,
    action="append",
)
moving_average_options = OptionGroup(parser, "Sliding functions")
moving_average_options.add_option(
    "--moving_average",
    dest="moving_average",
    default=None,
    type="int",
    help="Plot a moving average on the data with the\
                                  given window size. It also prints the values on stdout",
    metavar=10,
)
moving_average_options.add_option(
    "--no_gray_plot",
    dest="gray_plot",
    default=True,
    action="store_false",
    help="Do not plot original data in gray with moving_average option",
)
moving_average_options.add_option(
    "--slide",
    default="mean",
    help="Function to slide along the curve. Can be mean (same as moving_average), max, min, std",
)
moving_average_options.add_option(
    "--ws",
    default=None,
    type="int",
    help="Window size for the sliding function",
    metavar=10,
)
parser.add_option_group(moving_average_options)
parser.add_option(
    "--bar",
    dest="bar",
    default=False,
    action="store_true",
    help="Simple bar plot for single unidimensional data",
)
parser.add_option(
    "--normalize",
    dest="normalize",
    default=None,
    type="str",
    help="Normalize the values to 1. If normalize=x, normalize the \
x values, if normalize=y then normalize the y values. Normalization scales all \
numeric variables in the range [0,1] given the formula: (x-xmin)/(xmax-xmin)",
    metavar="x",
)
parser.add_option(
    "--semilog",
    dest="semilog",
    default=None,
    type="str",
    metavar="x",
    help="Log scale for the given axis (x or y)",
)
parser.add_option(
    "--transpose",
    dest="transpose",
    default=False,
    action="store_true",
    help="Transpose the input data",
)
parser.add_option(
    "--dax",
    dest="dax",
    default=None,
    type=str,
    help="Double axis plot. Can plot multiple dataset.\
    Give the 1 or 2 label to select the axis to plot on, e.g. 12 to plot the first dataset on axis 1 and the second on axis 2",
    metavar="12",
)
parser.add_option(
    "--daxma1",
    default=None,
    help="Optional moving average window for first y-axis",
    type=int,
    metavar=10,
)
parser.add_option(
    "--daxma2",
    default=None,
    help="Optional moving average window for second y-axis",
    type=int,
    metavar=10,
)

scatter_options = OptionGroup(parser, "Scatter plot")
scatter_options.add_option(
    "--scatter",
    action="store_true",
    dest="scatter",
    default=False,
    help="Scatter plot of the (x,y) data",
)
scatter_options.add_option(
    "--plot3d", help="Scatter plot in 3D", action="store_true", default=False
)
scatter_options.add_option(
    "--alpha", type=float, default=1.0, help="Transparency of the scatter dots"
)
scatter_options.add_option(
    "--fields",
    dest="fields",
    default=None,
    metavar="xy*zelwt",
    type="str",
    help="Fields for the data; e.g. 'xyxy'. By default\
                           the first column is for x data and the other for y data. \
If a 'z' field is given, this field is used to color \
the scatter dots. \
If a 'e' field is given it is used to plot the error. \
If a 'l' field is given this are the labels for the xticks -- xticklabels --.\
If a 'w' field is given it is used as weights for weighted hostogram plotting (see: -H).\
If a 't' field is given plot the given text at the given position (with x and y fields) using matplotlib.pyplot.text.\
If a 'm' field is given, use it as markers (see: https://matplotlib.org/stable/api/markers_api.html).\
If a 's' field is given, use it as a list of sizes for the markers.\
If a 'c' field is given, use it as a list of colors (text color: r, g, b, y, ...) for the markers.\
If --fields='*' is given all the columns are considered as y values.",
)
scatter_options.add_option(
    "--mintextdist",
    help="minimal distance between plotted text label in scatter plot (see t-field in fields option).",
    type=float,
    default=None,
)
scatter_options.add_option(
    "-s",
    "--size",
    default=2.0,
    type=float,
    help="size of the dots for the scatter plot (default: 2)",
)
scatter_options.add_option(
    "--sizez", action="store_true", help="Use a variable size given by the z-field"
)
parser.add_option(
    "--facetGrid",
    nargs=2,
    type=int,
    help="Seaborn scatter FacetGrid (see: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html). 'xyz' fields must be given (--fields). Give th number of rows and columns wanted (e.g. 10 10)",
)
parser.add_option(
    "--labels",
    dest="labels",
    default=None,
    type="str",
    help="Comma separated list of labels for each field \
                  defined with the --fields option.",
)
parser.add_option(
    "--loc",
    default="best",
    type="str",
    help="Location of the legend. Can be: 'upper left', 'upper right', 'lower left', 'lower right', 'upper center', 'lower center', 'center left', 'center right', 'center'. Default: 'best'",
)
parser.add_option(
    "--fontsize",
    default="medium",
    type="str",
    help="The font size for the legend and the text plot. If the value is numeric the size will be the absolute font size in points. String values are relative to the current default font size. int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. Default: 'medium'",
)
scatter_options.add_option(
    "--line",
    dest="line",
    default=False,
    action="store_true",
    help="Plot line between points",
)
scatter_options.add_option(
    "--histy",
    action="store_true",
    dest="histy",
    default=False,
    help="Plot 1D histogram for the y axis",
)
scatter_options.add_option(
    "--cmap",
    help="colormap to use. See: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html",
    default=None,
)
parser.add_option_group(scatter_options)

parser.add_option("--violin", help="Violin plot", action="store_true")
parser.add_option(
    "--violincolors",
    help="Comma separated list of colors for each violin of the violins plot",
    default=None,
    type=str,
)

histogram_options = OptionGroup(parser, "Plotting histogram")
histogram_options.add_option(
    "-H",
    "--histogram",
    action="store_true",
    dest="histogram",
    default=False,
    help="Compute and plot histogram from data",
)
histogram_options.add_option(
    "-b",
    "--bins",
    dest="n_bins",
    default=-1,
    type="int",
    help="Number of bins in the histogram. If -1 (default) the optimal number of bins is determined using the Freedman-Diaconis rule.",
    metavar=10,
)
histogram_options.add_option(
    "--histtype",
    dest="histtype",
    default="bar",
    type="str",
    help="Histogram type: bar, barstacked, step, stepfilled",
    metavar="bar",
)
histogram_options.add_option(
    "--normed",
    dest="normed",
    default=False,
    action="store_true",
    help="If True, the first element of the return tuple will be the counts normalized to form a probability density",
)
histogram_options.add_option(
    "--cumulative",
    action="store_true",
    dest="cumulative",
    default=False,
    help="Cumulative histogram",
)
histogram_options.add_option(
    "--cb",
    dest="centerbins",
    action="store_true",
    help="Center the bins of the histogram",
)
if is_sklearn:
    histogram_options.add_option(
        "--gmm",
        dest="gmm",
        default=None,
        type="int",
        help="Gaussian Mixture Model with n components. Trigger the normed option.",
        metavar=2,
    )
histogram_options.add_option(
    "--kde",
    dest="kde",
    default=False,
    action="store_true",
    help="Plot a gaussian kernel density estimate along with the histogram",
)
parser.add_option_group(histogram_options)

histogram2d_options = OptionGroup(parser, "Plotting 2D-histogram")
histogram2d_options.add_option(
    "--histogram2d",
    action="store_true",
    dest="histogram2d",
    default=False,
    help="Compute and plot 2D-histogram from data. -b (--bins) can be used to define the number of bins.",
)
histogram2d_options.add_option(
    "--logscale", action="store_true", dest="logscale", default=False, help="log scale"
)
histogram2d_options.add_option(
    "--projection1d",
    action="store_true",
    dest="projection1d",
    default=False,
    help="Plot 1D histogram for the x and y axis",
)
parser.add_option_group(histogram2d_options)
parser.add_option(
    "--minval",
    action="store_true",
    help="Plot an horizontal line at the minimum y-value",
)
parser.add_option(
    "--maxval",
    action="store_true",
    help="Plot an horizontal line at the maximum y-value",
)
parser.add_option(
    "--minval2",
    action="store_true",
    help="Plot an horizontal line at the minimum y-value for axis 2 (see: --dax)",
)
parser.add_option(
    "--maxval2",
    action="store_true",
    help="Plot an horizontal line at the maximum y-value for axis 2 (see: --dax)",
)
parser.add_option(
    "--mamin",
    action="store_true",
    help="Plot an horizontal line at the minimum y-value on the moving average (see: moving_average option)",
)
parser.add_option(
    "--mamax",
    action="store_true",
    help="Plot an horizontal line at the maximum y-value on the moving average (see: moving_average option)",
)
parser.add_option(
    "--corrcoef",
    help="Return Pearson product-moment correlation coefficients",
    action="store_true",
)

function_options = OptionGroup(parser, "Plotting functions")
function_options.add_option(
    "-f",
    "--func",
    type=str,
    default=None,
    action="append",
    help="Evaluate and plot the function given as a string. If you want to just plot the function without any piped data just run: 'cat /dev/null | plot -f 'x**2' --xmin 0 --xmax 10'. Multiple functions can be plotted at the same time by giving multiple expression with multiple -f option passed. Numpy functions can be used and given without np prefix (e.g. exp)",
)
function_options.add_option(
    "--func_label",
    type=str,
    default=None,
    action="append",
    help="optional label for the function",
)
parser.add_option_group(function_options)

(options, args) = parser.parse_args()

if options.ws is not None:
    options.moving_average = options.ws
if options.slide == "mean":
    options.slide = numpy.mean
if options.slide == "max":
    options.slide = numpy.max
if options.slide == "min":
    options.slide = numpy.min
if options.slide == "std":
    options.slide = numpy.std

if is_sklearn:
    # from: http://stackoverflow.com/a/19182915/1679629
    def fit_mixture(data, ncomp=2):
        clf = mixture.GMM(n_components=ncomp, covariance_type="full")
        clf.fit(data)
        ml = clf.means_
        wl = clf.weights_
        cl = clf.covars_
        ms = [m[0] for m in ml]
        cs = [numpy.sqrt(c[0][0]) for c in cl]
        ws = [w for w in wl]
        return ms, cs, ws


def bins_labels(bins, **kwargs):
    """
    Center bins for histogram (see: https://stackoverflow.com/a/42264525/1679629)
    """
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(
        numpy.arange(min(bins) + bin_w / 2, max(bins) + 1, bin_w), bins, **kwargs
    )
    plt.xlim(bins[0], bins[-1])
    plt.minorticks_off()


def movingaverage(data, window_size):
    """
    see: http://goo.gl/OMbvco
    """
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(data, window, "same")


def sliding_func(data, window_size, func):
    slop = sliding.Sliding_op(data, window_size, func, padding=True)
    return slop.transform()


def sort_scatter_data(data, nbins=None):
    """
    Sort scatter data such as less frequent data points are plot on the top of the scatter plot to remain visible
    • data: data to plot [[x,y,z], ...] where z is the color of the data points
    • nbins: The number of bins to use for the 3D histogram used to order the data
    """
    if nbins is None:
        nbins = int(len(data) * 0.002)
    if nbins < 10:
        nbins = 10
    print("Number of bins used to order the data: %d" % nbins)
    hdd, bins = numpy.histogramdd(data, bins=nbins)
    digits = numpy.asarray(
        [numpy.digitize(v, bins[i], right=True) for i, v in enumerate(data.T)]
    ).T
    digits[digits == nbins] -= 1
    counts = numpy.asarray([hdd[tuple(e)] for e in digits])
    return data[counts.argsort()][::-1]


def freedman_diaconis_rule(data):
    """
    Compute the optimal number of bins accordingly to the Freedman–Diaconis rule.
    See: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """
    q75, q25 = numpy.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_size = 2 * iqr / (len(data)) ** (1.0 / 3)
    n_bins = numpy.ceil(numpy.ptp(data) / bin_size)
    if numpy.isnan(n_bins) or n_bins == numpy.inf:
        n_bins = 2
    else:
        n_bins = int(n_bins)
    print("Freedman–Diaconis optimal number of bins: %d" % n_bins)
    return n_bins


def set_y_lim(ymin, ymax):
    axes = plt.gca()
    limits = plt.axis()
    if ymin is None:
        ymin = limits[-2]
    if ymax is None:
        ymax = limits[-1]
    axes.set_ylim([ymin, ymax])


def set_x_lim(xmin, xmax):
    axes = plt.gca()
    limits = plt.axis()
    if xmin is None:
        xmin = limits[0]
    if xmax is None:
        xmax = limits[1]
    axes.set_xlim([xmin, xmax])


def plot_functions(expression_strings, xlims, npts=100, func_label=None):
    """
    Plot a list of functions given as a list of expression strings
    """
    if func_label is None:
        func_label = [None] * len(expression_strings)
    for expression_string, label in zip(expression_strings, func_label):
        plot_function(expression_string, xlims, npts=npts, label=label)


def plot_function(expression_string, xlims, npts=100, color=None, label=None):
    """
    Plot a function given as an expression string
    """
    if label is None:
        label = expression_string
    if "x" not in expression_string:
        expression_string = expression_string + "+0*x"
    x = numpy.linspace(xlims[0], xlims[1], num=npts)
    y = ne.evaluate(expression_string)
    print(f">>> plot {getframeinfo(currentframe()).lineno}")
    plt.plot(x, y, label=label, color=color)
    plt.legend(loc=options.loc, fontsize=options.fontsize)
    return x, y


def polyfit(x, y, degree):
    p = numpy.polyfit(x, y, degree)
    n = degree
    poly = "+".join([f"{p[i]}*x**{n-i}" for i in range(n)])
    poly += f"+{p[n]}"
    label = "+".join([f"{p[i]:.2g}*x**{n-i}" for i in range(n)])
    label += f"+{p[n]:.2g}"
    return poly, label


def plot_extrema(y, minval=True, maxval=False, ax=None, color="blue"):
    if y.ndim == 1:
        y = y[:, None]
    if (~numpy.isnan(y)).any():
        if minval:
            minima = numpy.nanmin(y, axis=0)
            for i, v in enumerate(minima):
                if ax is None:
                    plt.axhline(y=v, color=color, linestyle="--", linewidth=1.0)
                else:
                    ax.axhline(y=v, color=color, linestyle="--", linewidth=1.0)
        if maxval:
            maxima = numpy.nanmax(y, axis=0)
            for i, v in enumerate(maxima):
                if ax is None:
                    plt.axhline(y=v, color=color, linestyle="--", linewidth=1.0)
                else:
                    ax.axhline(y=v, color=color, linestyle="--", linewidth=1.0)


def add_extrema_to_label(x, y, label, minval=True, maxval=False):
    if (~numpy.isnan(y)).any():
        if minval:
            imin = numpy.nanargmin(y)
            label = label + f" min={y[imin]:.3g} (x={x[imin]})"
        if maxval:
            imax = numpy.nanargmax(y)
            label = label + f" max={y[imax]:.3g} (x={x[imax]})"
    return label


def get_current_limits():
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    return x_min, x_max, y_min, y_max


def plot_text(x, y, text):
    current_limits = get_current_limits()
    set_x_lim(options.xmin, options.xmax)
    set_y_lim(options.ymin, options.ymax)
    plotted = []
    distmin = numpy.inf
    if options.mintextdist is None:
        options.mintextdist = 0.0
    for i, (x_, y_) in enumerate(zip(x, y)):
        if (
            x_ >= current_limits[0]
            and x_ <= current_limits[1]
            and y_ >= current_limits[2]
            and y_ <= current_limits[3]
        ):
            if i > 0:
                distmin = numpy.sqrt(
                    ((numpy.asarray(plotted) - numpy.asarray([x_, y_])) ** 2).sum(
                        axis=1
                    )
                ).min()
            if distmin >= options.mintextdist:
                plt.text(x_, y_, text[i], fontsize=options.fontsize, in_layout=True)
                plotted.append([x_, y_])


def hierarchy_sort(pmat):
    Z = hierarchy.linkage(pmat, method=options.linkage)
    order = hierarchy.leaves_list(Z)
    return pmat[order][:, order], order


def plot_heatmap(mat, xticklabels=None, yticklabels=None):
    n, p = mat.shape
    if options.linkage is not None:
        if n == p:
            print(
                f">>> linkage hierarchical sort {getframeinfo(currentframe()).lineno}"
            )
            mat, order = hierarchy_sort(mat)
        else:
            print(
                f">>> cannot apply linkage hierarchical sort as the matrix is not a square matrix ({n}, {p}) {getframeinfo(currentframe()).lineno}"
            )
    print(f">>> plotting heatmap {getframeinfo(currentframe()).lineno}")
    plt.matshow(mat, cmap=options.cmap)
    if xticklabels is not None:
        if options.linkage is not None and xticklabels is not None:
            xticklabels = numpy.asarray(xticklabels)[order]
        plt.xticks(
            ticks=range(p), labels=xticklabels, rotation=90, fontsize=options.fontsize
        )
    if yticklabels is not None:
        if options.linkage is not None and yticklabels is not None:
            yticklabels = numpy.asarray(yticklabels)[order]
        plt.yticks(ticks=range(n), labels=yticklabels, fontsize=options.fontsize)
    if options.xlabel is not None:
        plt.xlabel(options.xlabel)
    if options.ylabel is not None:
        plt.ylabel(options.ylabel)
    plt.colorbar()
    display_or_save(data)


def do_plot(
    x,
    y,
    z=None,
    e=None,
    markers=None,
    sizes=None,
    colors=None,
    histogram=options.histogram,
    scatter=options.scatter,
    histogram2d=options.histogram2d,
    logscale=options.logscale,
    projection1d=options.projection1d,
    n_bins=options.n_bins,
    xmin=options.xmin,
    xmax=options.xmax,
    func=options.func,
    dax=options.dax,
    vline=None,
    vlabel=None,
    xticklabels=None,
    yticklabelformat=None,
    weights=None,
    text=None,
    data=None,
):
    if options.subsample is not None:
        n = x.shape[0]
        n_sub = int(options.subsample * n)
        print(f"# Subsampling {n_sub} points over {n} ({options.subsample})")
        subsampling = numpy.random.choice(n, size=n_sub, replace=False)
        subsampling = numpy.sort(subsampling)
        x = x[subsampling]
        y = y[subsampling]
        if z is not None:
            z = z[subsampling]
        data = data[subsampling]
    if options.tsne:
        print(f">>> computing TSNE {getframeinfo(currentframe()).lineno}")
        data = tsne_embed(data, perplexity=options.perplexity)
        x = data[:, 0]
        y = data[:, 1]
    if options.corrcoef:
        corr = numpy.corrcoef(x, y)[0, 1]  # x (0) vs y (1)
        print(f"Pearson correlation coefficient: {corr:.3g}")
        poly, polylabel = polyfit(x, y, 1)
        plot_function(poly, (x.min(), x.max()), color="red", label=f"corr: {corr:.3g}")
    if yticklabelformat is not None:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter(yticklabelformat))
    if options.aspect_ratio is not None:
        plt.figure(figsize=(options.aspect_ratio[0], options.aspect_ratio[1]))
        if yticklabelformat is not None:
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter(yticklabelformat))
    if options.grid:
        plt.grid()
    if options.bw:
        cmap = cm.Greys
    else:
        if options.cmap is None:
            cmap = cm.rainbow
        else:
            cmap = matplotlib.cm.get_cmap(name=options.cmap)
    if xmin is None and func is not None:
        xmin = x.min()
    if xmax is None and func is not None:
        xmax = x.max()
    if vline is not None:
        if len(vline) == 1:
            colors = cmap([0.5])
        else:
            colors = cmap(numpy.linspace(0, 1, len(vline)))
        for i, xvline in enumerate(vline):
            if vlabel is not None:
                vlabel_i = vlabel[i]
            else:
                vlabel_i = None
            plt.axvline(x=xvline, color=colors[i], ls="--", label=vlabel_i)
            plt.legend(loc=options.loc, fontsize=options.fontsize)
    if options.polyfit is not None:
        poly, polylabel = polyfit(x, y, options.polyfit)
        print(f"polyfit: {polylabel}")
        plot_function(poly, (x.min(), x.max()), color="red", label=polylabel)
    if options.semilog is not None:
        if options.semilog == "x":
            plt.xscale("log")
            if yticklabelformat is not None:
                plt.gca().yaxis.set_major_formatter(
                    StrMethodFormatter(yticklabelformat)
                )
        elif options.semilog == "y":
            plt.yscale("log")
            if yticklabelformat is not None:
                plt.gca().yaxis.set_major_formatter(
                    StrMethodFormatter(yticklabelformat)
                )
    if options.labels is not None:
        labels = options.labels.split(",")
    else:
        labels = None
    if (
        not histogram
        and not scatter
        and not histogram2d
        and not dax
        and not options.violin
    ):
        if (options.minval or options.maxval) and options.moving_average is None:
            plot_extrema(y, minval=options.minval, maxval=options.maxval)
        if options.moving_average is None:
            if len(x.shape) == 1 and len(y.shape) == 1:
                if not options.bar:
                    if labels is not None:
                        label = labels[0]
                        label = add_extrema_to_label(
                            x, y, label, minval=options.minval, maxval=options.maxval
                        )
                    else:
                        label = None
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    plt.plot(x, y, label=label)
                else:
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    plt.bar(x, y, align="center")
                    if len(text) > 0:
                        print(
                            f">>> plotting text {getframeinfo(currentframe()).lineno}"
                        )
                        plot_text(x, y, text)
            else:
                if len(x.shape) == 1:
                    x = x[:, None]
                if x.shape[1] > 1:
                    colors = cmap(numpy.linspace(0, 1, x.shape[1]))
                    for i, xi in enumerate(x.T):
                        if labels is not None:
                            label = labels[i]
                        else:
                            label = None
                        yi = y.T[i]
                        print(f">>> plot {getframeinfo(currentframe()).lineno}")
                        label = add_extrema_to_label(
                            xi.flatten(),
                            yi.flatten(),
                            label,
                            minval=options.minval,
                            maxval=options.maxval,
                        )
                        plt.plot(xi.flatten(), yi.flatten(), c=colors[i], label=label)
                elif y.shape[1] > 1:
                    colors = cmap(numpy.linspace(0, 1, y.shape[1]))
                    if options.fix_overlap:
                        # Options to change the linewidth to be able to see overlapping curves if any
                        # The curve below has a larger linewidth
                        linewidth = y.shape[1] + 1
                    else:
                        linewidth = None
                    for i, yi in enumerate(y.T):
                        if labels is not None:
                            label = labels[i]
                            label = add_extrema_to_label(
                                x,
                                yi,
                                label,
                                minval=options.minval,
                                maxval=options.maxval,
                            )
                        else:
                            label = None
                        print(f">>> plot {getframeinfo(currentframe()).lineno}")
                        if options.fix_overlap:
                            linewidth -= 1
                        plt.plot(
                            x.flatten(),
                            yi.flatten(),
                            c=colors[i],
                            label=label,
                            alpha=options.alpha,
                            linewidth=linewidth,
                        )
            if e is not None:
                if len(y.shape) == 1:  # No more than 1 curve
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    plt.fill_between(
                        x.flatten(),
                        y.flatten() - e.flatten(),
                        y.flatten() + e.flatten(),
                        facecolor="gray",
                        alpha=0.5,
                    )
                else:
                    for i, errror in enumerate(e.T):
                        print(f">>> plot {getframeinfo(currentframe()).lineno}")
                        plt.fill_between(
                            x[:, i],
                            y[:, i] - e[:, i],
                            y[:, i] + e[:, i],
                            facecolor="gray",
                            alpha=0.5,
                        )
        else:  # Moving average
            if y.ndim == 1:
                y = y[..., None]
            if options.gray_plot:
                for y_ in y.T:
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    plt.plot(
                        x.flatten(), y_.flatten(), "-", color="gray", alpha=0.25, lw=1.0
                    )
            ws = options.moving_average  # window size
            if options.fix_overlap:
                linewidth = y.shape[1] + 1
            else:
                linewidth = 2.0
            for i, y_ in enumerate(y.T):
                ma_array = numpy.c_[
                    x.flatten()[int(ws / 2) : int(-ws / 2)],
                    sliding_func(y_, ws, options.slide)[int(ws / 2) : int(-ws / 2)],
                ]
                if labels is not None:
                    label = labels[i]
                    label = add_extrema_to_label(
                        ma_array[:, 0],
                        ma_array[:, 1],
                        label,
                        minval=options.minval,
                        maxval=options.maxval,
                    )
                else:
                    label = None
                print(
                    f">>> plotting moving average {getframeinfo(currentframe()).lineno}"
                )
                if options.fix_overlap:
                    linewidth -= 1
                if ws > 1:
                    plt.plot(
                        ma_array[:, 0], ma_array[:, 1], linewidth=linewidth, label=label
                    )
                else:
                    plt.plot(x.flatten(), y_, linewidth=linewidth, label=label)
                if options.minval or options.maxval:
                    if (
                        options.mamin or options.mamax
                    ):  # plot min value of the moving average
                        plot_extrema(
                            ma_array[:, 1], minval=options.minval, maxval=options.maxval
                        )
                    else:
                        plot_extrema(y_, minval=options.minval, maxval=options.maxval)
    elif options.violin:
        if options.violincolors is not None:
            violincolors = options.violincolors.split(",")
        else:
            violincolors = None
        violin = Violin(y, labels, colors=violincolors)
        print(f">>> plotting violins {getframeinfo(currentframe()).lineno}")
        violin.plot()
        options.labels = None
    elif scatter:
        if len(x.shape) == 1:
            x = x[:, None]
        if len(y.shape) == 1:
            y = y[:, None]
        if x.shape[1] == 1 and y.shape[1] == 1:
            if z is not None:
                if options.sizez:
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    plt.scatter(
                        x,
                        y,
                        s=z * options.size,
                        alpha=options.alpha,
                        facecolor="none",
                        edgecolor="blue",
                    )
                else:
                    if not options.plot3d:
                        if options.facetGrid is not None:
                            print(
                                f">>> plotting FacetGrid {getframeinfo(currentframe()).lineno}"
                            )
                            mesh = numpy.meshgrid(
                                range(options.facetGrid[0]), range(options.facetGrid[1])
                            )
                            keys = numpy.unique(z)
                            rowdict = dict(zip(keys, mesh[0].flatten()))
                            coldict = dict(zip(keys, mesh[1].flatten()))
                            keys = list(rowdict.keys())
                            sel = numpy.isin(z[:, 0], keys)
                            if len(sizes) == 0:
                                s = numpy.ones_like(x[:, 0][sel]) * options.size
                            else:
                                s = numpy.asarray(sizes)[sel] * options.size
                            hue_kws = {"s": s}
                            if len(colors) == 0:
                                hueval = z[:, 0][sel]
                            else:
                                colors = numpy.asarray(colors)
                                hueval = [
                                    matplotlib.colors.to_rgb(e) for e in colors[sel]
                                ]
                            g = sns.FacetGrid(
                                pd.DataFrame(
                                    {
                                        "x": x[:, 0][sel],
                                        "y": y[:, 0][sel],
                                        "z": z[:, 0][sel],
                                        "row": [rowdict[e] for e in z[:, 0][sel]],
                                        "col": [coldict[e] for e in z[:, 0][sel]],
                                        "hue": hueval,
                                    }
                                ),
                                row="row",
                                col="col",
                                hue="hue",
                                hue_kws=hue_kws,
                                palette=options.cmap,
                            )
                            g = g.map(
                                plt.scatter,
                                "x",
                                "y",
                                alpha=options.alpha,
                            )
                        else:
                            if markers == []:
                                print(
                                    f">>> plotting scatter with z-colormap {getframeinfo(currentframe()).lineno}"
                                )
                                if sizes == []:
                                    s = options.size
                                else:
                                    s = numpy.asarray(sizes) * options.size
                                plt.scatter(
                                    x,
                                    y,
                                    c=z,
                                    s=s,
                                    alpha=options.alpha,
                                    cmap=options.cmap,
                                )
                            else:
                                print(
                                    f">>> plotting scatter with z-colormap and markers {getframeinfo(currentframe()).lineno}"
                                )
                                markers_unique = numpy.unique(markers)
                                markers = numpy.asarray(markers)
                                for marker in markers_unique:
                                    sel = markers == marker
                                    plt.scatter(
                                        x[sel],
                                        y[sel],
                                        c=z[sel],
                                        s=options.size,
                                        alpha=options.alpha,
                                        cmap=options.cmap,
                                        marker=marker,
                                    )
                            plt.colorbar()
                        if len(text) > 0:
                            print(
                                f">>> plotting text {getframeinfo(currentframe()).lineno}"
                            )
                            plot_text(x, y, text)
                    else:
                        ax = plt.axes(projection="3d")
                        ax.scatter3D(x, y, z, s=options.size)
            else:
                if options.line:
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    plt.plot(x, y, ",-")
                else:
                    if options.histy:
                        left, width = 0.1, 0.65
                        bottom, height = 0.1, 0.8
                        bottom_h = left_h = left + width + 0.02
                        rect_scatter = [left, bottom, width, height]
                        rect_histx = [left, bottom_h, width, 0.2]
                        rect_histy = [left_h, bottom, 0.2, height]
                        axScatter = plt.axes(rect_scatter)
                        if options.xlabel is not None:
                            axScatter.set_xlabel(options.xlabel)
                        if options.ylabel is not None:
                            axScatter.set_ylabel(options.ylabel)
                        axHisty = plt.axes(rect_histy, sharey=axScatter)
                        axHisty.tick_params(labelleft=False)
                        # nullfmt = matplotlib.ticker.NullFormatter()
                        # axHisty.yaxis.set_major_formatter(nullfmt)
                        if n_bins == -1:
                            n_bins = freedman_diaconis_rule(y)
                        axScatter.scatter(x, y, s=options.size, alpha=options.alpha)
                        axHisty.hist(y, bins=n_bins, orientation="horizontal")
                    else:
                        print(
                            f">>> plotting scatter {getframeinfo(currentframe()).lineno}"
                        )
                        plt.scatter(x, y, s=options.size, alpha=options.alpha, c=colors)
                        if len(text) > 0:
                            print(
                                f">>> plotting text {getframeinfo(currentframe()).lineno}"
                            )
                            plot_text(x, y, text)
        else:
            if x.shape[1] > 1:
                colors = cmap(numpy.linspace(0, 1, x.shape[1]))
                for i, xi in enumerate(x.T):
                    yi = y.T[i]
                    if z is not None:
                        zi = z.T[i]
                    if labels is not None:
                        if options.line:
                            print(f">>> plot {getframeinfo(currentframe()).lineno}")
                            plt.plot(xi, yi, ",-", c=colors[i], label=labels[i])
                        else:
                            if options.sizez:
                                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                                plt.scatter(
                                    xi,
                                    yi,
                                    label=labels[i],
                                    s=zi * options.size,
                                    alpha=options.alpha,
                                    facecolor="none",
                                    edgecolor=colors[i],
                                )
                            else:
                                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                                plt.scatter(
                                    xi,
                                    yi,
                                    c=colors[i],
                                    label=labels[i],
                                    alpha=options.alpha,
                                    s=options.size,
                                )
                    else:
                        if options.line:
                            print(f">>> plot {getframeinfo(currentframe()).lineno}")
                            plt.plot(xi, yi, ",-", c=colors[i])
                        else:
                            if options.sizez:
                                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                                plt.scatter(
                                    xi,
                                    yi,
                                    s=zi * options.size,
                                    alpha=options.alpha,
                                    facecolor="none",
                                    edgecolor=colors[i],
                                )
                            else:
                                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                                plt.scatter(
                                    xi,
                                    yi,
                                    c=colors[i],
                                    alpha=options.alpha,
                                    s=options.size,
                                )
            elif y.shape[1] > 1:
                colors = cmap(numpy.linspace(0, 1, y.shape[1]))
                for i, yi in enumerate(y.T):
                    if options.line:
                        if labels is not None:
                            print(f">>> plot {getframeinfo(currentframe()).lineno}")
                            plt.plot(x, yi, ",-", c=colors[i], label=labels[i])
                        else:
                            print(f">>> plot {getframeinfo(currentframe()).lineno}")
                            plt.plot(x, yi, ",-", c=colors[i])
                    else:
                        if labels is not None:
                            print(f">>> plot {getframeinfo(currentframe()).lineno}")
                            plt.scatter(
                                x, yi, c=colors[i], label=labels[i], alpha=options.alpha
                            )
                        else:
                            print(f">>> plot {getframeinfo(currentframe()).lineno}")
                            plt.scatter(x, yi, c=colors[i], alpha=options.alpha)
        if e is not None:
            if y.shape[1] == 1:
                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                plt.errorbar(x.flatten(), y.flatten(), yerr=e.flatten(), markersize=0.0)
            else:
                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                for i, yi in enumerate(y.T):
                    plt.errorbar(
                        x[:, i].flatten(), yi, yerr=e[:, i], markersize=0.0, c=colors[i]
                    )
    elif histogram2d:
        x, y = x.flatten(), y.flatten()
        if projection1d:  # 1D projections of histogram
            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            bottom_h = left_h = left + width + 0.02

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom_h, width, 0.2]
            rect_histy = [left_h, bottom, 0.2, height]
            # start with a rectangular Figure
            plt.figure(1, figsize=(8, 8))
            axScatter = plt.axes(rect_scatter)
            axHistx = plt.axes(rect_histx)
            axHisty = plt.axes(rect_histy)
            # no labels
            nullfmt = matplotlib.ticker.NullFormatter()
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHistx.grid()
            axHisty.yaxis.set_major_formatter(nullfmt)
            axHisty.grid()
            if logscale:
                axScatter.hist2d(x, y, bins=n_bins, norm=matplotlib.colors.LogNorm())
                axScatter.grid()
            else:
                axScatter.hist2d(x, y, bins=n_bins)
                axScatter.grid()
            if options.xlabel is not None:
                axScatter.set_xlabel(options.xlabel)
            if options.ylabel is not None:
                axScatter.set_ylabel(options.ylabel)
            axHistx.hist(x, bins=n_bins)
            axHisty.hist(y, bins=n_bins, orientation="horizontal")
            axHistx.set_xlim(axScatter.get_xlim())
            axHisty.set_ylim(axScatter.get_ylim())
            # Cool trick that changes the number of tickmarks for the histogram axes
            axHisty.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axHistx.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        else:
            if logscale:
                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                plt.hist2d(x, y, bins=n_bins, norm=matplotlib.colors.LogNorm())
            else:
                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                plt.hist2d(x, y, bins=n_bins)
            plt.colorbar()
    elif dax is not None:
        dax = numpy.asarray([int(e) for e in dax])
        plt.close()
        if options.aspect_ratio is None:
            fig, ax1 = plt.subplots()
        else:
            fig, ax1 = plt.subplots(
                figsize=(options.aspect_ratio[0], options.aspect_ratio[1])
            )
        if func is not None:
            plot_functions(func, [xmin, xmax], func_label=options.func_label)
        if vline is not None:
            if len(vline) == 1:
                colors = cmap([0.5])
            else:
                colors = cmap(numpy.linspace(0, 1, len(vline)))
            for i, xvline in enumerate(vline):
                if vlabel is not None:
                    vlabel_i = vlabel[i]
                else:
                    vlabel_i = None
                plt.axvline(x=xvline, color=colors[i], ls="--", label=vlabel_i)
                plt.legend(loc=options.loc, fontsize=options.fontsize)
        ax2 = ax1.twinx()
        for datai, daxi in enumerate(dax):
            if daxi == 1:
                if options.daxma1 is None:
                    if labels is not None:
                        label = labels[datai]
                        label = add_extrema_to_label(
                            x,
                            y[:, datai],
                            label,
                            minval=options.minval,
                            maxval=options.maxval,
                        )
                    else:
                        label = None
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    ax1.plot(x, y[:, datai], "g-", alpha=options.alpha, label=label)
                    if label is not None:
                        ax1.legend(loc=options.loc, fontsize=options.fontsize)
                    if options.minval or options.maxval:
                        plot_extrema(
                            y[:, datai],
                            minval=options.minval,
                            maxval=options.maxval,
                            ax=ax1,
                            color="g",
                        )
                else:
                    yma = movingaverage(y[:, datai], options.daxma1)
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    ax1.plot(x, y[:, datai], "gray", alpha=0.25)
                    ax1.plot(x, yma, "g-", alpha=options.alpha)
            else:
                if options.daxma2 is None:
                    if labels is not None:
                        label = labels[datai]
                        label = add_extrema_to_label(
                            x,
                            y[:, datai],
                            label,
                            minval=options.minval2,
                            maxval=options.maxval2,
                        )
                    else:
                        label = None
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    ax2.plot(x, y[:, datai], "b-", alpha=options.alpha, label=label)
                    if label is not None:
                        ax2.legend(loc=options.loc, fontsize=options.fontsize)
                    if options.minval2 or options.maxval2:
                        plot_extrema(
                            y[:, datai],
                            minval=options.minval2,
                            maxval=options.maxval2,
                            ax=ax2,
                            color="b",
                        )
                else:
                    yma = movingaverage(y[:, datai], options.daxma2)
                    print(f">>> plot {getframeinfo(currentframe()).lineno}")
                    ax2.plot(x, y[:, datai], "gray", alpha=0.25)
                    ax2.plot(x, yma, "b-", alpha=options.alpha)
        _, _, ymin1, ymax1 = ax1.axis()
        _, _, ymin2, ymax2 = ax2.axis()
        if options.ymin1 is not None:
            ymin1 = options.ymin1
        if options.ymax1 is not None:
            ymax1 = options.ymax1
        if options.ymin2 is not None:
            ymin2 = options.ymin2
        if options.ymax2 is not None:
            ymax2 = options.ymax2
        ax1.set_ylim([ymin1, ymax1])
        ax2.set_ylim([ymin2, ymax2])
        if options.xlabel is not None:
            ax1.set_xlabel(options.xlabel)
        if options.ylabel is not None:
            ax1.set_ylabel(options.ylabel, color="g")
        if options.ylabel2 is not None:
            ax2.set_ylabel(options.ylabel2, color="b")
    else:
        if xmin is None:
            xmin = min(y.flatten())
        if xmax is None:
            xmax = max(y.flatten())
        if is_sklearn and options.gmm is not None:
            options.normed = True
        if n_bins == -1:
            n_bins = freedman_diaconis_rule(y)
        if not options.kde:
            bins = numpy.linspace(xmin, xmax, n_bins)
            # print(f"bins: {bins}")
            if len(weights) == 0:
                weights = None
            print(f">>> plot {getframeinfo(currentframe()).lineno}")
            histo = plt.hist(
                y,
                bins=bins,
                range=(xmin, xmax),
                histtype=options.histtype,
                density=options.normed,
                label=labels,
                cumulative=options.cumulative,
                alpha=options.alpha,
                edgecolor="black",
                weights=weights,
            )
            if options.centerbins:
                bins_labels(numpy.int_(bins))
        else:
            if data.ndim > 1:
                for ndim_ in range(data.shape[1]):
                    subsampling = ~numpy.isnan(data[:, ndim_])
                    if labels is not None:
                        label = labels[ndim_]
                    else:
                        label = "col %d" % (ndim_ + 1)
                    sns.distplot(data[:, ndim_][subsampling], label=label)
                plt.legend(loc=options.loc, fontsize=options.fontsize)
            else:
                sns.distplot(y)
        if is_sklearn:
            if options.gmm is not None:
                ms, cs, ws = fit_mixture(y, ncomp=options.gmm)
                print("Gaussian Mixture Model with %d components" % options.gmm)
                print("Means: %s" % ms)
                print("Variances: %s" % cs)
                print("Weights: %s" % ws)
                fitting = numpy.zeros_like(histo[1])
                for w, m, c in zip(ws, ms, cs):
                    fitting += w * matplotlib.mlab.normpdf(histo[1], m, c)
                print(f">>> plot {getframeinfo(currentframe()).lineno}")
                plt.plot(histo[1], fitting, linewidth=3)
    if func is not None:
        if dax is None:
            plot_functions(func, [xmin, xmax], func_label=options.func_label)
    if not projection1d and not options.histy:
        if options.xlabel is not None:
            plt.xlabel(options.xlabel)
        if options.ylabel is not None and options.dax is None:
            plt.ylabel(options.ylabel)
    set_x_lim(options.xmin, options.xmax)
    set_y_lim(options.ymin, options.ymax)
    if options.title is not None:
        plt.suptitle(options.title)
    if options.labels is not None:
        plt.legend(loc=options.loc, fontsize=options.fontsize)
    if xticklabels is not None:
        plt.xticks(ticks=x, labels=xticklabels, rotation=90)
    if options.subsample is None:
        subsampling = None
    display_or_save(data, subsampling)


def display_or_save(data, subsampling=None):
    if options.outfilename is None:
        plt.show()
    else:
        metadata = dict()
        datastr = ""
        for i, l in enumerate(data):
            if options.subsample is not None:
                datastr += f"{subsampling[i]} "
            if isinstance(l, Iterable):
                for e in l:
                    datastr += f"{e} "
            else:
                datastr += f"{l}"
            datastr += "\n"
        plt.savefig(options.outfilename)
        add_metadata(options.outfilename, datastr)


def add_metadata(filename, datastr, key="data"):
    """
    Add metadata to a png file
    """
    metadata = PngInfo()
    metadata.add_text(key, datastr, zip=True)
    metadata.add_text("cwd", os.getcwd())
    metadata.add_text("hostname", socket.gethostname())
    if options.subsample is not None:
        print("# Adding subsampling metadata")
        metadata.add_text("subsampling", "1st-column")
    if options.labels is not None:
        metadata.add_text("labels", options.labels)
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


if options.read_data is not None:
    datastr = read_metadata(options.read_data)
    print(datastr)
    sys.exit()

if options.roc:
    options.delimiter = ","
if options.fields is None:
    dtype = float
else:
    if (
        "l" in options.fields
        or "t" in options.fields
        or "m" in options.fields
        or "c" in options.fields
    ):
        dtype = None  # to be able to read also text
    else:
        dtype = float
print(f">>> reading data from stdin {getframeinfo(currentframe()).lineno}")
header = None
if options.header:
    print(f">>> reading header from stdin {getframeinfo(currentframe()).lineno}")
    header = sys.stdin.readline().strip().split()
    if options.fields.startswith("l"):
        header = header[1:]
data = numpy.genfromtxt(
    sys.stdin,
    invalid_raise=False,
    delimiter=options.delimiter,
    dtype=dtype,
    filling_values=numpy.nan,
)
if options.fields is not None:
    if (
        "l" in options.fields
        or "t" in options.fields
        or "m" in options.fields
        or "c" in options.fields
    ):
        data = numpy.asarray(
            data.tolist(), dtype="U22"
        )  # For formatting arrays with both data and text
xticklabels = None
n = data.shape[0]
if options.transpose:
    data = data.T
weights = []  # optional weights for histogram
text = []
if n > 1:
    if len(data.shape) == 1:
        print(f">>> reading 1-dimensional data {getframeinfo(currentframe()).lineno}")
        x = range(n)
        y = data
        z = None
        e = None
        x = numpy.asarray(x)[:, None]
        y = numpy.asarray(y)[:, None]
    else:
        if options.heatmap and options.fields is None:
            print(
                f">>> reading 2-dimensional data as heatmap {getframeinfo(currentframe()).lineno}"
            )
            plot_heatmap(data)
            sys.exit()
        if options.fields is None:
            print(
                f">>> reading 2-dimensional data {getframeinfo(currentframe()).lineno}"
            )
            x = data[:, 0]
            y = data[:, 1:]
            z = None
            e = None
        if options.fields is not None:
            print(f">>> reading data from fields {getframeinfo(currentframe()).lineno}")
            x, y, z, e, xticklabels, weights, text, markers, sizes, colors = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for i, field in enumerate(options.fields):
                if field == "x":
                    x.append(data[:, i])
                elif field == "y":
                    y.append(data[:, i])
                elif field == "z":
                    z.append(data[:, i])
                elif field == "e":
                    e.append(data[:, i])
                elif field == "l":  # label for bar plot
                    # xticklabels.extend(data[:, i])
                    xticklabels.extend(data[:, i])
                elif field == "t":  # plot text with plt.text
                    text.extend(data[:, i])
                elif field == "m":  # plot text with plt.text
                    markers.extend(data[:, i])
                elif field == "w":  # weight for weighted histogram
                    weights.extend(data[:, i])
                elif field == "s":  # weight for weighted histogram
                    sizes.extend(data[:, i])
                elif field == "c":  # weight for weighted histogram
                    colors.extend(data[:, i])
                elif field == "*":
                    y = data[:, i:].T
            if len(xticklabels) == 0:
                xticklabels = None
            x, y, z, e = (
                numpy.asarray(x, dtype=float).T,
                numpy.asarray(y, dtype=float).T,
                numpy.asarray(z, dtype=float).T,
                numpy.asarray(e, dtype=float).T,
            )
            weights = numpy.asarray(weights, dtype=float)
            if len(z) == 0:
                z = None
            # else:
            #     data_sorted = sort_scatter_data(numpy.c_[x,y,z])
            #     x = data_sorted[:,0][:,None]
            #     y = data_sorted[:,1][:,None]
            #     z = data_sorted[:,2]
            if len(e) == 0:
                e = None
            if options.heatmap:
                print(
                    f">>> sending y-data to heatmap {getframeinfo(currentframe()).lineno}"
                )
                plot_heatmap(y, xticklabels=header, yticklabels=xticklabels)
                sys.exit()
        x = numpy.asarray(x)[:, None]
    if options.roc:
        negatives = data[:, 0]
        positives = data[:, 1]
        x, y, auc, pROC_auc = ROC.ROC(positives, negatives)
        print(f"AUC: {auc:.2f}")
        print(f"pROC_AUC: {pROC_auc:.2f}")
    x = numpy.squeeze(x)
    y = numpy.squeeze(y)
    if x.shape == (0,):  # If not x field is given with fields option
        x = numpy.arange(y.shape[0])
    plt.clf()
    if options.normalize == "x":
        xmin, xmax = numpy.min(x, axis=0), numpy.max(x, axis=0)
        x = (x - xmin) / (xmax - xmin)
    if options.normalize == "y":
        ymin, ymax = numpy.min(y, axis=0), numpy.max(y, axis=0)
        y = (y - ymin) / (ymax - ymin)
    do_plot(
        x,
        y,
        z,
        e,
        markers=markers,
        sizes=sizes,
        colors=colors,
        vline=options.vline,
        vlabel=options.vlabel,
        xticklabels=xticklabels,
        yticklabelformat=options.yticklabelformat,
        weights=weights,
        text=text,
        data=data,
    )
else:
    plot_functions(
        options.func, xlims=[options.xmin, options.xmax], func_label=options.func_label
    )
    plt.show()
