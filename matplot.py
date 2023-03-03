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
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.cm as cm
# try:
#     import seaborn as sns
#     sns.set_context('paper')
# except ImportError:
#     print("seaborn not installed")
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

# To read large metadata from a png image file
# See: https://stackoverflow.com/a/61466412/1679629
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

parser = OptionParser()
parser.add_option("--save", help="Save the file", type=str, dest='outfilename')
parser.add_option("--read_data", help="Read plot data from the given png saved image using the --save option")
parser.add_option("--aspect_ratio", help="Change the aspect ratio of the figure", nargs=2, type=int)
parser.add_option("--title", help="Title of the plot", type=str)
parser.add_option("--grid", help="Display a grid on the plot", action='store_true')
parser.add_option("-d", "--delimiter", help="Delimiter to use to read the data", default=None)
parser.add_option("--xlabel", dest="xlabel", default=None, type='str', help="x axis label")
parser.add_option("--ylabel", dest="ylabel", default=None, type='str', help="y axis label")
parser.add_option("--ylabel2",
                  dest="ylabel2",
                  default=None,
                  type='str',
                  help="y axis label for second y-axis (see --dax)")
parser.add_option("--xmin", dest="xmin", default=None, type='float', help="Minimum x-value")
parser.add_option("--xmax", dest="xmax", default=None, type='float', help="Maximum x-value")
parser.add_option("--ymin", dest="ymin", default=None, type='float', help="Lower limit for y-axis")
parser.add_option("--ymax", dest="ymax", default=None, type='float', help="Upper limit for y-axis")
parser.add_option("--ymin1", dest="ymin1", default=None, type='float', help="Lower limit for first y-axis see --dax")
parser.add_option("--ymax1", dest="ymax1", default=None, type='float', help="Upper limit for first y-axis see --dax")
parser.add_option("--ymin2", dest="ymin2", default=None, type='float', help="Lower limit for second y-axis see --dax")
parser.add_option("--ymax2", dest="ymax2", default=None, type='float', help="Upper limit for second y-axis see --dax")
parser.add_option("--yticklabelformat",
                  dest="yticklabelformat",
                  default=None,
                  type='str',
                  help="Format of the y-ticks labels. E.g.: '{x:.2f}'")
parser.add_option("--polyfit",
                  dest="polyfit",
                  default=None,
                  type='int',
                  help="Least squares polynomial fit, with the given degree.")
parser.add_option("--bw", help='Plot in black and white using grey shade', action='store_true')
parser.add_option(
    "--roc",
    help=
    'Plot a ROC curve from the given data. Two columns separated by a comma must be given. The first column gives the negative values, the second the positive values.',
    action="store_true")
parser.add_option('--vline',
                  help='Draw a vertical line at the given x. --vline option can be given multiple time',
                  type=float,
                  default=None,
                  action='append')
parser.add_option('--vlabel',
                  help='Optional labels for the vertical lines. --vlabel option can be given multiple time',
                  default=None,
                  action='append')
moving_average_options = OptionGroup(parser, "Sliding functions")
moving_average_options.add_option("--moving_average",
                                  dest="moving_average",
                                  default=None,
                                  type='int',
                                  help="Plot a moving average on the data with the\
                                  given window size. It also prints the values on stdout",
                                  metavar=10)
moving_average_options.add_option("--no_gray_plot",
                                  dest="gray_plot",
                                  default=True,
                                  action="store_false",
                                  help="Do not plot original data in gray with moving_average option")
moving_average_options.add_option(
    "--slide",
    default='mean',
    help='Function to slide along the curve. Can be mean (same as moving_average), max, min, std')
moving_average_options.add_option("--ws",
                                  default=None,
                                  type='int',
                                  help='Window size for the sliding function',
                                  metavar=10)
parser.add_option_group(moving_average_options)
parser.add_option("--bar",
                  dest="bar",
                  default=False,
                  action="store_true",
                  help="Simple bar plot for single unidimensional data")
parser.add_option("--normalize",
                  dest="normalize",
                  default=None,
                  type='str',
                  help='Normalize the values to 1. If normalize=x, normalize the \
x values, if normalize=y then normalize the y values. Normalization scales all \
numeric variables in the range [0,1] given the formula: (x-xmin)/(xmax-xmin)',
                  metavar='x')
parser.add_option("--semilog",
                  dest="semilog",
                  default=None,
                  type='str',
                  metavar='x',
                  help="Log scale for the given axis (x or y)")
parser.add_option("--transpose", dest="transpose", default=False, action="store_true", help="Transpose the input data")
parser.add_option("--dax",
                  dest="dax",
                  default=None,
                  type=str,
                  help="Double axis plot. Can plot multiple dataset.\
    Give the 1 or 2 label to select the axis to plot on, e.g. 12 to plot the first dataset on axis 1 and the second on axis 2",
                  metavar='12')
parser.add_option("--subplot",
                  dest="subplot",
                  nargs=2,
                  action='append',
                  default=None,
                  help="Arrange multiple plot on a grid of shape n×p, given by \
                  arguments: --subplot n p")

scatter_options = OptionGroup(parser, "Scatter plot")
scatter_options.add_option("--scatter",
                           action="store_true",
                           dest="scatter",
                           default=False,
                           help="Scatter plot of the (x,y) data")
scatter_options.add_option("--plot3d", help="Scatter plot in 3D", action='store_true', default=False)
scatter_options.add_option("--alpha", type=float, default=1., help='Transparency of the scatter dots')
scatter_options.add_option("--fields",
                           dest="fields",
                           default=None,
                           type='str',
                           help="Fields for the data; e.g. 'xyxy'. By default\
                           the first column is for x data and the other for y data. \
If a 'z' field is given, this field is used to color \
the scatter dots. \
If a 'e' field is given it is used to plot the error. \
If a 'l' field is given this are the labels for the xticks -- xticklabels --.\
If --fields='*' is given all the columns are considered as y values.")
scatter_options.add_option("-s",
                           "--size",
                           default=2.,
                           type=float,
                           help='size of the dots for the scatter plot (default: 2)')
scatter_options.add_option("--sizez", action='store_true', help="Use a variable size given by the z-field")
parser.add_option("--labels",
                  dest="labels",
                  default=None,
                  type='str',
                  help="Comma separated list of labels for each field \
                  defined with the --fields option.")
scatter_options.add_option("--line", dest='line', default=False, action="store_true", help="Plot line between points")
scatter_options.add_option("--histy",
                           action="store_true",
                           dest="histy",
                           default=False,
                           help="Plot 1D histogram for the y axis")
scatter_options.add_option("--cmap",
                           help='colormap to use. See: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html',
                           default=None)
parser.add_option_group(scatter_options)

histogram_options = OptionGroup(parser, "Plotting histogram")
histogram_options.add_option("-H",
                             "--histogram",
                             action="store_true",
                             dest="histogram",
                             default=False,
                             help="Compute and plot histogram from data")
histogram_options.add_option(
    "-b",
    "--bins",
    dest="n_bins",
    default=-1,
    type="int",
    help=
    "Number of bins in the histogram. If -1 (default) the optimal number of bins is determined using the Freedman-Diaconis rule.",
    metavar=10)
histogram_options.add_option("--histtype",
                             dest="histtype",
                             default='bar',
                             type='str',
                             help="Histogram type: bar, barstacked, step, stepfilled",
                             metavar='bar')
histogram_options.add_option(
    "--normed",
    dest="normed",
    default=False,
    action="store_true",
    help="If True, the first element of the return tuple will be the counts normalized to form a probability density")
histogram_options.add_option("--cumulative",
                             action="store_true",
                             dest="cumulative",
                             default=False,
                             help="Cumulative histogram")
histogram_options.add_option('--cb', dest='centerbins', action='store_true', help='Center the bins of the histogram')
if is_sklearn:
    histogram_options.add_option("--gmm",
                                 dest="gmm",
                                 default=None,
                                 type='int',
                                 help="Gaussian Mixture Model with n components. Trigger the normed option.",
                                 metavar=2)
histogram_options.add_option("--kde",
                             dest="kde",
                             default=False,
                             action="store_true",
                             help="Plot a gaussian kernel density estimate along with the histogram")
parser.add_option_group(histogram_options)

histogram2d_options = OptionGroup(parser, "Plotting 2D-histogram")
histogram2d_options.add_option(
    "--histogram2d",
    action="store_true",
    dest="histogram2d",
    default=False,
    help="Compute and plot 2D-histogram from data. -b (--bins) can be used to define the number of bins.")
histogram2d_options.add_option("--logscale", action="store_true", dest="logscale", default=False, help="log scale")
histogram2d_options.add_option("--projection1d",
                               action="store_true",
                               dest="projection1d",
                               default=False,
                               help="Plot 1D histogram for the x and y axis")
parser.add_option_group(histogram2d_options)
parser.add_option("--minval", action="store_true", help='Plot an horizontal line at the minimum y-value')
parser.add_option(
    "--mamin",
    action="store_true",
    help='Plot an horizontal line at the minimum y-value on the moving average (see: moving_average option)')

function_options = OptionGroup(parser, "Plotting functions")
function_options.add_option(
    "-f",
    "--func",
    type=str,
    default=None,
    action='append',
    help=
    "Evaluate and plot the function given as a string. If you want to just plot the function without any piped data just run: 'cat /dev/null | plot -f 'x**2' --xmin 0 --xmax 10'. Multiple functions can be plotted at the same time by giving multiple expression with multiple -f option passed. Numpy functions can be used and given without np prefix (e.g. exp)"
)
function_options.add_option("--func_label",
                            type=str,
                            default=None,
                            action='append',
                            help="optional label for the function")
parser.add_option_group(function_options)

(options, args) = parser.parse_args()

if options.ws is not None:
    options.moving_average = options.ws
if options.slide == 'mean':
    options.slide = numpy.mean
if options.slide == 'max':
    options.slide = numpy.max
if options.slide == 'min':
    options.slide = numpy.min
if options.slide == 'std':
    options.slide = numpy.std

if is_sklearn:
    # from: http://stackoverflow.com/a/19182915/1679629
    def fit_mixture(data, ncomp=2):
        clf = mixture.GMM(n_components=ncomp, covariance_type='full')
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
    plt.xticks(numpy.arange(min(bins) + bin_w / 2, max(bins) + 1, bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
    plt.minorticks_off()


def movingaverage(data, window_size):
    """
    see: http://goo.gl/OMbvco
    """
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(data, window, 'same')


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
        nbins = int(len(data) * .002)
    if nbins < 10:
        nbins = 10
    print("Number of bins used to order the data: %d" % nbins)
    hdd, bins = numpy.histogramdd(data, bins=nbins)
    digits = numpy.asarray([numpy.digitize(v, bins[i], right=True) for i, v in enumerate(data.T)]).T
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
    bin_size = 2 * iqr / (len(data))**(1. / 3)
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
    if 'x' not in expression_string:
        expression_string = expression_string + "+0*x"
    x = numpy.linspace(xlims[0], xlims[1], num=npts)
    y = ne.evaluate(expression_string)
    plt.plot(x, y, label=label, color=color)
    plt.legend()
    return x, y


def polyfit(x, y, degree):
    p = numpy.polyfit(x, y, degree)
    n = degree
    poly = '+'.join([f"{p[i]}*x**{n-i}" for i in range(n)])
    poly += f"+{p[n]}"
    label = '+'.join([f"{p[i]:.2g}*x**{n-i}" for i in range(n)])
    label += f"+{p[n]:.2g}"
    return poly, label


def plot_minval(y):
    if y.ndim == 1:
        y = y[:, None]
    minval = y.min(axis=0)
    for v in minval:
        plt.axhline(y=v, color='blue', linestyle='--', label=f'min={v:.3g}')
    plt.legend()


def do_plot(x,
            y,
            z=None,
            e=None,
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
            yticklabelformat=None):
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
        cmap = cm.rainbow
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
            plt.axvline(x=xvline, color=colors[i], ls='--', label=vlabel_i)
            plt.legend()
    if options.polyfit is not None:
        poly, polylabel = polyfit(x, y, options.polyfit)
        print(f"polyfit: {polylabel}")
        plot_function(poly, (x.min(), x.max()), color='red', label=polylabel)
    if options.semilog is not None:
        if options.semilog == "x":
            plt.xscale('log')
            if yticklabelformat is not None:
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter(yticklabelformat))
        elif options.semilog == 'y':
            plt.yscale('log')
            if yticklabelformat is not None:
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter(yticklabelformat))
    if options.labels is not None:
        labels = options.labels.split(',')
    else:
        labels = None
    if options.subplot is not None:
        n, p = int(options.subplot[0][0]), int(options.subplot[0][1])
        # n: number of row
        # p: numper of column
        gs = matplotlib.gridspec.GridSpec(n, p)
        for i, ydata in enumerate(y.T):
            print(f"Subplot {i} from {y.T.shape}")
            xdata = x.T[i]
            plt.subplot(gs[i])
            # Set the same scale for all axis
            plt.axis((x.min(), x.max(), y.min(), y.max()))
            if labels is not None:
                plt.title(labels[i])
            if options.semilog is not None:
                if options.semilog == "x":
                    plt.semilogx(xdata, ydata)
                elif options.semilog == 'y':
                    plt.semilogy(xdata, ydata)
            else:
                if options.scatter:
                    plt.scatter(xdata, ydata, alpha=options.alpha)
                elif options.histogram2d:
                    plt.hist2d(xdata, ydata, bins=n_bins)
                else:
                    plt.plot(xdata, ydata)
            if i < n * p - p:  # Not last row
                # Hide xtick labels:
                plt.tick_params(labelbottom='off')
            if i % p != 0:  # Not first column
                # Hide y labels:
                plt.tick_params(labelleft='off')
        plt.show()
        return None  # This exits the function now (see: http://stackoverflow.com/a/6190798/1679629)
    if not histogram and not scatter and not histogram2d and not dax:
        if options.minval and options.moving_average is None:
            plot_minval(y)
        if options.moving_average is None:
            if len(x.shape) == 1 and len(y.shape) == 1:
                if not options.bar:
                    plt.plot(x, y)
                else:
                    plt.bar(x, y, align='center')
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
                        plt.plot(xi.flatten(), yi.flatten(), c=colors[i], label=label)
                elif y.shape[1] > 1:
                    colors = cmap(numpy.linspace(0, 1, y.shape[1]))
                    for i, yi in enumerate(y.T):
                        if labels is not None:
                            label = labels[i]
                        else:
                            label = None
                        plt.plot(x.flatten(), yi.flatten(), c=colors[i], label=label)
            if e is not None:
                if len(y.shape) == 1:  # No more than 1 curve
                    plt.fill_between(x.flatten(),
                                     y.flatten() - e.flatten(),
                                     y.flatten() + e.flatten(),
                                     facecolor='gray',
                                     alpha=.5)
                else:
                    for i, errror in enumerate(e.T):
                        plt.fill_between(x[:, i], y[:, i] - e[:, i], y[:, i] + e[:, i], facecolor='gray', alpha=.5)
        else:  # Moving average
            if y.ndim == 1:
                y = y[..., None]
            if options.gray_plot:
                for y_ in y.T:
                    plt.plot(x.flatten(), y_.flatten(), '-', color='gray', alpha=.25, lw=1.)
            ws = options.moving_average  # window size
            for i, y_ in enumerate(y.T):
                if labels is not None:
                    label = labels[i]
                else:
                    label = None
                ma_array = numpy.c_[x.flatten()[int(ws / 2):int(-ws / 2)],
                                    sliding_func(y_, ws, options.slide)[int(ws / 2):int(-ws / 2)]]
                plt.plot(ma_array[:, 0], ma_array[:, 1], linewidth=2., label=label)
                if options.minval:
                    if options.mamin:  # plot min value of the moving average
                        plot_minval(ma_array[:, 1])
                    else:
                        plot_minval(y_)
    elif scatter:
        if len(x.shape) == 1:
            x = x[:, None]
        if len(y.shape) == 1:
            y = y[:, None]
        if x.shape[1] == 1 and y.shape[1] == 1:
            if z is not None:
                if options.sizez:
                    plt.scatter(x, y, s=z * options.size, alpha=options.alpha, facecolor='none', edgecolor='blue')
                else:
                    if not options.plot3d:
                        plt.scatter(x, y, c=z, s=options.size, alpha=options.alpha, cmap=options.cmap)
                        plt.colorbar()
                    else:
                        ax = plt.axes(projection='3d')
                        ax.scatter3D(x, y, z, s=options.size)
            else:
                if options.line:
                    plt.plot(x, y, ',-')
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
                        axHisty.hist(y, bins=n_bins, orientation='horizontal')
                    else:
                        plt.scatter(x, y, s=options.size, alpha=options.alpha)
        else:
            if x.shape[1] > 1:
                colors = cmap(numpy.linspace(0, 1, x.shape[1]))
                for i, xi in enumerate(x.T):
                    yi = y.T[i]
                    if z is not None:
                        zi = z.T[i]
                    if labels is not None:
                        if options.line:
                            plt.plot(xi, yi, ',-', c=colors[i], label=labels[i])
                        else:
                            if options.sizez:
                                plt.scatter(xi,
                                            yi,
                                            label=labels[i],
                                            s=zi * options.size,
                                            alpha=options.alpha,
                                            facecolor='none',
                                            edgecolor=colors[i])
                            else:
                                plt.scatter(xi, yi, c=colors[i], label=labels[i], alpha=options.alpha, s=options.size)
                    else:
                        if options.line:
                            plt.plot(xi, yi, ',-', c=colors[i])
                        else:
                            if options.sizez:
                                plt.scatter(xi,
                                            yi,
                                            s=zi * options.size,
                                            alpha=options.alpha,
                                            facecolor='none',
                                            edgecolor=colors[i])
                            else:
                                plt.scatter(xi, yi, c=colors[i], alpha=options.alpha, s=options.size)
            elif y.shape[1] > 1:
                colors = cmap(numpy.linspace(0, 1, y.shape[1]))
                for i, yi in enumerate(y.T):
                    if options.line:
                        if labels is not None:
                            plt.plot(x, yi, ',-', c=colors[i], label=labels[i])
                        else:
                            plt.plot(x, yi, ',-', c=colors[i])
                    else:
                        if labels is not None:
                            plt.scatter(x, yi, c=colors[i], label=labels[i], alpha=options.alpha)
                        else:
                            plt.scatter(x, yi, c=colors[i], alpha=options.alpha)
        if e is not None:
            if y.shape[1] == 1:
                plt.errorbar(x.flatten(), y.flatten(), yerr=e.flatten(), markersize=0.)
            else:
                for i, yi in enumerate(y.T):
                    plt.errorbar(x[:, i].flatten(), yi, yerr=e[:, i], markersize=0., c=colors[i])
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
            axHisty.hist(y, bins=n_bins, orientation='horizontal')
            axHistx.set_xlim(axScatter.get_xlim())
            axHisty.set_ylim(axScatter.get_ylim())
            # Cool trick that changes the number of tickmarks for the histogram axes
            axHisty.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axHistx.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        else:
            if logscale:
                plt.hist2d(x, y, bins=n_bins, norm=matplotlib.colors.LogNorm())
            else:
                plt.hist2d(x, y, bins=n_bins)
            plt.colorbar()
    elif dax is not None:
        dax = numpy.asarray([int(e) for e in dax])
        plt.close()
        if options.aspect_ratio is None:
            fig, ax1 = plt.subplots()
        else:
            fig, ax1 = plt.subplots(figsize=(options.aspect_ratio[0], options.aspect_ratio[1]))
        ax2 = ax1.twinx()
        for datai, daxi in enumerate(dax):
            if daxi == 1:
                ax1.plot(x, y[:, datai], 'g-', alpha=options.alpha)
            else:
                ax2.plot(x, y[:, datai], 'b-', alpha=options.alpha)
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
            ax1.set_ylabel(options.ylabel, color='g')
        if options.ylabel2 is not None:
            ax2.set_ylabel(options.ylabel2, color='b')
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
            print(f"bins: {bins}")
            histo = plt.hist(y,
                             bins=bins,
                             range=(xmin, xmax),
                             histtype=options.histtype,
                             density=options.normed,
                             label=labels,
                             cumulative=options.cumulative,
                             alpha=options.alpha,
                             edgecolor='black')
            if options.centerbins:
                bins_labels(numpy.int_(bins))
        else:
            if data.ndim > 1:
                for ndim_ in range(data.shape[1]):
                    sel = ~numpy.isnan(data[:, ndim_])
                    if labels is not None:
                        label = labels[ndim_]
                    else:
                        label = 'col %d' % (ndim_ + 1)
                    sns.distplot(data[:, ndim_][sel], label=label)
                plt.legend()
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
                plt.plot(histo[1], fitting, linewidth=3)
    if func is not None:
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
        plt.legend()
    if xticklabels is not None:
        plt.xticks(ticks=x, labels=xticklabels, rotation=90)
    if options.outfilename is None:
        plt.show()
    else:
        metadata = dict()
        datastr = ''
        for l in data:
            if isinstance(l, Iterable):
                for e in l:
                    datastr += f"{e} "
            else:
                datastr += f"{l}"
            datastr += '\n'
        plt.savefig(options.outfilename)
        add_metadata(options.outfilename, datastr)


def add_metadata(filename, datastr, key='data'):
    """
    Add metadata to a png file
    """
    metadata = PngInfo()
    metadata.add_text(key, datastr, zip=True)
    metadata.add_text('cwd', os.getcwd())
    metadata.add_text('hostname', socket.gethostname())
    if options.labels is not None:
        metadata.add_text('labels', options.labels)
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
    if "labels" in im.info:
        datastr += f'#labels:{im.info["labels"]}\n'
    datastr += im.info['data']
    return datastr


if options.read_data is not None:
    datastr = read_metadata(options.read_data)
    print(datastr)
    sys.exit()

if options.roc:
    options.delimiter = ','
if options.fields is None:
    dtype = numpy.float
else:
    dtype = None  # to be able to read also text
data = numpy.genfromtxt(sys.stdin, invalid_raise=False, delimiter=options.delimiter, dtype=dtype)
if options.fields is not None:
    data = numpy.asarray(data.tolist(), dtype='U13')  # For formatting arrays with both data and text
xticklabels = None
n = data.shape[0]
if options.transpose:
    data = data.T
if n > 1:
    if len(data.shape) == 1:
        x = range(n)
        y = data
        z = None
        e = None
        x = numpy.asarray(x)[:, None]
        y = numpy.asarray(y)[:, None]
    else:
        if options.fields is None:
            x = data[:, 0]
            y = data[:, 1:]
            z = None
            e = None
        else:
            x, y, z, e, xticklabels = [], [], [], [], []
            for i, field in enumerate(options.fields):
                if field == 'x':
                    x.append(data[:, i])
                elif field == 'y':
                    y.append(data[:, i])
                elif field == 'z':
                    z.append(data[:, i])
                elif field == 'e':
                    e.append(data[:, i])
                elif field == 'l':  # label for bar plot
                    # xticklabels.extend(data[:, i])
                    xticklabels.extend(data[:, i])
                elif field == '*':
                    y = data.T
            if len(xticklabels) == 0:
                xticklabels = None
            x, y, z, e = numpy.asarray(x, dtype=float).T, numpy.asarray(y, dtype=float).T, numpy.asarray(
                z, dtype=float).T, numpy.asarray(e, dtype=float).T
            if len(z) == 0:
                z = None
            # else:
            #     data_sorted = sort_scatter_data(numpy.c_[x,y,z])
            #     x = data_sorted[:,0][:,None]
            #     y = data_sorted[:,1][:,None]
            #     z = data_sorted[:,2]
            if len(e) == 0:
                e = None
        x = numpy.asarray(x)[:, None]
    if options.roc:
        negatives = data[:, 0]
        positives = data[:, 1]
        x, y, auc = ROC.ROC(positives, negatives)
        print(f'AUC: {auc:.2f}')
    x = numpy.squeeze(x)
    y = numpy.squeeze(y)
    if x.shape == (0, ):  # If not x field is given with fields option
        x = numpy.arange(y.shape[0])
    plt.clf()
    if options.normalize == 'x':
        xmin, xmax = numpy.min(x, axis=0), numpy.max(x, axis=0)
        x = (x - xmin) / (xmax - xmin)
    if options.normalize == 'y':
        ymin, ymax = numpy.min(y, axis=0), numpy.max(y, axis=0)
        y = (y - ymin) / (ymax - ymin)
    do_plot(x,
            y,
            z,
            e,
            vline=options.vline,
            vlabel=options.vlabel,
            xticklabels=xticklabels,
            yticklabelformat=options.yticklabelformat)
else:
    plot_functions(options.func, xlims=[options.xmin, options.xmax], func_label=options.func_label)
    plt.show()
