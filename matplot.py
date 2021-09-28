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
# Allow to print unicode text (see: http://stackoverflow.com/a/21190382/1679629)
# reload(sys)
# sys.setdefaultencoding('utf8')
##############################
import matplotlib.pyplot as plt
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
    print(
        "sklearn is not installed you cannot use the Gaussian Mixture Model option"
    )
    is_sklearn = False
from prettytable import PrettyTable

parser = OptionParser()
parser.add_option("--save", help="Save the file", type=str, dest='outfilename')
parser.add_option("--title", help="Title of the plot", type=str)
parser.add_option("-d",
                  "--delimiter",
                  help="Delimiter to use to read the data",
                  default=None)
parser.add_option("--xlabel",
                  dest="xlabel",
                  default=None,
                  type='str',
                  help="x axis label")
parser.add_option("--ylabel",
                  dest="ylabel",
                  default=None,
                  type='str',
                  help="y axis label")
parser.add_option("--xmin",
                  dest="xmin",
                  default=None,
                  type='float',
                  help="Minimum x-value")
parser.add_option("--xmax",
                  dest="xmax",
                  default=None,
                  type='float',
                  help="Maximum x-value")
parser.add_option("--ymin",
                  dest="ymin",
                  default=None,
                  type='float',
                  help="Lower limit for y-axis")
parser.add_option("--ymax",
                  dest="ymax",
                  default=None,
                  type='float',
                  help="Upper limit for y-axis")
parser.add_option("--polyfit",
                  dest="polyfit",
                  default=None,
                  type='int',
                  help="Least squares polynomial fit, with the given degree.")
moving_average_options = OptionGroup(parser, "Moving average")
moving_average_options.add_option(
    "--moving_average",
    dest="moving_average",
    default=None,
    type='int',
    help="Plot a moving average on the data with the\
                                  given window size. It also prints the values on stdout",
    metavar=10)
moving_average_options.add_option(
    "--no_gray_plot",
    dest="gray_plot",
    default=True,
    action="store_false",
    help="Do not plot original data in gray with moving_average option")
parser.add_option_group(moving_average_options)
parser.add_option("--bar",
                  dest="bar",
                  default=False,
                  action="store_true",
                  help="Simple bar plot for single unidimensional data")
parser.add_option(
    "--normalize",
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
parser.add_option("--transpose",
                  dest="transpose",
                  default=False,
                  action="store_true",
                  help="Transpose the input data")
parser.add_option(
    "--subplot",
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
scatter_options.add_option("--plot3d",
                           help="Scatter plot in 3D",
                           action='store_true',
                           default=False)
scatter_options.add_option("--alpha",
                           type=float,
                           default=1.,
                           help='Transparency of the scatter dots')
scatter_options.add_option("--fields",
                           dest="fields",
                           default=None,
                           type='str',
                           help="Fields for the data; e.g. 'xyxy'. By default\
                           the first column is for x data and the other for y data. \
If a 'z' field is given, this field is used to color \
the scatter dots. \
If a 'e' field is given it is used to plot the error. \
If --fields='*' is given all the columns are considered as y values.")
scatter_options.add_option(
    "-s",
    "--size",
    default=2.,
    type=float,
    help='size of the dots for the scatter plot (default: 2)')
scatter_options.add_option("--sizez",
                           action='store_true',
                           help="Use a variable size given by the z-field")
parser.add_option("--labels",
                  dest="labels",
                  default=None,
                  type='str',
                  help="Comma separated list of labels for each field \
                  defined with the --fields option.")
scatter_options.add_option("--line",
                           dest='line',
                           default=False,
                           action="store_true",
                           help="Plot line between points")
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
histogram_options.add_option(
    "--histtype",
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
    help=
    "If True, the first element of the return tuple will be the counts normalized to form a probability density"
)
histogram_options.add_option("--cumulative",
                             action="store_true",
                             dest="cumulative",
                             default=False,
                             help="Cumulative histogram")
if is_sklearn:
    histogram_options.add_option(
        "--gmm",
        dest="gmm",
        default=None,
        type='int',
        help=
        "Gaussian Mixture Model with n components. Trigger the normed option.",
        metavar=2)
histogram_options.add_option(
    "--kde",
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
    help=
    "Compute and plot 2D-histogram from data. -b (--bins) can be used to define the number of bins."
)
histogram2d_options.add_option("--logscale",
                               action="store_true",
                               dest="logscale",
                               default=False,
                               help="log scale")
histogram2d_options.add_option("--projection1d",
                               action="store_true",
                               dest="projection1d",
                               default=False,
                               help="Plot 1D histogram for the x and y axis")
parser.add_option_group(histogram2d_options)

function_options = OptionGroup(parser, "Plotting functions")
function_options.add_option(
    "-f",
    "--func",
    type=str,
    default=None,
    action='append',
    help=
    "Evaluate and plot the function given as a string. If you want to just plot the function without any piped data just run: 'cat /dev/null | plot -f 'x**2' --xmin 0 --xmax 10'. Multiple functions can be plotted at the same time by giving multiple expression with multiple -f option passed."
)
parser.add_option_group(function_options)

(options, args) = parser.parse_args()

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


def prettyprint(A):
    """
    Pretty print of an array (A)
    """
    x = PrettyTable(A.dtype.names, header=False, border=False)
    for row in A:
        x.add_row(row)
    print(x)


def movingaverage(data, window_size):
    """
    see: http://goo.gl/OMbvco
    """
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(data, window, 'same')


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
    digits = numpy.asarray([
        numpy.digitize(v, bins[i], right=True) for i, v in enumerate(data.T)
    ]).T
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


def plot_functions(expression_strings, xlims, npts=100):
    """
    Plot a list of functions given as a list of expression strings
    """
    for expression_string in expression_strings:
        plot_function(expression_string, xlims, npts=npts)


def plot_function(expression_string, xlims, npts=100, color=None, label=None):
    """
    Plot a function given as an expression string
    """
    x = numpy.linspace(xlims[0], xlims[1], num=npts)
    y = eval(expression_string)
    if label is None:
        label = expression_string
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
            func=options.func):
    if xmin is None and func is not None:
        xmin = x.min()
    if xmax is None and func is not None:
        xmax = x.max()
    if options.polyfit is not None:
        poly, polylabel = polyfit(x, y, options.polyfit)
        print(f"polyfit: {polylabel}")
        plot_function(poly, (x.min(), x.max()), color='red', label=polylabel)
    if options.semilog is not None:
        if options.semilog == "x":
            plt.xscale('log')
        elif options.semilog == 'y':
            plt.yscale('log')
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
    if not histogram and not scatter and not histogram2d:
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
                    colors = cm.rainbow(numpy.linspace(0, 1, x.shape[1]))
                    for i, xi in enumerate(x.T):
                        yi = y.T[i]
                        plt.plot(xi.flatten(), yi.flatten(), c=colors[i])
                elif y.shape[1] > 1:
                    colors = cm.rainbow(numpy.linspace(0, 1, y.shape[1]))
                    for i, yi in enumerate(y.T):
                        plt.plot(x.flatten(), yi.flatten(), c=colors[i])
            if e is not None:
                if len(y.shape) == 1:  # No more than 1 curve
                    plt.fill_between(x.flatten(),
                                     y.flatten() - e.flatten(),
                                     y.flatten() + e.flatten(),
                                     facecolor='gray',
                                     alpha=.5)
                else:
                    for i, errror in enumerate(e.T):
                        plt.fill_between(x[:, i],
                                         y[:, i] - e[:, i],
                                         y[:, i] + e[:, i],
                                         facecolor='gray',
                                         alpha=.5)
        else:  # Moving average
            if options.gray_plot:
                plt.plot(x.flatten(),
                         y.flatten(),
                         '-',
                         color='gray',
                         alpha=.25)
            ws = options.moving_average  # window size
            ma_array = numpy.c_[x.flatten()[int(ws / 2):int(-ws / 2)],
                                movingaverage(y.flatten(), ws)[int(ws /
                                                                   2):int(-ws /
                                                                          2)]]
            plt.plot(ma_array[:, 0], ma_array[:, 1], 'r', linewidth=1.5)
    elif scatter:
        if len(x.shape) == 1:
            x = x[:, None]
        if len(y.shape) == 1:
            y = y[:, None]
        if x.shape[1] == 1 and y.shape[1] == 1:
            if z is not None:
                if options.sizez:
                    plt.scatter(x,
                                y,
                                s=z * options.size,
                                alpha=options.alpha,
                                facecolor='none',
                                edgecolor='blue')
                else:
                    if not options.plot3d:
                        plt.scatter(x,
                                    y,
                                    c=z,
                                    s=options.size,
                                    alpha=options.alpha)
                        plt.colorbar()
                    else:
                        ax = plt.axes(projection='3d')
                        ax.scatter3D(x, y, z, s=options.size)
            else:
                if options.line:
                    plt.plot(x, y, ',-')
                else:
                    plt.scatter(x, y, s=options.size, alpha=options.alpha)
        else:
            if x.shape[1] > 1:
                colors = cm.rainbow(numpy.linspace(0, 1, x.shape[1]))
                for i, xi in enumerate(x.T):
                    yi = y.T[i]
                    zi = z.T[i]
                    if labels is not None:
                        if options.line:
                            plt.plot(xi,
                                     yi,
                                     ',-',
                                     c=colors[i],
                                     label=labels[i])
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
                                plt.scatter(xi,
                                            yi,
                                            c=colors[i],
                                            label=labels[i],
                                            alpha=options.alpha)
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
                                plt.scatter(xi,
                                            yi,
                                            c=colors[i],
                                            alpha=options.alpha)
            elif y.shape[1] > 1:
                colors = cm.rainbow(numpy.linspace(0, 1, y.shape[1]))
                for i, yi in enumerate(y.T):
                    if options.line:
                        if labels is not None:
                            plt.plot(x, yi, ',-', c=colors[i], label=labels[i])
                        else:
                            plt.plot(x, yi, ',-', c=colors[i])
                    else:
                        if labels is not None:
                            plt.scatter(x,
                                        yi,
                                        c=colors[i],
                                        label=labels[i],
                                        alpha=options.alpha)
                        else:
                            plt.scatter(x,
                                        yi,
                                        c=colors[i],
                                        alpha=options.alpha)
        if e is not None:
            if y.shape[1] == 1:
                plt.errorbar(x.flatten(),
                             y.flatten(),
                             yerr=e.flatten(),
                             markersize=0.)
            else:
                for i, yi in enumerate(y.T):
                    plt.errorbar(x[:, i].flatten(),
                                 yi,
                                 yerr=e[:, i],
                                 markersize=0.,
                                 c=colors[i])
        if options.labels is not None:
            plt.legend()
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
                axScatter.hist2d(x,
                                 y,
                                 bins=n_bins,
                                 norm=matplotlib.colors.LogNorm())
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
                             cumulative=options.cumulative)
            if options.labels is not None:
                plt.legend()
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
                print("Gaussian Mixture Model with %d components" %
                      options.gmm)
                print("Means: %s" % ms)
                print("Variances: %s" % cs)
                print("Weights: %s" % ws)
                fitting = numpy.zeros_like(histo[1])
                for w, m, c in zip(ws, ms, cs):
                    fitting += w * matplotlib.mlab.normpdf(histo[1], m, c)
                plt.plot(histo[1], fitting, linewidth=3)
    if func is not None:
        plot_functions(func, [xmin, xmax])
    if not projection1d:
        if options.xlabel is not None:
            plt.xlabel(options.xlabel)
        if options.ylabel is not None:
            plt.ylabel(options.ylabel)
    set_x_lim(options.xmin, options.xmax)
    set_y_lim(options.ymin, options.ymax)
    if options.title is not None:
        plt.title(options.title)
    if options.outfilename is None:
        plt.show()
    else:
        plt.savefig(options.outfilename)


data = numpy.genfromtxt(sys.stdin,
                        invalid_raise=False,
                        delimiter=options.delimiter)
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
            x, y, z, e = [], [], [], []
            for i, field in enumerate(options.fields):
                if field == 'x':
                    x.append(data[:, i])
                elif field == 'y':
                    y.append(data[:, i])
                elif field == 'z':
                    z.append(data[:, i])
                elif field == 'e':
                    e.append(data[:, i])
                elif field == '*':
                    y = data.T
            x, y, z, e = numpy.asarray(x).T, numpy.asarray(y).T, numpy.asarray(
                z).T, numpy.asarray(e).T
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
    do_plot(x, y, z, e)
else:
    plot_functions(options.func, xlims=[options.xmin, options.xmax])
    plt.show()
