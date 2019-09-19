#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2016 09 26
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import time
import sys
# Allow to print unicode text (see: http://stackoverflow.com/a/21190382/1679629)
#reload(sys)
#sys.setdefaultencoding('utf8')
##############################
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
try:
    import seaborn as sns
    sns.set_context('talk')
except ImportError:
    print("seaborn not installed")
# For publication quality plot
    params = {
       'axes.labelsize': 14,
       'font.size': 14,
       'legend.fontsize': 12,
       'xtick.labelsize': 12,
       'ytick.labelsize': 12,
       'text.usetex': False,
       #'axes.color_cycle'    : 'b, g, r, c, m, y, k',
       }
    plt.rcParams.update(params)
    pass
import numpy
from optparse import OptionParser
from optparse import OptionGroup
try:
    from sklearn import mixture
    is_sklearn = True
except ImportError:
    print("sklearn is not installed you cannot use the Gaussian Mixture Model option")
    is_sklearn = False
from prettytable import PrettyTable


parser = OptionParser()
interactive_options = OptionGroup(parser, "Interactive")
interactive_options.add_option("--interactive", dest="interactive", default=False,
                               action="store_true",
                               help="Plot data interactively from stdin stream. \
                               Example usage:                                   \
                               tail -n 1000 -f file.txt | grep --line-buffered 'pattern' | awk '{print $2; system(\"\")}' | matplot --interactive")
interactive_options.add_option("--tail", dest="tail", default=None, type='int',
                               help="Plot only the last N lines in interactive plotting",
                               metavar="N")
parser.add_option_group(interactive_options)
parser.add_option("--xlabel", dest="xlabel", default=None, type='str',
                    help="x axis label")
parser.add_option("--ylabel", dest="ylabel", default=None, type='str',
                    help="y axis label")
parser.add_option("--ymin", dest="ymin", default=None, type='float',
                  help="Lower limit for y-axis")
parser.add_option("--ymax", dest="ymax", default=None, type='float',
                  help="Upper limit for y-axis")
moving_average_options = OptionGroup(parser, "Moving average")
moving_average_options.add_option("--moving_average", dest="moving_average", default=None,
                  type='int', help="Plot a moving average on the data with the\
                  given window size. It also prints the values on stdout", metavar=10)
moving_average_options.add_option("--no_gray_plot", dest="gray_plot", default=True,
                  action="store_false",
                  help="Do not plot original data in gray with moving_average option")
parser.add_option_group(moving_average_options)
parser.add_option("--bar", dest="bar", default=False, action="store_true",
                  help="Simple bar plot for single unidimensional data")
parser.add_option("--normalize", dest="normalize", default=None, type='str',
                  help='Normalize the values to 1. If normalize=x, normalize the \
x values, if normalize=y then normalize the y values. Normalization scales all \
numeric variables in the range [0,1] given the formula: (x-xmin)/(xmax-xmin)',
                  metavar='x')
parser.add_option("--semilog", dest="semilog", default=None, type='str',
                  metavar='x', help="Log scale for the given axis (x or y)")
parser.add_option("--transpose", dest="transpose", default=False,
                  action="store_true", help="Transpose the input data")
parser.add_option("--subplot", dest="subplot", nargs=2, action='append',
                  default=None,
                  help="Arrange multiple plot on a grid of shape n×p, given by \
                  arguments: --subplot n p")

scatter_options = OptionGroup(parser, "Scatter plot")
scatter_options.add_option("--scatter", action="store_true",
                  dest="scatter", default=False,
                  help="Scatter plot of the (x,y) data")
scatter_options.add_option("--fields", dest="fields", default=None, type='str',
                  help="Fields for the data; e.g. 'xyxy'. By default\
                  the first column is for x data and the other for y data. \
If a 'z' field is given, this field is used to color \
the scatter dots. \
If a 'e' field is given it is used to plot the error. \
If --fields='*' is given all the columns are considered as y values.")
parser.add_option("--labels", dest="labels", default=None, type='str',
                           help="Comma separated list of labels for each field \
defined with the --fields option.")
scatter_options.add_option("--line", dest='line', default=False,
                           action="store_true",
                           help="Plot line between points")
parser.add_option_group(scatter_options)


histogram_options = OptionGroup(parser, "Plotting histogram")
histogram_options.add_option("-H", "--histogram",
                    action="store_true", dest="histogram", default=False,
                    help="Compute and plot histogram from data")
histogram_options.add_option("-b", "--bins",
                    dest="n_bins", default=-1, type="int",
                    help="Number of bins in the histogram. If -1 (default) the optimal number of bins is determined using the Freedman-Diaconis rule.",
                    metavar=10)
histogram_options.add_option("--xmin", dest="xmin", default=None, type='float',
                            help="Minimum x-value of the histogram")
histogram_options.add_option("--xmax", dest="xmax", default=None, type='float',
                            help="Maximum x-value of the histogram")
histogram_options.add_option("--histtype", dest="histtype", default='bar', type='str',
                            help="Histogram type: bar, barstacked, step, stepfilled", metavar='bar')
histogram_options.add_option("--normed", dest="normed", default=False, action="store_true",
                            help="If True, the first element of the return tuple will be the counts normalized to form a probability density")
if is_sklearn:
    histogram_options.add_option("--gmm", dest="gmm", default=None, type='int',
                                help="Gaussian Mixture Model with n components. Trigger the normed option.", metavar=2)
histogram_options.add_option("--kde", dest="kde", default=False, action="store_true",
                            help="Plot a gaussian kernel density estimate along with the histogram")
parser.add_option_group(histogram_options)

histogram2d_options = OptionGroup(parser, "Plotting 2D-histogram")
histogram2d_options.add_option("--histogram2d", action="store_true",
                                dest="histogram2d", default=False,
                                help="Compute and plot 2D-histogram from data. -b (--bins) can be used to define the number of bins.")
histogram2d_options.add_option("--logscale", action="store_true",
                                dest="logscale", default=False,
                                help="log scale")
histogram2d_options.add_option("--projection1d", action="store_true",
                                dest="projection1d", default=False,
                                help="Plot 1D histogram for the x and y axis")
parser.add_option_group(histogram2d_options)
(options, args) = parser.parse_args()

if options.interactive:
    plt.ion() # Invoke matplotlib's interactive mode

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
    x = PrettyTable(A.dtype.names, header=False,border=False)
    for row in A:
        x.add_row(row)
    print(x)


def movingaverage(data, window_size):
    """
    see: http://goo.gl/OMbvco
    """
    window= numpy.ones(int(window_size))/float(window_size)
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
    print("Number of bins used to order the data: %d"%nbins)
    hdd, bins = numpy.histogramdd(data, bins=nbins)
    digits = numpy.asarray([numpy.digitize(v, bins[i], right=True) for i,v in enumerate(data.T)]).T
    digits[digits==nbins]-=1
    counts = numpy.asarray([hdd[tuple(e)] for e in digits])
    return data[counts.argsort()][::-1]


def freedman_diaconis_rule(data):
    """
    Compute the optimal number of bins accordingly to the Freedman–Diaconis rule.
    See: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """
    q75, q25 = numpy.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_size = 2*iqr/(len(data))**(1./3)
    n_bins = numpy.ceil(numpy.ptp(data)/bin_size)
    if numpy.isnan(n_bins) or n_bins == numpy.inf:
        n_bins = 2
    else:
        n_bins = int(n_bins)
    print("Freedman–Diaconis optimal number of bins: %d"%n_bins)
    return n_bins

def set_y_lim(ymin, ymax):
    axes = plt.gca()
    limits = plt.axis()
    if ymin is None:
        ymin = limits[-2]
    if ymax is None:
        ymax = limits[-1]
    axes.set_ylim([ymin,ymax])

def do_plot(x, y, z=None, e=None, histogram=options.histogram, scatter=options.scatter,
            histogram2d=options.histogram2d, logscale=options.logscale,
            projection1d=options.projection1d,
            n_bins=options.n_bins, xmin=options.xmin, xmax=options.xmax):
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
            plt.subplot(gs[i])
            # Set the same scale for all axis
            plt.axis((x.min(),x.max(),y.min(),y.max()))
            if options.semilog is not None:
                if options.semilog == "x":
                    plt.semilogx(x, ydata)
                elif options.semilog == 'y':
                    plt.semilogy(x, ydata)
            else:
                plt.plot(x, ydata)
            if i < n*p - p: # Not last row
                # Hide xtick labels:
                plt.tick_params(labelbottom='off')
            if i % p != 0: # Not first column
                # Hide y labels:
                plt.tick_params(labelleft='off')
        plt.show()
        return None # This exits the function now (see: http://stackoverflow.com/a/6190798/1679629)
    if not histogram and not scatter and not histogram2d:
        if options.moving_average is None:
            if len(x.shape) == 1 and len(y.shape) == 1:
                if not options.bar:
                    plt.plot(x, y)
                else:
                    plt.bar(x, y, align='center')
            else:
                if len(x.shape) == 1:
                    x = x[:,None]
                if x.shape[1] > 1:
                    colors = cm.rainbow(numpy.linspace(0,1,x.shape[1]))
                    for i, xi in enumerate(x.T):
                        yi = y.T[i]
                        plt.plot(xi.flatten(), yi.flatten(), c=colors[i])
                elif y.shape[1] > 1:
                    colors = cm.rainbow(numpy.linspace(0,1,y.shape[1]))
                    for i, yi in enumerate(y.T):
                        plt.plot(x.flatten(), yi.flatten(), c=colors[i])
            if e is not None:
                if len(y.shape) == 1: # No more than 1 curve
                    plt.fill_between(x.flatten(), y.flatten() - e.flatten(),
                                     y.flatten() + e.flatten(), facecolor='gray',
                                     alpha=.5)
                else:
                    for i, errror in enumerate(e.T):
                        plt.fill_between(x[:, i], y[:, i] - e[:, i],
                                         y[:, i] + e[:, i], facecolor='gray',
                                         alpha=.5)
        else: # Moving average
            if options.gray_plot:
                plt.plot(x.flatten(), y.flatten(), '-', color='gray', alpha=.25)
            ws =  options.moving_average # window size
            ma_array = numpy.c_[x.flatten()[ws/2:-ws/2],
                                movingaverage(y.flatten(), ws)[ws/2:-ws/2]]
            plt.plot(ma_array[:, 0], ma_array[:, 1], 'r',
                     linewidth=1.5)
            #prettyprint(ma_array)
    elif scatter:
        if len(x.shape) == 1:
            x = x[:,None]
        if len(y.shape) == 1:
            y = y[:,None]
        if x.shape[1] == 1 and y.shape[1] == 1:
            if z is not None:
                plt.scatter(x,y,c=z)
                plt.colorbar()
            else:
                if options.line:
                    plt.plot(x, y, ',-')
                else:
                    plt.scatter(x,y)
        else:
            if x.shape[1] > 1:
                colors = cm.rainbow(numpy.linspace(0,1,x.shape[1]))
                for i, xi in enumerate(x.T):
                    yi = y.T[i]
                    if labels is not None:
                        if options.line:
                            plt.plot(xi, yi, ',-', c=colors[i], label=labels[i])
                        else:
                            plt.scatter(xi, yi, c=colors[i], label=labels[i])
                    else:
                        if options.line:
                            plt.plot(xi, yi, ',-', c=colors[i])
                        else:
                            plt.scatter(xi, yi, c=colors[i])
            elif y.shape[1] > 1:
                colors = cm.rainbow(numpy.linspace(0,1,y.shape[1]))
                for i, yi in enumerate(y.T):
                    if options.line:
                        if labels is not None:
                            plt.plot(x, yi,  ',-', c=colors[i], label=labels[i])
                        else:
                            plt.plot(x, yi,  ',-', c=colors[i])
                    else:
                        if labels is not None:
                            plt.scatter(x, yi, c=colors[i], label=labels[i])
                        else:
                            plt.scatter(x, yi, c=colors[i])
        if e is not None:
            if y.shape[1] == 1:
                plt.errorbar(x.flatten(), y.flatten(), yerr=e.flatten(),
                             markersize=0.)
            else:
                for i, yi in enumerate(y.T):
                    plt.errorbar(x[:, i].flatten(), yi, yerr=e[:,i],
                                 markersize=0., c=colors[i])
        if options.labels is not None:
            plt.legend()
    elif histogram2d:
        x, y = x.flatten(), y.flatten()
        if projection1d: #1D projections of histogram
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
            #Cool trick that changes the number of tickmarks for the histogram axes
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
            xmin = min(y)
        if xmax is None:
            xmax = max(y)
        if is_sklearn and options.gmm is not None:
            options.normed = True
        if n_bins == -1:
            n_bins = freedman_diaconis_rule(y)
        if not options.kde:
            histo = plt.hist(y, bins=n_bins, range=(xmin,xmax), histtype=options.histtype, normed=options.normed)
        else:
            if data.ndim > 1:
                for ndim_ in range(data.ndim):
                    sel = ~numpy.isnan(data[:, ndim_])
                    if labels is not None:
                        label = labels[ndim_]
                    else:
                        label = 'col %d'%(ndim_+1)
                    sns.distplot(data[:, ndim_][sel], label=label)
                plt.legend()
            else:
                sns.distplot(y)
        if is_sklearn:
            if options.gmm is not None:
                ms, cs, ws = fit_mixture(y, ncomp = options.gmm)
                print("Gaussian Mixture Model with %d components"%options.gmm)
                print("Means: %s"%ms)
                print("Variances: %s"%cs)
                print("Weights: %s"%ws)
                fitting = numpy.zeros_like(histo[1])
                for w, m, c in zip(ws, ms, cs):
                    fitting += w*matplotlib.mlab.normpdf(histo[1],m,c)
                plt.plot(histo[1], fitting, linewidth=3)
    if not projection1d:
        if options.xlabel is not None:
            plt.xlabel(options.xlabel)
        if options.ylabel is not None:
            plt.ylabel(options.ylabel)
    set_y_lim(options.ymin, options.ymax)
    if options.interactive:
        plt.draw()
        plt.pause(0.0001)
    else:
        plt.show()

data = None
while True:
    if options.interactive:
        dataline = sys.stdin.readline()
        if len(dataline) == 0:
            time.sleep(2)
        if data is None:
            data = numpy.asarray(dataline.split(), dtype=numpy.float)
            print(data.shape)
        else:
            if options.tail is not None:
                data = numpy.r_[data[-(options.tail-1):].flatten(), numpy.asarray(dataline.split(), dtype=numpy.float)]
            else:
                data = numpy.r_[data.flatten(), numpy.asarray(dataline.split(), dtype=numpy.float)]
        if options.fields is not None:
            n_field = len(options.fields)
            n_point = len(data)/n_field
            data = data.reshape(n_point, n_field)
            n = n_point
        else:
            n = data.shape[0]
    else:
        data = numpy.genfromtxt(sys.stdin, invalid_raise=False)
        n = data.shape[0]
    if options.transpose:
        data = data.T
    if n > 1:
        if len(data.shape) == 1:
            x = range(n)
            y = data
            z = None
            e = None
            x = numpy.asarray(x)[:,None]
            y = numpy.asarray(y)[:,None]
        else:
            if options.fields is None:
                x = data[:,0]
                y = data[:,1:]
                z = None
                e = None
            else:
                x, y, z, e = [], [], [], []
                for i, field in enumerate(options.fields):
                    if field == 'x':
                        x.append(data[:,i])
                    elif field == 'y':
                        y.append(data[:,i])
                    elif field == 'z':
                        z.append(data[:,i])
                    elif field == 'e':
                        e.append(data[:,i])
                    elif field == '*':
                        y = data.T
                x, y, z, e = numpy.asarray(x).T, numpy.asarray(y).T, numpy.asarray(z).T,\
                             numpy.asarray(e).T
                if len(z) == 0:
                    z = None
                else:
                    data_sorted = sort_scatter_data(numpy.c_[x,y,z])
                    x = data_sorted[:,0][:,None]
                    y = data_sorted[:,1][:,None]
                    z = data_sorted[:,2]
                if len(e) == 0:
                    e = None
            x = numpy.asarray(x)[:,None]
        x = numpy.squeeze(x)
        y = numpy.squeeze(y)
        if x.shape == (0,): # If not x field is given with fields option
            x = numpy.arange(y.shape[0])
        #if not options.interactive:
        #    print "Shape of x and y data: %s %s"%(x.shape, y.shape)
        #else:
        #    print "x: %s; y: %s"%(x[-1], y[-1])
        plt.clf()
        if options.normalize == 'x':
            xmin, xmax = numpy.min(x, axis=0), numpy.max(x, axis=0)
            x = (x - xmin)/(xmax - xmin)
        if options.normalize == 'y':
            ymin, ymax = numpy.min(y, axis=0), numpy.max(y, axis=0)
            y = (y - ymin)/(ymax - ymin)
        do_plot(x, y, z, e)
    if not options.interactive:
        break
