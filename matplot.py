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

import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy
from optparse import OptionParser
from optparse import OptionGroup
try:
    from sklearn import mixture
    is_sklearn = True
except ImportError:
    print "sklearn is not installed you cannot use the Gaussian Mixture Model option"
    is_sklearn = False

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
#plt.axes(frameon=0)

parser = OptionParser()
parser.add_option("--xlabel", dest="xlabel", default=None, type='str',
                    help="x axis label")
parser.add_option("--ylabel", dest="ylabel", default=None, type='str',
                    help="y axis label")
parser.add_option("--moving_average", dest="moving_average", default=None,
                  type='int', help="Plot a moving average on the data with the\
                  given window size", metavar=10)
parser.add_option("--bar", dest="bar", default=False, action="store_true",
                  help="Simple bar plot for single unidimensional data")

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
parser.add_option_group(scatter_options)


histogram_options = OptionGroup(parser, "Plotting histogram")
histogram_options.add_option("-H", "--histogram",
                    action="store_true", dest="histogram", default=False,
                    help="Compute and plot histogram from data")
histogram_options.add_option("-b", "--bins",
                    dest="n_bins", default=10, type="int",
                    help="Number of bins in the histogram", metavar=10)
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
    print "Number of bins used to order the data: %d"%nbins
    hdd, bins = numpy.histogramdd(data, bins=nbins)
    digits = numpy.asarray([numpy.digitize(v, bins[i], right=True) for i,v in enumerate(data.T)]).T
    digits[digits==nbins]-=1
    counts = numpy.asarray([hdd[tuple(e)] for e in digits])
    return data[counts.argsort()][::-1]

def do_plot(x, y, z=None, e=None, histogram=options.histogram, scatter=options.scatter,
            histogram2d=options.histogram2d, logscale=options.logscale,
            projection1d=options.projection1d,
            n_bins=options.n_bins, xmin=options.xmin, xmax=options.xmax):
    if not histogram and not scatter and not histogram2d:
        if options.moving_average is None:
            if len(x.shape) == 1 and len(y.shape) == 1:
                if not options.bar:
                    plt.plot(x, y)
                else:
                    plt.bar(x, y)
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
                plt.fill_between(x.flatten(), y.flatten() - e.flatten(),
                                 y.flatten() + e.flatten(), facecolor='gray',
                                 alpha=.5)
            plt.grid()
        else: # Moving average
            plt.plot(x.flatten(), y.flatten(), '-', color='gray', alpha=.25)
            ws =  options.moving_average # window size
            plt.plot(x.flatten()[ws:-ws], movingaverage(y.flatten(), ws)[ws:-ws], 'r',
                     linewidth=1.5)
            plt.grid()
    elif scatter:
        if x.shape[1] == 1 and y.shape[1] == 1:
            if z is not None:
                plt.scatter(x,y,c=z)
                plt.colorbar()
            else:
                plt.scatter(x,y)
        else:
            if x.shape[1] > 1:
                colors = cm.rainbow(numpy.linspace(0,1,x.shape[1]))
                for i, xi in enumerate(x.T):
                    yi = y.T[i]
                    plt.scatter(xi, yi, c=colors[i])
            elif y.shape[1] > 1:
                colors = cm.rainbow(numpy.linspace(0,1,y.shape[1]))
                for i, yi in enumerate(y.T):
                    plt.scatter(x, yi, c=colors[i])
        if e is not None:
            plt.errorbar(x.flatten(), y.flatten(), yerr=e.flatten(),
                         markersize=0.)
        plt.grid()
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
                plt.grid()
            else:
                plt.hist2d(x, y, bins=n_bins)
                plt.grid()
            plt.colorbar()
    else:
        if xmin is None:
            xmin = min(y)
        if xmax is None:
            xmax = max(y)
        if is_sklearn and options.gmm is not None:
            options.normed = True
        histo = plt.hist(y, bins=n_bins, range=(xmin,xmax), histtype=options.histtype, normed=options.normed)
        plt.grid()
        if is_sklearn:
            if options.gmm is not None:
                ms, cs, ws = fit_mixture(y, ncomp = options.gmm)
                print "Gaussian Mixture Model with %d components"%options.gmm
                print "Means: %s"%ms
                print "Variances: %s"%cs
                print "Weights: %s"%ws
                fitting = numpy.zeros_like(histo[1])
                for w, m, c in zip(ws, ms, cs):
                    fitting += w*matplotlib.mlab.normpdf(histo[1],m,numpy.sqrt(c))
                plt.plot(histo[1], fitting, linewidth=3)
    if not projection1d:
        if options.xlabel is not None:
            plt.xlabel(options.xlabel)
        if options.ylabel is not None:
            plt.ylabel(options.ylabel)
    plt.show()

data = numpy.genfromtxt(sys.stdin, invalid_raise=False)
n = data.shape[0]
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
print "Shape of x and y data: %s %s"%(x.shape, y.shape)
do_plot(x, y, z, e)
