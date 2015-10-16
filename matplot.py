#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 10 16
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib
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

def do_plot(x, y, histogram=options.histogram, n_bins=options.n_bins, xmin=options.xmin,
            xmax=options.xmax):
    if not histogram:
        plt.plot(x,y)
    else:
        if xmin is None:
            xmin = min(y)
        if xmax is None:
            xmax = max(y)
        if is_sklearn and options.gmm is not None:
            options.normed = True
        histo = plt.hist(y, bins=n_bins, range=(xmin,xmax), histtype=options.histtype, normed=options.normed)
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
    if options.xlabel is not None:
        plt.xlabel(options.xlabel)
    if options.ylabel is not None:
        plt.ylabel(options.ylabel)
    plt.grid()
    plt.show()

data = numpy.genfromtxt(sys.stdin)
n = data.shape[0]
if len(data.shape) == 1:
    x = range(n)
    y = data
else:
    x = data[:,0]
    y = data[:,1:]
do_plot(x,y)
