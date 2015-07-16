#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 07 16
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys
import numpy
import matplotlib.pyplot as plt
from optparse import OptionParser
from optparse import OptionGroup

parser = OptionParser()
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
parser.add_option_group(histogram_options)
(options, args) = parser.parse_args()

def do_plot(x, y, histogram=options.histogram, n_bins=options.n_bins, xmin=options.xmin,
            xmax=options.xmax):
    if not histogram:
        plt.plot(x,y)
        plt.grid()
        plt.show()
    else:
        if xmin is None:
            xmin = min(y)
        if xmax is None:
            xmax = max(y)
        plt.hist(y, bins=n_bins, range=(xmin,xmax), histtype=options.histtype)
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
