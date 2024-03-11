#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################
import os

import numpy as np


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def ROC(positives, negatives):
    """
    TPR as a function of FPR
    TPR = TP/P
    FPR = FP/N

    >>> np.random.seed(0)
    >>> positives = np.random.uniform(size=10)
    >>> negatives = np.random.uniform(size=15)
    >>> x, y, auc = ROC(positives, negatives)
    >>> auc
    0.52
    >>> x
    [0.0, 0.06666666666666667, 0.13333333333333333, 0.2, 0.26666666666666666, 0.26666666666666666, 0.26666666666666666, 0.26666666666666666, 0.3333333333333333, 0.4, 0.4, 0.4, 0.4666666666666667, 0.4666666666666667, 0.4666666666666667, 0.4666666666666667, 0.5333333333333333, 0.6, 0.6666666666666666, 0.7333333333333333, 0.8, 0.8666666666666667, 0.8666666666666667, 0.9333333333333333, 0.9333333333333333, 1.0]
    >>> y
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0]
    """
    positives = np.asarray(positives)
    negatives = np.asarray(negatives)
    positives = positives[~np.isnan(positives)]
    negatives = negatives[~np.isnan(negatives)]
    P = len(positives)
    N = len(negatives)
    assert positives.ndim == 1
    assert negatives.ndim == 1
    alldata = np.concatenate((positives, negatives))
    sorter = alldata.argsort()
    positive_labels = np.ones(P, dtype=bool)
    negative_labels = np.zeros(N, dtype=bool)
    all_labels = np.concatenate((positive_labels, negative_labels))
    all_labels = all_labels[sorter]
    x, y = [0.], [0.]
    thresholds=[np.nan]
    auc = 0.
    # pROC: compute the pROC as explained in https://link.springer.com/article/10.1007/s10822-008-9181-z#Sec2
    pROC_auc = 0.
    TP = 0
    FP = 0
    for label, threshold in zip(all_labels, alldata[sorter]):
        if label:
            # positive
            TP += 1
        else:
            # negative
            FP += 1
        if P > 0:
            TPR = TP / P
        else:
            TPR = 0.
        if N > 0:
            FPR = FP / N
        else:
            FPR = 0.
        x.append(FPR)
        y.append(TPR)
        thresholds.append(threshold)
        auc += (x[-1] - x[-2]) * y[-1]
        if x[-1] > 0 and x[-2] > 0:
            pROC_auc += -np.log10(x[-2] / x[-1]) * y[-1]
    return x, y, auc, pROC_auc, thresholds


def print_roc(x, y, thresholds, auc, pROC_auc):
    # print(f'#AUC: {auc:.2f}')
    # print(f'#pROC_AUC: {pROC_auc:.2f}')
    # print('#FPR #TPR')
    # for xval, yval in zip(x, y):
    #     print(f'{xval:.2f} {yval:.2f}')
    print(f"{auc=}")
    print(f"{pROC_auc=}")
    print('--')
    for xval, yval, threshold in zip(x, y, thresholds):
        print(f'fpr={xval}')
        print(f'tpr={yval}')
        print(f'threshold={threshold}')
        print('--')


if __name__ == '__main__':
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(
        description=
        'Compute a ROC curve from the given data. Two columns separated by a comma must be given in stdin (using a pipe). The first column gives the negative values, the second the positive values.'
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--test_random', help='Test with random data for positives and negatives', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.test_random:
        # positives = np.random.uniform(size=500)
        # negatives = np.random.uniform(size=49900)
        positives = np.random.normal(size=50000, loc=1)
        negatives = np.random.normal(size=49900, loc=2)
        x, y, auc, pROC_auc = ROC(positives, negatives)
        print_roc(x, y, auc, pROC_auc)
        sys.exit()
    data = np.genfromtxt(sys.stdin, invalid_raise=False, delimiter=',')
    negatives = data[:, 0]
    positives = data[:, 1]
    x, y, auc, pROC_auc, thresholds = ROC(positives, negatives)
    print_roc(x, y, thresholds, auc, pROC_auc)
