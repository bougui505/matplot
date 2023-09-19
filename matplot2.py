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
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import socket


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def read_data(fields):
    inp = np.genfromtxt(sys.stdin, dtype=str)
    print(f"{inp.ndim=}")
    print(f"{len(inp)=}")
    data = dict()
    if inp.ndim == 1:
        data["x"] = np.arange(len(inp))
        data["y"] = inp
    return data


def add_metadata(filename, datastr, key="data"):
    """
    Add metadata to a png file
    """
    metadata = PngInfo()
    metadata.add_text(key, datastr, zip=True)
    metadata.add_text("cwd", os.getcwd())
    metadata.add_text("hostname", socket.gethostname())
    # if options.subsample is not None:
    #     print("# Adding subsampling metadata")
    #     metadata.add_text("subsampling", "1st-column")
    # if options.labels is not None:
    #     metadata.add_text("labels", options.labels)
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


def plot(data):
    """
    Simple plot
    """
    assert "x" in data
    assert "y" in data
    x = data["x"].astype(float)
    y = data["y"].astype(float)
    plt.plot(x, y)


def scatter(data):
    """
    Simple plot
    """
    assert "x" in data
    assert "y" in data
    x = data["x"].astype(float)
    y = data["y"].astype(float)
    plt.scatter(x, y)


def get_datastr(data):
    n = len(data["x"])
    keys = data.keys()
    outstr = ""
    for k in keys:
        outstr += f"#{k} "
    outstr += "\n"
    for i in range(n):
        for k in keys:
            outstr += f"{data[k][i]} "
        outstr += "\n"
    return outstr


def save(outfilename, datastr):
    plt.savefig(outfilename)
    ext = os.path.splitext(outfilename)[1]
    print(f"{ext=}")
    if ext == ".png":
        add_metadata(outfilename, datastr)


if __name__ == "__main__":
    import doctest
    import argparse

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
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "-f",
        "--fields",
        help="Fields for the data; e.g. 'xyxy'.\
        By default the first column is for x data and the other for y data.\
        If a 'z' field is given, this field is used to color the scatter dots.\
        If a 'e' field is given it is used to plot the error.\
        If a 'l' field is given this are the labels for the xticks -- xticklabels --.\
        If a 'w' field is given it is used as weights for weighted hostogram plotting (see: -H).\
        If a 't' field is given plot the given text at the given position (with x and y fields) using matplotlib.pyplot.text.\
        If a 'm' field is given, use it as markers (see: https://matplotlib.org/stable/api/markers_api.html).\
        If a 's' field is given, use it as a list of sizes for the markers.\
        If a 'c' field is given, use it as a list of colors (text color: r, g, b, y, ...) for the markers.\
        If --fields='*' is given all the columns are considered as y values.",
        default="xy",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Scatter plot of the (x,y) data",
    )
    parser.add_argument(
        "--xlabel", dest="xlabel", default=None, type=str, help="x axis label"
    )
    parser.add_argument(
        "--ylabel", dest="ylabel", default=None, type=str, help="y axis label"
    )
    parser.add_argument("--save", help="Save the file", type=str)
    parser.add_argument(
        "--read_data",
        help="Read plot data from the given png saved image using the --save option",
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()

    if args.read_data is not None:
        DATASTR = read_metadata(args.read_data)
        print(DATASTR)
    if (
        not sys.stdin.isatty()
    ):  # stdin is not empty (see: https://stackoverflow.com/a/17735803/1679629)
        DATA = read_data(args.fields)
        DATASTR = get_datastr(DATA)
        if args.scatter:
            scatter(DATA)
        else:
            plot(DATA)

        if args.xlabel is not None:
            plt.xlabel(args.xlabel)
        if args.ylabel is not None:
            plt.ylabel(args.ylabel)
        if args.save is None:
            plt.show()
        else:
            save(args.save, DATASTR)
