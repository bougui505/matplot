#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2021-03-04 15:53:05 (UTC+0100)

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Read CSV file from stdin and draw a pairplot
    """
    data = pd.read_csv(sys.stdin)
    g = sns.PairGrid(data)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    plt.show()
