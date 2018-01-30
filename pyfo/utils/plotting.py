#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:14
Date created:  09/01/2018

License: MIT
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:41
Date created:  06/09/2017

License: MIT
'''
import numpy as np
import pandas as pd
import sys
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# from pandas.plotting import autocorrelation_plot
# from statsmodels.graphics import tsaplots
import platform

# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 8,               # LaTeX default is 10pt font.
#     "font.size": 8,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 5,
#     "ytick.labelsize": 5,
#     "figure.figsize": [4,4],     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)
class Plotting():

    def __init__(self, dataframe_samples, keys,lag, burn_in=False):
        self.samples = dataframe_samples[keys]
        self.lag = lag
        self.burn_in = burn_in
        self.keys = keys
        self.PATH  = sys.path[0]
        os.makedirs(self.PATH, exist_ok=True)
        self.PATH_fig = os.path.join(self.PATH, 'figures')
        os.makedirs(self.PATH_fig, exist_ok=True)
        self.PATH_data =  os.path.join(self.PATH, 'data')
        os.makedirs(self.PATH_data, exist_ok=True)
    
        # self.colors = cycle([ "blue", "green","black", "maroon", "navy", "olive", "purple", "red", "teal"])

    def plot_trace(self, all_on_one=True):
        '''
        Plots the traces for all parameters on one plot, if all_on_one flag is true
        Else, plots the trace of each parameter.
        :param all_on_one type: bool
        :return:
        '''
        if all_on_one:
            fname1 = 'trace_of_parameters.pdf'
            # fname2 = 'trace_of_parameters_wo_burnin.pdf'
            # self.samples.plot(subplots=True, figsize=(6,6))
            # plt.savefig(os.path.join(self.PATH_fig, fname1))
            # plt.clf()
            self.samples.plot(subplots=True, figsize=(6,6))
            plt.savefig(os.path.join(self.PATH_fig,fname1))
            path_image1 = self.PATH_fig + '/' + fname1
            # path_image2= self.PATH_fig + '/' + fname2
            print(50 * '=')
            print('Saving trace of all samples with burnin {0}'.format(path_image1))
            # print('Saving trace of all samples to {0} \n and with burnin to {1}'.format(path_image1,path_image2))
            print(50 * '=')

    def plot_density(self, all_on_one=True):
        """
        Plots either all the histograms for each param on one plot, or does it indiviually
        dependent on keys
        :param all_on_one type: bool
        :param keys type:list
        :return:
        """

        if all_on_one:
            fname = 'density_plot_of_parameters.pdf'
            path_image = self.PATH_fig + '/' + fname
            self.samples.plot(subplots=True, kind='kde')
            plt.savefig(os.path.join(self.PATH_fig,fname))
            print(50 * '=')
            print('Saving desnity of samples w/o burnin plot to {0}'.format(path_image))
            print(50 * '=')

    def auto_corr(self):
        """
        Plots for each parameter the autocorrelation of the samples for a specified lag.
        :return:
        """
        x = {}
        keys = copy.copy(self.keys)
        fig, axes = plt.subplots(ncols=len(self.keys), sharex=True, sharey=True)
        fig.text(0.5, 0.04, 'lag', ha='center', va='center')
        fig.text(0.02, 0.5, 'autocorrelation', ha='center', va='center', rotation='vertical')
        # fig.suptitle('Autocorrelation')
        for key in self.keys:
            x[key] = []
            for i in range(self.lag):
                x[key].append(self.samples[key].autocorr(lag=i))
        if len(self.keys) == 1:
            key = keys.pop()
            axes.stem(np.arange(0, len(x[key]), step=1), x[key], linestyle='None', markerfmt='.', markersize=0.2,basefmt="None", label=key)
            axes.legend(loc='upper right')
        else:
            for axis in axes.ravel():
                # https: // stackoverflow.com / questions / 4700614 / how - to - put - the - legend - out - of - the - plot
                key = keys.pop()
                axis.stem(np.arange(0, len(x[key]), step=1), x[key], linestyle='None', markerfmt='.', markersize=0.2,basefmt="None", label=key)
                axis.legend(loc='upper right')

        fname2 = 'Autocorrelationplot.pdf'
        plt.savefig(os.path.join(self.PATH_fig, fname2), dpi=400)
        path_image = self.PATH_fig +'/' +fname2
        print(50 * '=')
        print('Saving  autocorrelation plots to: {0}'.format(path_image))
        print(50 * '=')
