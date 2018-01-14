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
from itertools import cycle
import copy
from torch.autograd import Variable
# # mpl.use('pgf')
# import matplotlib as mpl
from matplotlib import pyplot as plt
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

    def __init__(self, dataframe_samples,dataframe_samples_woburin, keys,lag, burn_in=False):
        self.samples = dataframe_samples
        self.samples_withbin = dataframe_samples_woburin
        self.keys = keys
        self.lag = lag
        self.burn_in = burn_in
        self.PATH  = sys.path[0]
        os.makedirs(self.PATH, exist_ok=True)
        self.PATH_fig = os.path.join(self.PATH, 'figures')
        os.makedirs(self.PATH_fig, exist_ok=True)
        self.PATH_data =  os.path.join(self.PATH, 'data')
        os.makedirs(self.PATH_data, exist_ok=True)
    
        self.colors = cycle([ "blue", "green","black", "maroon", "navy", "olive", "purple", "red", "teal"])

    def plot_trace_histogram(self, all_on_one=False):
        '''
        Plots the traces all on one histogram, if flag set to true.
        Else, plots the trace of each parameter and the corresponding histogram in the cloumn adjacent to it.
        :param all_on_one type: bool
        :return:
        '''

        print('Saving trace plots.....')
        fig, axes = plt.subplots(nrows=2, ncols=len(self.keys))
        key = copy.copy(self.keys) # stops keys from been deleted from self.keys

        if self.burn_in:
            print('Burn in plots')

        for axis in axes.ravel():
            # https: // stackoverflow.com / questions / 4700614 / how - to - put - the - legend - out - of - the - plot
            key = key.pop()
            axis.plot(list(range(0, len(self.samples[key]))), self.samples[key],label=key)
            axis.legend(loc='upper right')
            # Here at 12/01/18 15:08 to finish

        for key in self.keys:
            ax.plot(iter, self.samples[key], label='{0} '.format(key))
            ax.set_title('Trace plot for the parameter')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Sampled values of the Parameter')
            plt.legend()
            fname2 = 'trace.pdf'
            fig.savefig(os.path.join(self.PATH_fig, fname2))

        weights = np.ones_like(self.samples) / float(len(self.samples))
        fig, ax = plt.subplots()
        if np.shape(self.samples)[1] > 1:
            for i in range(np.shape(self.samples_with_burnin)[1]):
                ax.hist(self.samples[:,i],  bins = 'auto', normed=1, label= r'$\mu_{\mathrm{emperical}}$' + '=' + '{0}'.format(
                        self.mean.data[0][i]))
                ax.set_title('Histogram of samples ')
                ax.set_xlabel(' Samples ')
                ax.set_ylabel('Density')
            plt.legend()
            fname2 = 'parameterplots.pdf'
                # Ensures directory for this figure exists for model, if not creates it
            fig.savefig(os.path.join(self.PATH_fig,fname2))
            path_image = self.PATH_fig + '/' + fname2
            print(50 * '=')
            print('Saving trace and histogram plot to {0}'.format(path_image))
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

    def save_data(self):
        # Ensures directory for this data exists for model, if not creates it
        path1 =  'samples_after_burnin.csv'
        path2 =  'samples_with_burnin.csv'
        self.samples.to_csv(os.path.join(self.PATH_data,path1))
        self.samples_withbin.to_csv(os.path.join(self.PATH_data,path2))
        print(50*'=')
        print('Saving data in: {0}'.format(self.PATH_data))
        print(50 * '=')
