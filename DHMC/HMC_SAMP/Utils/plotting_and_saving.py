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
import os
from itertools import cycle
from torch.autograd import Variable
from statsmodels.graphics import tsaplots
from matplotlib import pyplot as plt

import seaborn as sns
import matplotlib as mpl
plotparams= {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "axes.linewidth":2,
    "axes.grid": False,
    "patch.force_edgecolor"  : True,
    "legend.framealpha"    : None,
    "font.size": 8,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": [3.2,3.2],     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
seaborn_plot = {'axes.axisbelow': True,
 'axes.edgecolor': '0.3',
 'axes.facecolor': 'white',
 'axes.grid': False,
 'axes.labelsize': 8,
 'axes.linewidth': 2.0,
 'figure.facecolor': 'white',
 'font.family': 'serif',
 'font.sans-serif': [u'Arial',
  u'DejaVu Sans',
  u'Liberation Sans',
  u'Bitstream Vera Sans',
  u'sans-serif'],
  'legend.frameon': True}
sns.set(rc={"figure.figsize": (3.2,3.2)})
sns.set_style('white',seaborn_plot)

class Plotting():

    def __init__(self, samples,samples_with_burnin, mean, model = None, cov= None, hmcflag = True):
        if isinstance(samples, Variable):
            self.samples = samples.data.numpy()
        else:
            self.samples = samples.numpy()
        if isinstance(samples_with_burnin, Variable):
            self.samples_with_burnin = samples_with_burnin.data.numpy()
        else:
            self.samples_with_burnin  = samples_with_burnin.numpy()
        self.mean    = mean
        if cov is not None:
            self.cov = cov
        if hmcflag:
            string = 'hmc/'
        else:
            string = 'dhmc/'
        if model is not None:
            # model      = model + 'small'
            model = model
            self.PATH  = '/Users/bradley/Documents/Aims_work/Miniproject2/Project_notes/MCMC/ADHMC/ad-hmc/paper_AISTATS/plots/' + string + model
            os.makedirs(self.PATH, exist_ok=True)
            self.PATH_fig = os.path.join(self.PATH, 'figures')
            os.makedirs(self.PATH_fig, exist_ok=True)
            self.PATH_data =  os.path.join(self.PATH, 'data')
            os.makedirs(self.PATH_data, exist_ok=True)
        else:
            self.PATH = '/Users/bradley/Documents/Aims_work/Miniproject2/Project_notes/MCMC/ADHMC/ad-hmc/paper_AISTATS/plots/'
        self.colors = cycle([ "blue", "green","black", "maroon", "navy", "olive", "purple", "red", "teal"])
    def plot_trace(self):
        '''

        :param samples:  an nparray
        :param parameters:  Is a list of which parameters to take the traces of
        :return:
        '''
        print('Saving trace plots.....')
        fig, ax = plt.subplots()
        iter = np.arange(0, np.shape(self.samples)[0])
        for i in range(np.shape(self.samples)[1]):
            ax.plot(iter, self.samples[:,i], label='Parameter {0} '.format(i))
            ax.set_title('Trace plot for the parameters')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Sample Values')
            plt.legend()
            fname = 'trace.pgf'
            fname2 = 'trace.pdf'
            fig.savefig(os.path.join(self.PATH_fig, fname), dpi=400)
            fig.savefig(os.path.join(self.PATH_fig, fname2))

    def histogram(self):
        print('Saving histogram.....')
        weights = np.ones_like(self.samples) / float(len(self.samples))
        fig, ax = plt.subplots()
        if np.shape(self.samples)[1] > 1:
            for i in range(np.shape(self.samples_with_burnin)[1]):
                ax.hist(self.samples[:,i],  bins = 'auto', normed=1, label= r'$\mu_{\mathrm{i}}$' + '=' + '{0}'.format(
                        self.mean.data[0][i]))
                ax.set_title('Histogram of samples ')
                ax.set_xlabel(' Samples ')
                ax.set_ylabel('Density')
            plt.legend()
            fname = 'histogram.pgf'
            fname2 = 'histogram.pdf'
            # Ensures directory for this figure exists for model, if not creates it
            # fig.savefig(os.path.join(self.PATH_fig, fname))
            fig.savefig(os.path.join(self.PATH_fig,fname2), dpi =400)


        else:
            ax.hist(self.samples, bins='auto', normed=1, label= r'$\mu_{\mathrm{0}}$' + '=' + '{0}'.format(self.mean.data[0][0]))
            ax.set_title(
                'Histogram of samples')
            ax.set_xlabel(' Samples ')
            ax.set_ylabel('Density')
        # plt.axis([40, 160, 0, 0.03])
            plt.legend()
        # Ensures directory for this figure exists for model, if not creates it
            fig.savefig(os.path.join(self.PATH_fig, 'histogram.pdf'), dpi =400)
        # plt.show()
    def auto_corr(self):
        print('Plotting autocorrelation.....')
        fig, ax = plt.subplots(nrows = 1, ncols = np.shape(self.samples)[1], sharex= True, sharey= True, squeeze=False)
        #squeeze ensures that we can use size(1) object, and size(n,n) objects.
        # sub plots spits back on figure object and 1 X np.shape(self.samples)[1] axis objects, stored in ax.
        i = 0

        def label(ax, string):
            ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                        size=14, xycoords='axes fraction', textcoords='offset points')
        lag = 50
        for row in ax:
            for col in row:
                tsaplots.plot_acf(self.samples[:,i], ax=col, title= '',lags= lag,alpha=0.05,use_vlines=True)
                # alpha sets 95% confidence interval, lag - is autocorrelation lag
                # label(col, '  ')
                i = i + 1

        plt.xlabel("")
        plt.ylabel('')
        plt.suptitle('Autocorrelation for lag {}'.format(lag))
        fname = 'Autocorrelationplot.pgf'
        plt.savefig(os.path.join(self.PATH_fig, fname), dpi=400)
        fname2 = 'Autocorrelationplot.pdf'
        plt.savefig(os.path.join(self.PATH_fig, fname2), dpi=400)

        # print('Plotting autocorrelation......')
        # for i in range(self.samples.shape[1]):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     x = self.samples[:,i].flatten()
        #     x = x - x.mean()
        #
        #     autocorr = np.correlate(x, x, mode='full')
        #     autocorr = autocorr[x.size:]
        #     autocorr /= autocorr.max()
        #     markerline, stemline, sline = ax.stem(autocorr, label = 'Parameter ' + str(i))
        #     plt.setp(stemline,color= next(self.colors),linewidth= 0.2)
        #     plt.setp(markerline, markerfacecolor = next(self.colors), markersize =0.3)
        #     plt.setp(sline, linewidth = 0.2)
        #     ax.set_title(' Autocorrelation plot')
        #     ax.set_ylabel('Autocorrelation')
        #     ax.set_xlabel('Samples')
        #     ax.legend(loc="best")
        #     fname = 'Autocorrelation plot_' +'parameter_' +str(i)+ '.png'
        #     fig.savefig(os.path.join(self.PATH_fig, fname), dpi=400)
        #     plt.clf()
    def save_data(self):
        print('Saving data....')
        df1 = pd.DataFrame(self.samples)
        df2 = pd.DataFrame(self.samples_with_burnin)
        # Ensures directory for this data exists for model, if not creates it
        path1 =  'samples_after_burnin.csv'
        path2 =  'samples_with_burnin.csv'
        df1.to_csv(os.path.join(self.PATH_data,path1))
        df2.to_csv(os.path.join(self.PATH_data,path2))

    def call_all_methods(self):
        self.plot_trace()
        self.histogram()
        self.auto_corr()
        self.save_data()