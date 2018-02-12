"""
Deterministic Ridge Leverage Score Sampling, Copyright (C) 2018,  Shannon McCurdy and Regents of the University of California.

Contact:  Shannon McCurdy, smccurdy@berkeley.edu.

    This file is part of Deterministic Ridge Leverage Score Sampling.

    Deterministic Ridge Leverage Score Sampling is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Deterministic Ridge Leverage Score Sampling is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Deterministic Ridge Leverage Score Sampling program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.stats
import pandas as pd
import os
import pickle

import matplotlib
#workaround for x - windows
matplotlib.use('Agg')
#from ggplot import *
import pylab as pl
import matplotlib.pyplot as plt
matplotlib.rc('axes.formatter', useoffset=False)

if matplotlib.__version__[0] != '1':
    matplotlib.style.use('classic')

def det_ridge_leverage(A, k, epsilon, plot, plot_loc):
    """
    for the data matrix A=U \Sigma V^T
    A.shape =(n=number of samples, d=number of features)
    V.shape=(d, n)
    k is the rank of the PCA leverage score
    epsilon is the error parameter.
    the function returns
    theta: the number of kept columns
    index_keep: the index of the selected columns
    tau_sorted:  the sorted ridge leverage scores of all the columns
    index_drop:  the index of the dropped columns
    """
    AAt = A.dot(A.T)
    U, eig, Ut = scipy.linalg.svd(AAt)
    if plot:
        plot_eigenvalues(eig, plot_loc)
        plot_svd(U, eig, k, plot_loc)
    print(AAt.shape, eig[k:].shape)
    AnotkF2 = np.sum(eig[k:])
    ridge_kernel = U.dot( np.diag(1/(eig + AnotkF2/k))).dot(Ut)
    #ridge_kernel = scipy.linalg.inv(AAt + AnotkF2/k *np.diag(np.ones(AAt.shape[0])))
    tau = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        tau[i] = A.iloc[:, i][:, None].T.dot(ridge_kernel).dot(A.iloc[:, i])
    tau = pd.DataFrame(tau)
    #print(tau)
    tau_tot = np.sum(tau)
    print(2 * k, tau_tot)
    tau_sorted = tau.sort_values(0,ascending=False, inplace=False)
    tau_sorted_sum= np.cumsum(tau_sorted)
    theta = (len(tau_sorted) - np.sum(tau_sorted_sum> tau_tot[0] - epsilon)+1)[0]
    if theta < k:
        theta = k
    index_keep = tau_sorted.index[0:theta]
    index_keep = index_keep.values
    index_drop = tau_sorted.index[theta:]
    index_drop = index_drop.values
    if plot:
        plot_tau(tau_sorted, theta, k, plot_loc)
            #powerlaw plot
        #pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted[0:1000])), tau_sorted[0:1000].values[:,0], k, plot_loc)
        pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted)), tau_sorted.values[:,0],1000 , theta, k, plot_loc)
        plot_columns_error(tau_sorted, pl_index, k, plot_loc)  
        plot_random_orthog_proj(A, index_keep, epsilon, k, plot_loc)
        eigenvalues_both(A, index_keep, eig, epsilon, plot_loc)
    return(theta, index_keep, tau_sorted, index_drop, tau_tot, AnotkF2)


def det_ksub_leverage(A, k, epsilon, plot, plot_loc):
    """
    """
    AAt = A.dot(A.T)
    U, eig, Ut = scipy.linalg.svd(AAt)
    print(AAt.shape)
    kernel = U[:, 0:k].dot( np.diag(1/(eig[0:k]))).dot(U[:, 0:k].T)
    #ridge_kernel = scipy.linalg.inv(AAt[] + AnotkF2/k *np.diag(np.ones(AAt.shape[0])))
    tau = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        tau[i] = A.iloc[:, i][:, None].T.dot(kernel).dot(A.iloc[:, i])
    #U, sing, Vt = scipy.sparse.linalg.svds(A)
    tau = pd.DataFrame(tau)
    #print(tau)
    tau_tot = np.sum(tau)
    print(k, tau_tot)
    tau_sorted = tau.sort_values(0,ascending=False, inplace=False)
    tau_sorted_sum= np.cumsum(tau_sorted)
    theta = (len(tau_sorted) - np.sum(tau_sorted_sum> tau_tot[0] - epsilon)+1)[0]
    if theta < k:
        theta = k
    index_keep = tau_sorted.index[0:theta]
    index_keep = index_keep.values
    index_drop = tau_sorted.index[theta:]
    index_drop = index_drop.values
    if plot:
        plot_tau(tau_sorted, theta, k , plot_loc)
        pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted)), tau_sorted.values[:,0],1000 , theta, k, plot_loc)
        plot_columns_error(tau_sorted, pl_index, k, plot_loc)  
    return(theta, index_keep, tau_sorted, index_drop, tau_tot)



def plot_columns_error(tau_sorted, pl_index, k, plot_loc):
    errors = np.cumsum(tau_sorted.values[::-1])
    pl_column_pred=np.zeros(errors.shape)
    for i in range(errors.shape[0]):
        pl_column_pred[i]= predict_n_columns(pl_index, k, errors[i])
    kept_columns=np.array(range(0, errors.shape[0])[::-1])
    fig, ax = plt.subplots()
    ax.scatter(errors, kept_columns,c='gray',s=36,edgecolors='gray',
                    lw = 0.5, label='columns')
    ax.set_ylim((-0.05, errors.shape[0]))
    ax.set_xlim((-0.05, np.max(errors)))
    ax.set_xlabel('Error, ' +r'$\tilde{\epsilon}$')
    ax.set_ylabel('Number of columns kept')
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_error_columns_' +str(k) +'.pdf')
    plt.close()


def plot_tau(tau_sorted, theta, k, plot_loc):
    """ 
    """
    fig, ax = plt.subplots()
    ax.loglog(range(len(tau_sorted)), tau_sorted, color='black', ls='solid', label='Data')
    #ax.scatter(range(len(tau_sorted)),tau_sorted,c='red',s=36,edgecolors='gray',
    #                lw = 0.5, label='Ridge leverage scores')
    ax.axvline(x=theta, color='grey', ls='dashed', label=r'$\theta$')
    ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1))
    ax.set_xlim((-10, len(tau_sorted)))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlabel('Sorted column index')
    ax.set_ylabel('Ridge leverage score')
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_ridge_leverage_score_' +str(k) +'.pdf')
    plt.close()

def plot_eigenvalues(eig, plot_loc):
    """ 
    """       
    fig, ax = plt.subplots()
    im=ax.plot(range(0, len(eig)), sorted(eig, reverse=True), color='gray', marker = 'o', ls ='solid' ,
                    lw = 0.5) #, label=r'Eigenvalues of A')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_yscale('log')
    ax.set_xlim((-1, len(eig)+1))
    fig.tight_layout()
    plt.savefig(plot_loc + 'plot_eigenvalues.pdf')
    plt.close()

def powerlaw(x, amp, index):
    return amp * (x**index)

def powerlaw_fit(xdata, ydata, nfit, theta, k, plot_loc):
    """
    http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
    Fitting the data -- Least Squares Method
    #########
    Power-law fitting is best done by first converting
    to a linear equation and then fitting to a straight line.
    y = a * x^b
    log(y) = log(a) + b*log(x)
    modified by SRM
    """
    logx = np.log10(xdata[0:nfit])
    logy = np.log10(ydata[0:nfit])
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    pinit = [1.0, -1.0]
    print(logx.shape, logy.shape)
    out = scipy.optimize.leastsq(errfunc, pinit,
                           args=(logx, logy), full_output=1)
    pfinal = out[0]
    covar = out[1]
    print( "pfinal", pfinal)
    print( "covar", covar)
    index = pfinal[1]
    amp = 10.0**pfinal[0]
    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp
    ##########
    # Plotting data
    ##########
    fig, ax = plt.subplots()
    ax.loglog(xdata, powerlaw(xdata, amp, index), color='gray', ls='dashed', lw=3, label='Fit')
    ax.plot(xdata, ydata, color='black', ls='solid', lw=3, label='Data')  # Data
    ax.annotate('b = %5.2f +/- %5.2f' % (amp, ampErr),  xy=(.30, .30), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='bottom', fontsize=20)
    ax.annotate('a = %5.2f +/- %5.2f' % (index, indexErr),  xy=(.30, .35), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='bottom', fontsize=20)
    ax.axvline(x=theta, color='black', ls='-.', lw=3, label=r'$\theta$')
    ax.set_xlabel('Sorted column index')
    ax.set_ylabel('Ridge leverage score')
    ax.set_xlim([0, len(xdata)])
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
    fig.tight_layout()
    #plt.xlim(1.0, 11)
    fig.savefig(plot_loc + 'plot_powerlaw_' + str(k) + '.pdf')
    plt.close()
    return(amp, index)


def haar_measure(n, seed):
    """
    A Random matrix distributed with Haar measure
    How to generate random matrices from the classical compact groups
    Francesco Mezzadri
    from scipy import *
    """
    z = scipy.stats.norm.rvs(loc=0, scale=1, size=(n, n), random_state=seed)
    q, r = scipy.linalg.qr(z)
    d = scipy.diagonal(r)
    ph = d/scipy.absolute(d)
    q = scipy.multiply(q, ph, q)
    return q

def plot_random_orthog_proj(A, index_keep, epsilon, k, plot_loc):
    nrep=1000
    ratio=np.zeros(nrep)
    C = A.iloc[:, index_keep]
    for i in range(nrep):
        orthog=haar_measure(A.shape[0], i)
        rop= orthog[:, 0:k].dot(orthog[:, 0:k].T)
        denom = A - rop.dot(A)
        num= C - rop.dot(C)
        ratio[i]= np.sum(np.diag(num.dot(num.T)))/np.sum(np.diag(denom.dot(denom.T)))
    num_bins = 100
    plt.clf()
    fig, ax = plt.subplots()
    ax.hist(ratio, num_bins, facecolor='grey', alpha=0.5)
    ax.set_xlabel(r'$||(I-X)C||_F^2/||(I-X)A||_F^2$')
    ax.set_ylabel('Number')
    plt.savefig(plot_loc + 'plot_random_orthog_proj_histogram' + str(k) + '.pdf')
    plt.close()

def plot_comparison(tau_sorted, tau_sorted_k):
    d=len(tau_sorted)
    fig, ax = plt.subplots()
    ax.scatter(tau_sorted, tau_sorted_k.loc[tau_sorted.index],c='gray',s=36,edgecolors='gray',
                    lw = 0.5)
    ax.set_xlabel('Ridge leverage score')
    ax.set_ylabel('Classical leverage score')
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlim((-0.05, 1.05))
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_score_comparison_' +str(k) +'.pdf')
    plt.close()

plot_comparison(tau_sorted, tau_sorted_cl)



def get_rnaseq_data_more(setup_dir):
    if not os.path.exists(setup_dir + '/data'):
        os.system('mkdir data')
    if not os.path.exists(setup_dir + '/data/tcga_LGG_RNASeq2_530.csv'):
        disease_type='LGG'
        rscript_path = os.getenv('RSCRIPT_PATH', '/usr/bin/Rscript')
        os.system(rscript_path + ' download_rnaseq2.R '+ disease_type + ' ' + setup_dir + '/data/')
    x=pd.read_csv(setup_dir + '/data/tcga_LGG_RNASeq2_530.csv', sep = ',', index_col=0)
    x=x.T
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x.iloc[inds]=np.take(col_mean, inds[1])
    x= x.loc[:, np.sum(x, axis=0)!=0]
    return(x)


if __name__ == "__main__":
    k = 3
    epsilon = 0.1
    plot = True
    plot_loc = 'LGG_RNASeq2_530_'
    setup_dir= os.getcwd()
    A= get_rnaseq_data_more(setup_dir)
    theta, index_keep, tau_sorted, index_drop, tau_tot, AnotkF2 = det_ridge_leverage(A, k, epsilon, plot, plot_loc)
    print('ridge theta', theta)
    theta_cl, index_keep_cl, tau_sorted_cl, index_drop_cl, tau_tot_cl = det_ksub_leverage(A,A.shape[0],  epsilon, plot, plot_loc + '_classical_')
    plot_comparison(tau_sorted, tau_sorted_cl)