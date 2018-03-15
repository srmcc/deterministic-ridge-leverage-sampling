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
import glob
from time import time
import re

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
    tau_sorted:  the sorted leverage scores of all the columns
    index_drop:  the index of the dropped columns
    """
    AAt = A.dot(A.T)
    U, eig, Ut = scipy.linalg.svd(AAt)
    if plot:
        #if mean subtracted, change plot to drop the last eigenvalue.
        plot_eigenvalues(eig, plot_loc)
        #plot_svd(U, eig, k, plot_loc)
    print(AAt.shape, eig[k:].shape)
    AnotkF2 = np.sum(eig[k:])
    ridge_kernel = U.dot( np.diag(1/(eig + AnotkF2/k))).dot(Ut)
    #ridge_kernel = scipy.linalg.inv(AAt + AnotkF2/k *np.diag(np.ones(AAt.shape[0])))
    tau = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        tau[i] = A.iloc[:, i][:, None].T.dot(ridge_kernel).dot(A.iloc[:, i])
    #U, sing, Vt = scipy.sparse.linalg.svds(A)
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
        #plot_css_frob_2(A, tau_sorted, AnotkF2, theta, k, 10, plot_loc)
        plot_random_orthog_proj(A, index_keep, epsilon, k, plot_loc)
        eigenvalues_both(A, index_keep, eig, epsilon, plot_loc)
    return(theta, index_keep, tau_sorted, index_drop, tau_tot, AnotkF2)


def det_ksub_leverage(A, k, epsilon, plot, plot_loc):
    """
    for the data matrix A=U \Sigma V^T
    A.shape =(n=number of samples, d=number of features)
    V.shape=(d, n)
    epsilon is the error parameter.
    the function returns
    theta: the number of kept columns
    index_keep: the index of the selected columns
    tau_sorted:  the sorted leverage scores of all the columns
    index_drop:  the index of the dropped columns
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
            #powerlaw plot
        #pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted[0:1000])), tau_sorted[0:1000].values[:,0], k, plot_loc)
        pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted)), tau_sorted.values[:,0],1000 , theta, k, plot_loc)
        plot_columns_error(tau_sorted, pl_index, k, plot_loc)  
        #plot_css_frob_2(A, tau_sorted, AnotkF2, theta, k, 10, plot_loc)
        #plot_random_orthog_proj(A, index_keep, epsilon, k, plot_loc)
    return(theta, index_keep, tau_sorted, index_drop, tau_tot)

def random_ridge_leverage(A, k, epsilon=0.1, delta=0.1, without_replacement=True):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d, d>> n
    k is the rank of the projection with theoretical gaurantees.
    choose epsilon and delta to be less than one.
    """
    AAt = A.dot(A.T)
    U, eig, Ut = scipy.linalg.svd(AAt)
    if plot:
        plot_eigenvalues(eig, plot_loc)
        plot_svd(U, eig, k, plot_loc)
    print(AAt.shape, eig[k:].shape)
    AnotkF2 = np.sum(eig[k:])
    ridge_kernel = U.dot( np.diag(1/(eig + AnotkF2/k))).dot(Ut)
    for i in range(A.shape[1]):
        tau[i] = A.iloc[:, i][:, None].T.dot(ridge_kernel).dot(A.iloc[:, i])
    #U, sing, Vt = scipy.sparse.linalg.svds(A)
    tau = pd.DataFrame(tau)
    #print(tau)
    tau_tot = np.sum(tau)
    theta = np.int( 2 * k / epsilon**2 * np.log(16* k / delta))
    print('number of columns sampled', theta)
    p = tau/tau_tot
    if without_replacement:
    else:
        R = np.random.multinomial(theta , p)
        print('number of distinct columns sampled ', np.sum(R != 0))
        C = np.zeros((A.shape[0], theta))
        #D= np.zeros((A.shape[0], np.sum(R != 0)))
        counter=0
        #counterD=0
        for i in range(A.shape[1]):
            if R[i]!=0: 
                # D[:, counterD] = np.sqrt(R[i]) * A[:, i] /np.sqrt(t* p[i])
                # counterD=counterD+1
                for j in range(R[i]):
                    C[:, counter] =  A[:, i] /np.sqrt(theta* p[i])
                    counter = counter +1  
    return(theta, C, R)

def plot_columns_error(tau_sorted, pl_index, k, plot_loc):
    errors = np.cumsum(tau_sorted.values[::-1])
    pl_column_pred=np.zeros(errors.shape)
    for i in range(errors.shape[0]):
        pl_column_pred[i]= predict_n_columns(pl_index, k, errors[i])
    kept_columns=np.array(range(0, errors.shape[0])[::-1])
    fig, ax = plt.subplots()
    ax.scatter(errors, kept_columns,c='gray',s=36,edgecolors='gray',
                    lw = 0.5, label='columns')
    # ax.scatter(errors, pl_column_pred,c='blue',s=36,edgecolors='gray',
    #                 lw = 0.5, label='predicted')
    #ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1))
    ax.set_ylim((-0.05, errors.shape[0]))
    ax.set_xlim((-0.05, np.max(errors)))
    ax.set_xlabel('Error, ' +r'$\tilde{\epsilon}$')
    ax.set_ylabel('Number of columns kept')
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_error_columns_' +str(k) +'.pdf')
    plt.close()

def plot_css_frob_2(A, tau_sorted, AnotkF2, theta, k, step, plot_loc):
    #css_frob_2_ratio= np.zeros(tau_sorted.shape[0])
    css_frob_2_ratio=[]
    #for i in range(k, tau_sorted.shape[0], step):
    for i in range(k, A.shape[0], step):
        index_keep = tau_sorted.index[0:i]
        #print(css_projection(A, index_keep))
        #css_frob_2_ratio[i]= css_projection(A, index_keep)/AnotkF2
        css_frob_2_ratio.append(css_projection(A, index_keep)/AnotkF2)
    fig, ax = plt.subplots()
    # ax.scatter(np.array(range(k, tau_sorted.shape[0])[::-1]), css_frob_2_ratio[k:],c='red',s=36,edgecolors='gray',
    #                 lw = 0.5, label='CSS frobenius norm ratio')
    # ax.scatter(np.array(range(k, tau_sorted.shape[0], step)[::-1]), css_frob_2_ratio,c='red',s=36,edgecolors='gray',
    #                 lw = 0.5, label='CSS frobenius norm ratio')
    ax.scatter(np.array(range(k, A.shape[0], step)[::-1]), css_frob_2_ratio,c='red',s=36,edgecolors='gray',
                    lw = 0.5, label='CSS frobenius norm ratio')
    # ax.axvline(x=theta, color='black', ls='dashed', lw=1, label='theta')
    ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('Number of columns kept')
    ax.set_ylabel('CSS Frobenius norm ratio')
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_css_frob_norm_ratio_' +str(k) +'.pdf')
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

def plot_svd(A, U, eig, k, colorcat, plot_loc):
    U=pd.DataFrame(U, index=A.index)
    cma= plt.cm.get_cmap('viridis')
    vminn= np.min(range(colorcat.shape[0]))
    #vminn= np.min(range(colorcat.shape[0]))-1 #: needed for when missing is one of the rows.
    vmaxx= np.max(range(colorcat.shape[0]))
    colorlist=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vminn, vmax=vmaxx, clip=False), cmap=cma).to_rgba(range(colorcat.shape[0]))
    for z1 in range(k):
        for z2 in range(k+1):
            if z2< z1:
                print(z1, np.max(eig[z1] *U.ix[:, z1]))
                leg=[]
                names=[]
                fig, ax = plt.subplots()
                for l in range(colorcat.shape[0]):
                #for l in range(colorcat.shape[0]-1)#: needed for when missing is one of the rows.
                    lsamp= colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns
                    x = eig[z2]* U.ix[lsamp, z2]
                    y = eig[z1] * U.ix[lsamp, z1]
                    im=ax.scatter(x, y, color=colorlist[l], s=36, edgecolors='gray',
                        lw = 0.5, label='Samples')
                    leg.append(im)
                    names.append(colorcat.index[l])
                ax.legend(leg, names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax.set_xlabel(r'$V_%d $' % (z2 +1), fontsize=20)
                ax.set_ylabel(r'$V_%d$' %(z1 +1), fontsize=20)
                # ax.set_xlim((np.min(eig[z2]*U.ix[:, z2]), np.max(eig[z2]*U.ix[:, z2])))
                # ax.set_ylim((np.min(eig[z1] *U.ix[:, z1]), np.max(eig[z1] *U.ix[:, z1])))
                fig.tight_layout()
                plt.savefig(plot_loc + 'plot_svd'+str(z1)+'_'+ str(z2)+ '.pdf', bbox_inches='tight')
                plt.close()




def powerlaw(x, amp, index):
    return amp * (x**index)

def powerlaw_fit(xdata, ydata, nfit,theta, k, plot_loc):
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


def powerlaw_fit_hist(xdata, ydata, nfit, plot_loc):
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
    pinit = [1.0, -0.5]
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
    ax.set_xlabel(r'$(AA^T)_{ij}$')
    ax.set_ylabel('Number')
    #ax.set_xlim([0, len(xdata)])
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
    fig.tight_layout()
    #plt.xlim(1.0, 11)
    fig.savefig(plot_loc + 'plot_powerlaw_histogram.pdf')
    plt.close()
    return(amp, index)




def predict_n_columns(index, k, epsilon):
    eta = -index-1
    return np.max(((4*k/epsilon)**(1/(1+eta)), (4*k/epsilon/eta)**(1/eta), k))

def css_projection(A, index_keep):
    C = A.iloc[:, index_keep]
    CCt=C.dot(C.T)
    print('C', C.shape)
    Uc, eig_c, Uct = scipy.linalg.svd(CCt)
    l= np.linalg.matrix_rank(CCt)
    #print(eig_c)
    # p=Uc[:, 0:l].dot(Uc[:, 0:l].T).dot(A)
    # Up, eig_p, Upt = scipy.linalg.svd(p.dot(p.T))
    css = A - Uc[:, 0:l].dot(Uc[:, 0:l].T).dot(A)
    css2=css.dot(css.T)
    css_frob_2=np.sum(np.diag(css2))
    return(css_frob_2)

##bug?
def eigenvalues_both(A, index_keep, eig, epsilon, plot_loc):
    C = A.iloc[:, index_keep]
    CCt=C.dot(C.T)
    print('C', C.shape)
    Uc, eig_c, Uct = scipy.linalg.svd(CCt)
    fig, ax = plt.subplots()
    ax.plot(range(0, len(eig)), np.array(sorted(eig, reverse=True))-np.array(sorted(eig_c, reverse=True)), color='grey', marker = 'o', ls ='solid', lw=1, label='A-C')
    ax.plot(range(0, len(eig)), np.array(sorted(eig_c, reverse=True))- (1-2* epsilon) *np.array(sorted(eig, reverse=True)), color='black', marker = 'o', ls ='solid', lw=1, label='C- (1-2'+ r'$\epsilon$' +')A' )
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue difference')
    ax.set_yscale('log')
    ax.set_xlim((-1, len(eig)+1))
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
    fig.tight_layout()
    plt.savefig(plot_loc + 'plot_eigenvalues_both.pdf')
    plt.close()

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
    # ax.axvline(x=1, color='black', ls='-.', lw=1, label=r'$1$')
    # ax.axvline(x=1- epsilon, color='black', ls='-.', lw=1, label=r'$1-\epsilon$')
    # n, bins, patches = plt.hist(ratio, num_bins, facecolor='grey', alpha=0.5)
    # # plt.vline(x=1, color='black', ls='-.', lw=1, label='1')
    # # plt.vline(x=1- epsilon, color='black', ls='-.', lw=1, label='1 -epsilon')
    # plt.xlabel(r'$||(I-X)C||_F^2/||(I-X)A||_F^2$')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # plt.ylabel('Number')
    plt.savefig(plot_loc + 'plot_random_orthog_proj_histogram' + str(k) + '.pdf')
    plt.close()



def make_categorical(row):
    ## row should be (samples, )
    row = row.replace(np.nan, 'missing')
    items = list(np.unique(row))
    empty = np.zeros((len(items), row.shape[0]))
    for s, samples in enumerate(row.index):
        i = items.index(row.loc[samples])  
        empty[i, s]=1
    cat = pd.DataFrame(empty, index = items, columns = row.index)
    return(cat)



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


def bound_check(A, index_keep, k, epsilon):
    AAt = A.dot(A.T)
    U, eig, Ut = scipy.linalg.svd(AAt)
    print(AAt.shape, eig[k:].shape)
    AnotkF2 = np.sum(eig[k:])
    ridge_kernel_eigs =  1/(eig + AnotkF2/k)
    C = A.iloc[:, index_keep]
    CCt=C.dot(C.T)
    Uc, eig_c, Uct = scipy.linalg.svd(CCt)
    CnotkF2 = np.sum(eig_c[k:])
    ridge_kernel_eigs_c =  1/(eig_c + CnotkF2/k)
    alpha = 2 *(2+ np.sqrt(2))
    print('lowerbound eqn 7', np.sum((1-epsilon)* eig - epsilon*AnotkF2/k  <= eig_c ) )
    print('upperbound eqn 7', np.sum( eig_c <= eig))
    print('average eig ratio',  np.average(eig_c/eig) )
    print('lowerbound eqn 53', (1- alpha* epsilon)* AnotkF2 <= CnotkF2)
    print('upperbound eqn 53', CnotkF2 <= AnotkF2)
    print('frob ratio', CnotkF2/AnotkF2)
    print('lowerbound eqn 10', np.sum(ridge_kernel_eigs <= ridge_kernel_eigs_c))
    print('upperbound eqn 10', np.sum(ridge_kernel_eigs_c <= (1-(alpha +1 )* epsilon)**(-1)* ridge_kernel_eigs))
    print('ridge_kernel ratio', np.average( ridge_kernel_eigs_c/ ridge_kernel_eigs)  )


def do_ridge_reg(y, A, index_keep, k, AnotkF2, epsilon, plot_loc, plot=True):
    # this is the slow way.
    # big_kernel= scipy.linalg.inv( A.T.dot(A) + AnotkF2/k *np.diag(np.ones(A.shape[1])))
    # hatx_A= big_kernel.dot(A.T).dot(y)
    hatx_A=np.linalg.solve(A.T.dot(A) + AnotkF2/k *np.diag(np.ones(A.shape[1])), (A.T).dot(y)) 
    #print(np.allclose(hatx_A, hatx_A_alt))
    haty_A= A.dot(hatx_A)
    C = A.iloc[:, index_keep]
    CCt=C.dot(C.T)
    Uc, eig_c, Uct = scipy.linalg.svd(CCt)
    CnotkF2 = np.sum(eig_c[k:])
    print('lowerbound eqn 32', (1-epsilon)* AnotkF2 < CnotkF2)
    print('upperbound eqn 32', CnotkF2 < AnotkF2)
    # big_kernel_c= scipy.linalg.inv( C.T.dot(C) + CnotkF2/k *np.diag(np.ones(C.shape[1])))
    # hatx_C= big_kernel_c.dot(C.T).dot(y)
    hatx_C=np.linalg.solve(C.T.dot(C) + CnotkF2/k *np.diag(np.ones(C.shape[1])), (C.T).dot(y)) 
    hatx_C_big = np.zeros(hatx_A.shape)
    hatx_C_big[index_keep] = hatx_C
    haty_C= C.dot(hatx_C)
    print('nans?', np.sum(np.isnan(haty_C)))
    if plot: 
        fig, ax = plt.subplots()
        ax.scatter(haty_C, y,c='gray',s=36,edgecolors='gray',
                        lw = 0.5)
        ax.set_xlabel('Ridge regression prediction with column subset')
        ax.set_ylabel('Ridge regression outcome')
        fig.tight_layout()
        plt.savefig(plot_loc+ 'plot_ridge_regression_prediction_outcome_' +str(k) +'.pdf')
        plt.close()
        ##overlapping histogram plot
        num_bins = 20
        plt.clf()
        fig, ax = plt.subplots()
        for item in np.unique(y):
            ax.hist(haty_C[(y==item).values], num_bins, alpha=0.5, label='y = '+ str(item))
        ax.set_xlabel(r'$\hat{y}_C$')
        ax.set_ylabel('Number')
        ax.legend(loc='upper right')
        plt.savefig(plot_loc + 'plot_ridge_regression_yhatC_truth_histograms_' + str(k) + '.pdf')
        plt.close()
        num_bins = 20
        plt.clf()
        fig, ax = plt.subplots()
        ax.hist(haty_A -haty_C, num_bins, facecolor='grey', alpha=0.5)
        ax.set_xlabel(r'$\hat{y}_A - \hat{y}_C$')
        ax.set_ylabel('Number')
        plt.savefig(plot_loc + 'plot_ridge_regression_yhat_comparison_histogram_' + str(k) + '.pdf')
        plt.close()
        num_bins = 100
        plt.clf()
        fig, ax = plt.subplots()
        ax.hist(hatx_A -hatx_C_big, num_bins, facecolor='grey', alpha=0.5)
        ax.set_xlabel(r'$\hat{x}_A - \hat{x}_C$')
        ax.set_ylabel('Number')
        plt.savefig(plot_loc + 'plot_ridge_regression_coeff_comparison_histogram_' + str(k) + '.pdf')
        plt.close()
    return(haty_A, haty_C)


def do_ridge_reg_sim(A, index_keep, k, AnotkF2, epsilon, sigma2, plot_loc, nrep, seed, plot=True):
    xstar = scipy.stats.norm.rvs(loc=0, scale=1, size=(A.shape[1], 1), random_state=seed)
    ystar = A.dot(xstar)
    err_yC = 0
    err_yA = 0
    print('hello')
    for i in range(nrep):
        error = scipy.stats.norm.rvs(loc=0, scale=1, size=(A.shape[0], 1), random_state=seed+i)
        y = ystar + sigma2* error
        haty_A, haty_C= do_ridge_reg(y, A, index_keep, k, AnotkF2, epsilon, plot_loc, plot=False)
        err_yA = err_yA + (ystar - haty_A)**2
        err_yC = err_yC + (ystar - haty_C)**2
        print(err_yA.shape)
    riskA = np.sum(err_yA)/(nrep * A.shape[0])
    riskC = np.sum(err_yC)/(nrep * A.shape[0])
    alpha = 2 *(2+ np.sqrt(2))
    beta = 2 * alpha (-1 + 2* alpha + 3* alpha**2)/((1-alpha)**2)
    print('bound', riskC <(1-beta* epsilon)**(-2)* riskA )
    print('ratio', riskC/riskA)
    return(riskA, riskC)

def load_data(data_directory):
    #filenames = glob.glob(data_directory + '*279*')
    filenames = glob.glob(data_directory+ '*280*')
    data = pd.DataFrame([])
    data_order = []
    c = 0
    for f in filenames:
        f_short = re.search(r'/[\w\.-]+.csv', f)
        if f_short:
            print f_short.group()[1:]
            f_short = f_short.group()[1:]
        else:
            print('something went wrong with regular expressions')
            return
        data_file = pd.read_csv(f, sep=',', index_col=0)
        print('data_file name and shape', f_short, data_file.shape)
        print('checking nans', np.sum(np.sum(np.isnan(data_file))))
        data= pd.concat([data, data_file])
        print(data.shape)
        data_order.append(f_short)
    return(data, data_order)

def nan_to_mean(X, big=False):
    if big:
        print('checking nans before', np.sum(np.sum(np.isnan(X))))
        Xmean = np.nanmean(X, 1)
        print('shape of mean', Xmean.shape)
        for i in range(X.shape[0]):
            if np.sum(np.isnan(X.iloc[i, :]))!=0:
                inds = np.where(np.isnan(X.iloc[i, :]))
                #print(inds)
                X.iloc[i, inds[0]] = Xmean[i]
        print('checking nans after', np.sum(np.sum(np.isnan(X))))
        return(X)
    else:
        print('checking nans before', np.sum(np.sum(np.isnan(X))))
        Xmean = np.nanmean(X, 1)
        print('shape of mean', Xmean.shape)
        inds = np.where(np.isnan(X))
        X.iloc[inds] = np.take(Xmean, inds[0])
        print('checking nans after', np.sum(np.sum(np.isnan(X))))
        return(X)


def deal_with_missing(item, missing_cutoff):
    num_samp= item.shape[1]
    item= item[np.sum(np.isnan(item), axis=1) < missing_cutoff* num_samp]
    return(item)

def non_zero_cols(x):
    # print(np.sum(x, axis=0).shape) 
    x= x.loc[:, np.sum(x, axis=0)!=0]
    #print(np.sum(np.sum(x, axis=0)!=0), x.shape)
    return(x)


if __name__ == "__main__":
    setup_dir= os.getcwd()
    disease_type = 'LGG'
    data_directory= setup_dir + '/data/'
    if not os.path.exists(data_directory):
        os.system('mkdir ' + data_directory)
        os.system('Rscript download_data.R '+ disease_type + ' '+ data_directory)
        os.system('mkdir ' + data_directory +'not_in_use/')
        for csvs in ['tcga_LGG_Methylation_280.csv', 'tcga_LGG_RNASeq2_barcodes.csv', 'tcga_LGG_Clinical_280.csv']:
            os.system('mv ' +data_directory + csvs+ ' ' + data_directory +'not_in_use/' )
    mo, mo_order=load_data(data_directory)
    mo=mo.T
    mo=non_zero_cols(mo)
    #outlier
    mo=mo.drop('TCGA-CS-4944')
    #no molecular classification label
    mo=mo.drop(['TCGA-DB-A64S', 'TCGA-DB-A64X', 'TCGA-DH-A66F','TCGA-QH-A65V', 'TCGA-QH-A65Z'])
    mo=mo.T
    #deleting columns with too many missing entries.
    missing_cutoff=0.1
    mo=deal_with_missing(mo, missing_cutoff)
    # replacing missing entries with mean of column
    mo=nan_to_mean(mo, False)
    mo=mo.T
    #mean subtracting
    momean = np.mean(mo, 0)
    mo = mo - momean

    plot_loc = 'LGG_'+ str(mo.shape[1]) + '_' + str(mo.shape[0]) +'_'
    k = 3
    epsilon = 0.1
    plot = True

    #getting and making the sample labels
    labels=pd.read_csv(setup_dir+ 'tcga_LGG_tumor_labels.csv', index_col=0, sep=',')
    all_labels=pd.DataFrame(np.zeros(mo.shape[0]), index=mo.index)
    c=0
    for item in mo.index:
        if item[0:12] in labels.index:
            c=c+1
            all_labels.ix[item]=labels.loc[item[0:12]][0]

    all_labels=all_labels.replace(0, np.nan)
    short_labels=make_categorical(all_labels.iloc[:, 0])
    #doing ridge leverage
    theta, index_keep, tau_sorted, index_drop, tau_tot, AnotkF2 = det_ridge_leverage(mo, k, epsilon, plot, plot_loc)
    print("ridge leverage number of kept columns", theta)
    theta_cl, index_keep_cl, tau_sorted_cl, index_drop_cl, tau_tot_cl = det_ksub_leverage(mo, mo.shape[0],  epsilon, plot, plot_loc + '_classical_')
    plot_comparison(tau_sorted, tau_sorted_cl)
    #svd
    Umo, eigmo, Umot = scipy.linalg.svd(mo.dot(mo.T))
    plot_svd(mo, Umo, eigmo, k, short_labels, plot_loc)
    #check bounds
    bound_check(mo, index_keep, k, epsilon)
    #ridge regression
    hatx_A, hatx_C=do_ridge_reg(all_labels.loc[short_labels.columns].replace(['IDHmut-non-codel',  'IDHmut-codel', 'IDHwt'], [1, 1, -1 ]), mo, index_keep, k, AnotkF2, epsilon, plot_loc+ 'idh_', plot=True)
    hatx_A, hatx_C= do_ridge_reg(all_labels.loc[short_labels.columns].replace(['IDHmut-non-codel',  'IDHmut-codel', 'IDHwt'], [-1, 1, -1 ]), mo, index_keep, k, AnotkF2, epsilon, plot_loc+ 'codel_', plot=True)
    #ridge regression simulation
    nrep=1
    riskA, riskC =do_ridge_reg_sim(mo, index_keep, k, AnotkF2, epsilon, 1,  plot_loc, nrep, 239873, plot=True)
    riskA, riskC =do_ridge_reg_sim(mo, index_keep, k, AnotkF2, epsilon, 10**(-3),  plot_loc, nrep, 8987432, plot=True)
    riskA, riskC =do_ridge_reg_sim(mo, index_keep, k, AnotkF2, epsilon, 10**(3),  plot_loc, nrep, 723421, plot=True)
