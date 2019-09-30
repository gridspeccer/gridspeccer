#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib import gridspec as gs
from matplotlib import collections as coll
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
import pylab as p
import copy
import numpy as np
import scipy as scp
import scipy.special as special

from . import core
from .core import log


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(2, 1, hspace=0.5, height_ratios = [1.,1.3],
                          left=0.05, right=0.98, top=.95, bottom=0.1)
    gs_top = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.3,
                                           width_ratios=[0.5, 1.])
    gs_bot = gs.GridSpecFromSubplotSpec(1, 3, gs_main[1, 0], wspace=0.2,
                                           width_ratios=[1.,.75,.75])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "strucPoisson": gs_top[0, 0],

        "actFunc": gs_bot[0, 0],

        "strucSon": gs_top[0, 1],

        "histMiddle": gs_bot[0, 1],

        "histSigma": gs_bot[0, 2],

    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
        "strucPoisson",
        "strucSon",
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "strucPoisson",
                         "strucSon",
                         "actFunc",
                         "histMiddle",
                         "histSigma"
                     ],
                     label_ypos = {},
                     label_xpos={},
                     )


def get_fig_kwargs():
    width = 12.
    alpha = 0.45
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#


def plot_strucPoisson(ax):

    # Structure of the network with Poisson noise sources

    pass

def plot_strucSon(ax):

    # Structure of the network with sea of noise sources

    pass

def plot_actFunc(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # Do the activation function
    # Dummy data to work with something
    wBias  = np.arange(-15, 16)
    dataP = core.get_data('fig3/actFunc.npy')
    dataSon = core.get_data('fig3/actFuncSon.npy')
    ax.plot(wBias, np.mean(dataP, axis=1),
                linewidth=0.6, marker='x',
                label='Poisson',
                color='tab:blue')
    ax.plot(wBias, np.mean(dataSon,axis=1),
                linewidth=0.6, marker='x',
                label='RN',
                color='tab:orange')
    ax.set_xlabel(r'$w_\mathrm{b} \; [HW. u.]$',fontsize=12)
    ax.set_ylabel(r'mean frequency $\langle \nu \rangle \; [Hz]$',fontsize=12)
    ax.set_xlim([-15.5, 15.5])
    ax.legend(loc=4)

    # make the inset
    # make dummy data
    data = core.get_data('fig3/biasTraces.npy')
    index = np.where(data[:,0] == 0)[0]
    time = data[index,1]
    voltage = data[index,2]
    iax = inset_axes(ax,
                     width = "80%",
                     height= "80%",
                     loc=10,
                     bbox_transform=ax.transAxes,
                     bbox_to_anchor=(.15,.3, .34, .6))
    core.show_axis(iax)
    core.make_spines(iax)
    iax.plot(time, voltage, linewidth=1, color='xkcd:brick red')
    iax.set_ylabel(r'$u_\mathrm{b} \; [mV]$', fontsize=12)
    iax.set_xlabel(r'$t \; [ms]$', fontsize=12)
    iax.set_ylim([-55, -37])
    iax.set_xlim([2000., 2035.])
    iax.set_title('bias neuron')
    iax.tick_params(axis='both', which='both')

    


    return

def plot_histMiddle(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # Generate dummy data
    midPoisson = core.get_data('fig3/poisson_middle.npy')
    midSon = core.get_data('fig3/son_middle.npy')

    # Cut off outlier
    midSon = midSon[np.where(midSon>-10.)[0]]

    # Make the histograms
    maxVal = np.max([np.max(midPoisson), np.max(midSon)])
    minVal = np.min([np.min(midPoisson), np.min(midSon)])
    upper = maxVal + 0.1 * (maxVal - minVal)
    lower = minVal - 0.1 * (maxVal - minVal)
    bins = np.linspace(lower, upper, 30)
    alpha = 0.7
    ax.hist(midPoisson,
            color='tab:blue',
            label='Poisson',
            bins=bins,
            alpha=alpha,
            normed=True)
    ax.hist(midSon,
            color='tab:orange',
            label='RN',
            bins=bins,
            alpha=alpha,
            normed=True)
    ax.set_xlabel(r'$w^0_\mathrm{b} \; [HW.u.]$', fontsize=12)
    ax.set_ylabel(r'density [1]', fontsize=12)
    ax.legend(loc='upper right')

    return

def plot_histSigma(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # Generate dummy data
    midPoisson = core.get_data('fig3/poisson_sigma.npy')
    midSon = core.get_data('fig3/son_sigma.npy')

    # Cut off outlier
    midSon = midSon[np.where(midSon<10.)[0]]

    # Make the histograms
    maxVal = np.max([np.max(midPoisson), np.max(midSon)])
    minVal = np.min([np.min(midPoisson), np.min(midSon)])
    upper = maxVal + 0.1 * (maxVal - minVal)
    lower = minVal - 0.1 * (maxVal - minVal)
    bins = np.linspace(lower, upper, 30)
    alpha = 0.7
    ax.hist(midPoisson,
            color='tab:blue',
            label='Poisson',
            bins=bins,
            alpha=alpha,
            normed=True)
    ax.hist(midSon,
            color='tab:orange',
            label='RN',
            bins=bins,
            alpha=alpha,
            normed=True)
    ax.set_xlabel(r'$s \; [HW.u.]$', fontsize=12)
    ax.set_ylabel(r'density [1]', fontsize=12)
    ax.legend(loc='upper right')

    return
