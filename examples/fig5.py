#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib import gridspec as gs
from matplotlib import collections as coll
import pylab as p
import copy
import numpy as np

from . import core
from .core import log
from . import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(4, 1, hspace=.3,
                          left=0.1, right=0.95, top=.95, bottom=0.03)
    gs_top1 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.3,
                                        width_ratios=[1.,1.])
    gs_top2 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[1, 0], wspace=0.3,
                                        width_ratios=[1.,1.])
    gs_top3 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[2, 0], wspace=0.3,
                                        width_ratios=[1.,1.])
    gs_top4 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[3, 0], wspace=0.3,
                                        width_ratios=[1.,1.])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "trainingPoisson": gs_top1[0, 0],
        "trainingSon": gs_top1[0, 1],
        "dklPoisson": gs_top2[0, 0],
        "dklSon": gs_top2[0, 1],
        "jointPoisson": gs_top3[0, 0],
        "jointSon": gs_top3[0, 1],
        "marginalPoisson": gs_top4[0, 0],
        "marginalSon": gs_top4[0, 1],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    # TODO: Uncomment & decide for each subplot!
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "trainingPoisson",
                         "trainingSon",
                         "dklPoisson",
                         "dklSon",
                         "jointPoisson",
                         "jointSon",
                         "marginalPoisson",
                         "marginalSon",
                     ],
                     #    label_ypos = {'delays_all': 0.95},
                     label_xpos={'dklPoisson': 0.1,
                                 'dklSon': 0.1
                                 }
                     )


def get_fig_kwargs():
    width = 6
    alpha = 11./7.
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#
def plot_trainingPoisson(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLiter = core.get_data('fig4/fig4_DKLiterArrPoisson.npy')
    DKLiterValue = core.get_data('fig4/fig4_DKLiterValuePoisson.npy')
    DKLfinal = core.get_data('fig4/fig4_DKLtimeValuePoisson.npy')

    aux.suppPlotTraining(ax, DKLiterValue, DKLfinal, DKLiter)

    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.7))
    ax.set_ylim([8e-3,2.2e0])
    ax.text(150, 4., 'Poisson noise', weight='bold')


def plot_dklPoisson(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLtimeValue = core.get_data('fig4/fig4_DKLtimeValuePoisson.npy')
    DKLtimeArray = core.get_data('fig4/fig4_DKLtimeArrayPoisson.npy')

    aux.suppPlotDklTime(ax, DKLtimeArray, DKLtimeValue)
    ax.set_ylim([7e-3,5e0])


def plot_jointPoisson(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig4/fig4_finalJointPoisson.npy')
    target = core.get_data('fig4/fig4_targetJoint.npy')

    aux.suppPlotDistributions(ax, sampled, target, errorBar=False)
    ax.set_xlabel(r'$\mathbf{z}$, states')
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.5))
    ax.set_ylim([0., 0.26])


def plot_marginalPoisson(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig4/fig4_finalMarginalPoisson.npy')
    target = core.get_data('fig4/fig4_targetMarginal.npy')

    aux.suppPlotDistributions(ax, sampled, target)

    ax.set_xlabel(r'neuron id')
    ax.legend(loc='lower left', bbox_to_anchor=(0.2, 0.75), ncol=2, fontsize=8)
    ax.set_ylim([0., 1.10])


def plot_trainingSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLiter = core.get_data('fig4/fig4_DKLiterArraySon.npy')
    DKLiterValue = core.get_data('fig4/fig4_DKLiterValueSon.npy')
    DKLfinal = core.get_data('fig4/fig4_DKLtimeValueSon.npy')

    aux.suppPlotTraining(ax, DKLiterValue, DKLfinal, DKLiter)

    ax.text(70, 4., 'Decorrelation Network', weight='bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.7))
    ax.set_ylim([8e-3,2.2e0])


def plot_dklSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLtimeValue = core.get_data('fig4/fig4_DKLtimeValueSon.npy')
    DKLtimeArray = core.get_data('fig4/fig4_DKLtimeArraySon.npy')

    aux.suppPlotDklTime(ax, DKLtimeArray, DKLtimeValue)
    ax.set_ylim([7e-3,5e0])


def plot_jointSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig4/fig4_finalJointSon.npy')
    target = core.get_data('fig4/fig4_targetJoint.npy')

    aux.suppPlotDistributions(ax, sampled, target, errorBar=False)
    ax.set_xlabel(r'$\mathbf{z}$, states')
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.5))
    ax.set_ylim([0., 0.26])


def plot_marginalSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig4/fig4_finalMarginalSon.npy')
    target = core.get_data('fig4/fig4_targetMarginal.npy')

    aux.suppPlotDistributions(ax, sampled, target)

    ax.set_xlabel(r'neuron id')
    ax.legend(loc='lower left', bbox_to_anchor=(0.2, 0.75), ncol=2, fontsize=8)
    ax.set_ylim([0., 1.1])
