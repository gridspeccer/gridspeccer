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
    gs_main = gs.GridSpec(3, 1, hspace=.3,
                          left=0.1, right=0.95, top=.95, bottom=0.07)
    gs_top1 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.3,
                                         width_ratios=[1., 1.])
    gs_top2 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[1, 0], wspace=0.3,
                                         width_ratios=[1., 1.])
    gs_top3 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[2, 0], wspace=0.3,
                                         width_ratios=[1., 1.])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "dklPoisson": gs_top1[0, 0],
        "dklSon": gs_top1[0, 1],
        "jointPoisson": gs_top2[0, 0],
        "jointSon": gs_top2[0, 1],
        "marginalPoisson": gs_top3[0, 0],
        "marginalSon": gs_top3[0, 1],

    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    # TODO: Uncomment & decide for each subplot!
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "dklPoisson",
                         "dklSon",
                         "jointPoisson",
                         "jointSon",
                         "marginalPoisson",
                         "marginalSon"
                     ],
                     #    label_ypos = {'delays_all': 0.95},
                     label_xpos={'dummy2': 0.02,
                                 "dummy1": 0.02
                                 }
                     )


def get_fig_kwargs():
    width = 6
    alpha = 33./28.
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#
def plot_dklPoisson(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLtimeValue = core.get_data('fig5/fig5_dklTimeValuePoisson.npy')
    DKLtimeArray = core.get_data('fig5/fig5_dklTimeArrayPoisson.npy')

    aux.suppPlotDklTime(ax, DKLtimeArray, DKLtimeValue)

    ax.text(1e2, 8., 'Poisson noise', weight='bold')
    ax.set_ylim([6e-3, 6e0])


def plot_jointPoisson(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig5/fig5_jointPoisson.npy')
    target = core.get_data('fig5/fig5_jointTarget.npy')

    aux.suppPlotDistributions(ax, sampled, target, errorBar=False)
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.4))
    ax.set_xlabel(r'$\mathbf{z}$, states')
    ax.set_ylim([0., 0.79])


def plot_marginalPoisson(ax):

    # This is a tikz picture and it will be created in the tex document

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig5/fig5_margPoisson.npy')
    target = core.get_data('fig5/fig5_margTarget.npy')

    aux.suppPlotDistributions(ax, sampled, target)
    ax.set_xlabel(r'neuron id')
    ax.legend(loc='lower left', bbox_to_anchor=(0.2, 0.8), ncol=2, fontsize=8)
    ax.set_ylim([0., 1.2])


def plot_dklSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLtimeValue = core.get_data('fig5/fig5_dklTimeValueSon.npy')
    DKLtimeArray = core.get_data('fig5/fig5_dklTimeArraySon.npy')

    aux.suppPlotDklTime(ax, DKLtimeArray, DKLtimeValue)
    ax.text(8e1, 8., 'Decorrelation Network', weight='bold')
    ax.set_ylim([6e-3, 6e0])


def plot_jointSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig5/fig5_jointSon.npy')
    target = core.get_data('fig5/fig5_jointTarget.npy')

    aux.suppPlotDistributions(ax, sampled, target, errorBar=False)
    ax.set_xlabel(r'$\mathbf{z}$, states')

    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.4))
    ax.set_ylim([0., 0.79])


def plot_marginalSon(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampled = core.get_data('fig5/fig5_margSon.npy')
    target = core.get_data('fig5/fig5_margTarget.npy')

    aux.suppPlotDistributions(ax, sampled, target)

    ax.set_xlabel(r'neuron id')
    ax.legend(loc='lower left', bbox_to_anchor=(0.2, 0.8), ncol=2, fontsize=8)
    ax.set_ylim([0., 1.2])