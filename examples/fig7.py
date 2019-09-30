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
    gs_main = gs.GridSpec(2, 1, hspace=0.4, height_ratios=[0.8, 1.],
                          left=0.1, right=0.95, top=.95, bottom=0.16)
    gs_top = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.4,
                                        width_ratios=[1., 1.])
    gs_bottom = gs.GridSpecFromSubplotSpec(1, 2, gs_main[1, 0], wspace=0.5,
                                           width_ratios=[1., 1.])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "mnistTraining": gs_top[0, 0],

        "fashionTraining": gs_top[0, 1],

        "mnistMixtureMatrix": gs_bottom[0, 0],

        "fashionMixtureMatrix": gs_bottom[0, 1],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
        "mnistMixtureMatrix",
        "fashionMixtureMatrix"
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "mnistTraining",
                         "fashionTraining",
                         "mnistMixtureMatrix",
                         "fashionMixtureMatrix"
                     ],
                     label_ypos={'mnistTraining': 1.,
                                 'mnistMixtureMatrix': 1.,
                                 'fashionTraining': 1.,
                                 'fashionMixtureMatrix': 1.},
                     label_xpos={'mnistTraining': -.15,
                                 'mnistMixtureMatrix': -.2,
                                 'fashionTraining': -.15,
                                 'fashionMixtureMatrix': -.2
                                 }
                     )


def get_fig_kwargs():
    width = 6.
    alpha = 5. / 6.16
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

def plot_mnistTraining(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # Load the data
    abstractRatio = core.get_data('figInference/mnistAbstract.npy')
    classRatio = core.get_data('figInference/mnistClassRatios.npy')
    iterNumb = core.get_data('figInference/mnistClassRatiosArray.npy')

    # Do the plot
    aux.plotITLTraining(ax, abstractRatio, classRatio, iterNumb)
    ax.legend()

    return


def plot_fashionTraining(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # Load the data
    abstractRatio = core.get_data('figInference/fashionAbstract.npy')
    classRatio = core.get_data('figInference/fashionClassRatios.npy')
    iterNumb = core.get_data('figInference/fashionClassRatiosArray.npy')

    # Do the plot
    aux.plotITLTraining(ax, abstractRatio, classRatio, iterNumb)
    ax.legend()

    return

    return


def plot_mnistMixtureMatrix(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # Load the data
    mixtureMatrix = core.get_data('figInference/mnistConfMatrix.npy')
    labels = [0, 1, 4, 7]

    # Do the plot
    aux.plotMixtureMatrix(ax, mixtureMatrix, labels)

    return


def plot_fashionMixtureMatrix(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # Load the data
    mixtureMatrix = core.get_data('figInference/fashionConfMatrix.npy')
    labels = ['t-shirt', 'trouser', 'sneaker']

    # Do the plot
    aux.plotMixtureMatrix(ax, mixtureMatrix, labels)

    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_yticklabels(labels, fontsize=8, rotation=45)

    return
