#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib import gridspec as gs
from matplotlib import collections as coll
import pylab as p
import copy
import numpy as np
from scipy.misc import imresize

from . import core
from . import aux
from .core import log


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(1, 1, hspace=0.01,
                          left=0.06, right=0.95, top=.98, bottom=0.01)
    gs_top = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.2,
                                        width_ratios=[1., 1.])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "fashion": gs_top[0, 1],
        "mnist": gs_top[0, 0]
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
                         "mnist",
                         "fashion"
                     ],
                     label_ypos = {'fashion': 1.,
                                   'mnist': 1.},
                     label_xpos={'fashion': -0.15,
                                 "mnist": -0.15
                                 }
                     )


def get_fig_kwargs():
    width = 6.
    alpha = 0.6
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#


def plot_fashion(ax):

    # load the data
    original = core.get_data('fig2/fashion_orig.npy')

    # Do the actual plotting
    aux.plotExamplePictures(ax, original, (12,12), (28,28), (2,4))


def plot_mnist(ax):

    # load the data
    original = core.get_data('fig2/mnist_orig.npy')

    # Do the actual plotting
    aux.plotExamplePictures(ax, original, (12,12), (28,28), (2,4))


####################
# Support functions to reduce code duplicates
# These functions appear several times in this script not in any other



