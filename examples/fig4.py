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
from .core import log


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(1, 1, hspace=0.1,
                          left=0.05, right=0.95, top=.90, bottom=0.16)
    gs_bottom = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.1,
                                           width_ratios=[1., 1])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "moduleSketch": gs_bottom[0, 0],
        "modulePhoto": gs_bottom[0, 1],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    # TODO: Uncomment & decide for each subplot!
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
        'moduleSketch',
        'modulePhoto',
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         'moduleSketch',
                         'modulePhoto',
                     ],
                     label_ypos = {'moduleSketch': 1.,
                                   'modulePhoto': 1.},
                     label_xpos={'moduleSketch': -0.05,
                                 'modulePhoto': -0.05
                                 }
                     )


def get_fig_kwargs():
    width = 6.
    alpha = 0.4
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#
def plot_moduleSketch(ax):

    # This is a tikz-figure and it will be created in the tex document

    pass


def plot_modulePhoto(ax):

    # This is a tikz picture and it will be created in the tex document

    pass
