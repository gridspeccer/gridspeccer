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
    gs_main = gs.GridSpec(2, 1, hspace=0.1,
                          left=0.1, right=0.95, top=.95, bottom=0.16)
    gs_bottom = gs.GridSpecFromSubplotSpec(1, 3, gs_main[1, 0], wspace=0.1,
                                           width_ratios=[1., 2., 1.3])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "privatePoisson": gs_bottom[0, 0],
        "sonInput": gs_bottom[0, 1],
        "rbmStructure": gs_bottom[0, 2],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    # TODO: Uncomment & decide for each subplot!
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
        "privatePoisson",
        "sonInput",
        "rbmStructure"
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "privatePoisson",
                         "sonInput",
                         "rbmStructure",
                     ],
                     #    label_ypos = {'delays_all': 0.95},
                     label_xpos={'dummy2': 0.02,
                                 "dummy1": 0.02
                                 }
                     )


def get_fig_kwargs():
    width = 12.
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
def plot_privatePoisson(ax):

    # This is a tikz-figure and it will be created in the tex document

    pass


def plot_sonInput(ax):

    # This is a tikz picture and it will be created in the tex document

    pass


def plot_rbmStructure(ax):

    # This is a tikz picture and it will be created in the tex document

    pass



####################
# Support functions to reduce code duplicates
# These functions appear several times in this script not in any other


def plotExamplePictures(ax, original):

    # Layout specification
    picSizeRed = (12, 12)
    picSize = (28, 28)
    N_vertical = 2
    N_horizontal = 4
    half = 3
    frame = 1

    # Do the actual plotting
    # create the picture matrix
    pic = np.ones(((2 * N_vertical + 1) * frame + 2 * N_vertical * picSize[0] + half,
                   (N_horizontal + 1) * frame + N_horizontal * picSize[1])) * 255

    # Plot the upper 8 examples (originals)
    for counter in range(N_vertical * N_horizontal):
        i = counter % N_vertical
        j = int(counter / N_vertical)
        picVec = original[counter, 1:]
        picCounter = np.reshape(picVec, picSize)

        pic[(i + 1) * frame + i * picSize[0]: (i + 1) * frame + (i + 1) * picSize[0],
            (j + 1) * frame + j * picSize[1]: (j + 1) * frame + (j + 1) * picSize[1]] = picCounter

    # Plot the lower 8 examples (reduced)
    for counter in range(N_vertical * N_horizontal):
        i = counter % N_vertical + 2
        j = int(counter / N_vertical)
        picVec = original[counter, 1:]
        picCounter = np.reshape(picVec, picSize)
        picCounter = imresize(picCounter, picSizeRed, interp='nearest')
        median = np.percentile(picCounter, 50)
        picCounter = ((np.sign(picCounter - median) + 1) / 2) * 255.
        picCounter = imresize(picCounter, picSize, interp='nearest')

        pic[(i + 1) * frame + half + i * picSize[0]: (i + 1) * frame + (i + 1) * picSize[0] + half,
            (j + 1) * frame + j * picSize[1]: (j + 1) * frame + (j + 1) * picSize[1]] = picCounter

    ax.imshow(pic, cmap='Greys')
