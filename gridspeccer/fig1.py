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

from . import core
from .core import log
from . import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(1, 1, hspace=0.65,
                          left=0.02, right=0.98, top=.9, bottom=0.1)
    gs_aux = gs.GridSpecFromSubplotSpec(1, 4, gs_main[0, 0], wspace=0.1,
                                           width_ratios=[.8, 1.0, 1.4, 1.])
    gs_aux2 = gs.GridSpecFromSubplotSpec(2, 1, gs_aux[0, 3], hspace=0.5,
                                           height_ratios=[1., 1.])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "waferPhoto": gs_aux[0, 0],

        "waferTopView": gs_aux[0, 1],

        "reticle": gs_aux[0, 2],

        "pspPreCalib": gs_aux2[0, 0],

        "pspPostCalib": gs_aux2[1,0],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
        "waferPhoto",
        "waferTopView",
        "reticle"
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "waferPhoto",
                         "waferTopView",
                         "reticle",
                         "pspPreCalib",
                         "pspPostCalib"
                     ],
                     label_ypos = {},
                     label_xpos={},
                     )


def get_fig_kwargs():
    width = 12.
    alpha = 4./12.
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#


def plot_waferPhoto(ax):

    # Photo of an assembled wafer unit

    pass

def plot_waferTopView(ax):

    # top view of the wafer without post-porcessing
    # To connections and the zooming area for the c part should be
    # drawn into it

    pass

def plot_reticle(ax):

    # Zoom to a retoicle with an incoming connection
    # neurons and synapses are marked

    pass

def plot_pspPreCalib(ax):

    # Bunch of psps on several neurons overleaid using the ideal
    # translation formula
    time = core.get_data('fig1/pspTimePre.npy')
    v = core.get_data('fig1/pspVoltagePre.npy')

    aux.plotPSPs(ax, time, v)
    ax.set_title('pre-calibration')

    # make inset
    iax = inset_axes(ax, width = "40%", height= "50%", loc="upper right")
    aux.plotPSPs(iax, time, v, normed=True)
    iax.set_xticks([])
    iax.set_yticks([])
    [yLow, yHigh] = iax.get_ylim()
    iax.set_ylim([yLow, yLow + 1.1 * (yHigh - yLow)])
    iax.set_xlabel(r't [ms]', fontsize=8)
    iax.set_ylabel(r'$u_\mathrm{normed}$ [1]', fontsize=8)

    pass


def plot_pspPostCalib(ax):

    # Bunch of psps on several neurons overleaid
    # using the calibration
    time = core.get_data('fig1/pspTimePost.npy')
    v = core.get_data('fig1/pspVoltagePost.npy')

    aux.plotPSPs(ax, time, v)
    ax.set_title('post-calibration')

    # make inset
    iax = inset_axes(ax, width = "40%", height= "50%", loc="upper right")
    aux.plotPSPs(iax, time, v, normed=True)
    iax.set_xticks([])
    iax.set_yticks([])
    [yLow, yHigh] = iax.get_ylim()
    iax.set_ylim([yLow, yLow + 1.1 * (yHigh - yLow)])
    iax.set_xlabel(r't [ms]', fontsize=8)
    iax.set_ylabel(r'$u_\mathrm{normed}$ [1]', fontsize=8)

    pass
