#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib import gridspec as gs
from matplotlib import collections as coll
import pylab as p
import copy
import numpy as np
import matplotlib.patches as patches

from . import core
from . import aux
from .core import log


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(2, 1, hspace=0.3, wspace=0.02,
                          left=0.08, right=0.98, top=.95, bottom=0.1,
                          height_ratios=[1., 1.])
    gs_aux_upper = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.15,
                                           width_ratios=[.6, 1.])
    gs_aux_lower = gs.GridSpecFromSubplotSpec(1, 2, gs_main[1, 0], wspace=0.10,
                                              width_ratios=[0.5,1.0])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "bmStructure": gs_aux_upper[0, 0],

        "tracesToStates": gs_aux_upper[0, 1],

        "distr": gs_aux_lower[0, 0],

        "rbmSketch": gs_aux_lower[0, 1],

    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
        "tracesToStates",
        "bmStructure",
        "rbmSketch",
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "bmStructure",
                         "tracesToStates",
                         "distr",
                         "rbmSketch",
                     ],
                     label_ypos = {},
                     label_xpos={'tracesToStates': -.2,
                                 'rbmSketch': 0.02},
                     )


def get_fig_kwargs():
    width = 9.
    alpha = 10./12.
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#


def plot_bmStructure(ax):

    # Photo of an assembled wafer unit

    pass

def plot_tracesToStates(ax):

    volts = core.get_data('fig2/fig2_trace.npy')
    fontSize = 14

    for i, v in enumerate(volts):
        ax.plot(v + i * 3. - 3., '-k')
        if i == 2:
            t1Max = np.max(v + i * 3. - 3.)
            t1Reset = (v + i * 3. - 3.)[150]
        if i == 0:
            t3Max = np.max(v + i * 3. - 3.)
            t3Reset = (v + i * 3. - 3.)[150]
    ax.text(170, -57.5, r't [a.u.]', fontsize=fontSize)

    ax.text(-50, -48.5, r'$u_3$', fontsize=fontSize)
    ax.text(-50, -51.5, r'$u_2$', fontsize=fontSize)
    ax.text(-50, -54.5, r'$u_1$', fontsize=fontSize)

    ax.text(140., -44.7, '101', color='r', rotation=90, fontsize=fontSize)
    ax.text(290., -44.7, '100', color='r', rotation=90, fontsize=fontSize)
    ax.text(-50., -46.4, 'z', color='r', fontsize=fontSize)

    ax.axvline(150., ymin=.015, ymax=.78, color='r')
    ax.axvline(300., ymin=.015, ymax=.78, color='r')

    # Add path to trace one
    t1rect = patches.Rectangle((140,t1Reset),100,t1Max - t1Reset ,
                                linewidth=0,facecolor="tab:green",
                                alpha=0.3)
    ax.add_patch(t1rect)

    # Add path to trace three
    spikeTimes = [0, 62, 258.5, 376]
    widths = [46, 100, 100, 40]
    for (spikeTime, width) in zip(spikeTimes, widths):
        t3rect = patches.Rectangle((spikeTime,t3Reset),width,t3Max - t3Reset ,
                                linewidth=0,facecolor="tab:green",
                                alpha=0.3)
        ax.add_patch(t3rect)


    rect = patches.Rectangle((140,-51),50,3,linewidth=0,edgecolor='r',facecolor="red", alpha=0.3)
    #ax.add_patch(rect)


    ax.set_ylim(-55.5, -44)

    return

def plot_distr(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # Load data
    theo_distr = core.get_data('fig2/sampling_illustr_theo.npy')
    sim_distr = core.get_data('fig2/sampling_illustr_sim.npy')


    x = np.array(range(0, len(sim_distr)))
    # make the bar plots
    ylabels = ['0.05', '0.1', '0.2']
    ax.bar(x, theo_distr, width=0.35, label='target',
           bottom=1E-3, color='tab:blue')
    ax.bar(x + 0.35, sim_distr, width=0.35,
           label='sampled', bottom=1E-3, color='tab:orange')
    ylim = ax.get_ylim()
    ylimNew = [0., ylim[1]]
    ax.set_ylim(ylimNew)
    ax.set_xlim([min(x - .35), max(x + 0.35 + .35)])
    ax.legend(fontsize=13)

    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=16)
    ax.set_ylabel(r'$p_\mathrm{joint}(\mathbf{z})$', fontsize=16)
    ax.set_xticks(x + 0.35/2.0)
    ax.set_xticklabels(['000', '001', '010', '011', '100', '101', '110', '111'], fontsize = 10)

def plot_rbmSketch(ax):

    # Zoom to a retoicle with an incoming connection
    # neurons and synapses are marked

    pass