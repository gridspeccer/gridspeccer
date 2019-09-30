#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    gs_main = gs.GridSpec(3, 1, hspace=0.35,
                          left=0.07, right=0.98, top=.96, bottom=0.08)
    gs_top = gs.GridSpecFromSubplotSpec(1, 4, gs_main[0, 0], wspace=0.3,
                                           width_ratios=[.6, .6, 1.5, 0.4])
    gs_bot = gs.GridSpecFromSubplotSpec(1, 6, gs_main[1, 0],
                                        wspace=0.35,
                                        width_ratios=[5.0, 3.5, 3.5, 3.5, 3.5, 3.5])
    gs_botBot = gs.GridSpecFromSubplotSpec(1, 5, gs_main[2, 0], wspace=0.7,
                                           width_ratios=[0.02,.75, 1., .6,0.02])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "trainingDKL": gs_top[0, 0],
        "singleDKL": gs_top[0, 1],
        "probDist": gs_top[0, 2],
        "marginal": gs_top[0, 3],

        "boxAndWhiskers": gs_bot[0, 0],
        "otherDistr1": gs_bot[0, 1],
        "otherDistr2": gs_bot[0, 2],
        "otherDistr3": gs_bot[0, 3],
        "otherDistr4": gs_bot[0, 4],
        "otherDistr5": gs_bot[0, 5],

        "infSingleDKL": gs_botBot[0, 1],
        "infProbDist": gs_botBot[0, 2],
        "infMarginal": gs_botBot[0, 3],

    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                            "trainingDKL",
                            "singleDKL",
                            "probDist",
                            "marginal",
                            "boxAndWhiskers",
                            "otherDistr1",
                            "infSingleDKL",
                            "infProbDist",
                            "infMarginal",
                     ],
                     label_ypos = {},
                     label_xpos={"singleDKL": 0.1,
                                 "otherDistr1": 0.05},
                     )


def get_fig_kwargs():
    width = 12.
    alpha = 10.0/12.
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

def plot_trainingDKL(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLiterPoisson = core.get_data('fig4/fig4_DKLiterArrPoisson.npy')
    DKLiterValuePoisson = core.get_data('fig4/fig4_DKLiterValuePoisson.npy')
    DKLfinalPoisson = core.get_data('fig4/fig4_DKLtimeValuePoisson.npy')
    DKLiterSon = core.get_data('fig4/fig4_DKLiterArraySon.npy')
    DKLiterValueSon = core.get_data('fig4/fig4_DKLiterValueSon.npy')
    DKLfinalSon = core.get_data('fig4/fig4_DKLtimeValueSon.npy')

    aux.suppPlotTraining(ax,
                         DKLiterValuePoisson,
                         DKLfinalPoisson,
                         DKLiterPoisson,
                         DKLiterValueSon,
                         DKLfinalSon,
                         DKLiterSon)

    #ax.legend(loc='upper right')#, bbox_to_anchor=(0.3, 0.6))
    ax.set_ylim([8e-3,5.e0])

    pass

def plot_singleDKL(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLtimeValuePoisson = core.get_data('fig4/fig4_DKLtimeValuePoisson.npy')
    DKLtimeArrayPoisson = core.get_data('fig4/fig4_DKLtimeArrayPoisson.npy')
    DKLtimeValueSon = core.get_data('fig4/fig4_DKLtimeValueSon.npy')
    DKLtimeArraySon = core.get_data('fig4/fig4_DKLtimeArraySon.npy')

    aux.suppPlotDklTime(ax, DKLtimeArrayPoisson, DKLtimeValuePoisson,
                        DKLtimeArraySon, DKLtimeValueSon)

    #ax.legend(loc='upper right')#, bbox_to_anchor=(0.3, 0.6))
    ax.set_ylim([8e-3,5.e0])
    ax.set_xlim([0., 5e5])

    pass

def plot_probDist(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampledPoisson = core.get_data('fig4/fig4_finalJointPoisson.npy')
    sampledSon = core.get_data('fig4/fig4_finalJointSon.npy')
    target = core.get_data('fig4/fig4_targetJoint.npy')

    aux.suppPlotThreeDistributions(ax,
                                   sampledPoisson,
                                   sampledSon,
                                   target,
                                   errorBar=True)
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)
    ax.set_ylim([0., 0.26])
    ax.legend(bbox_to_anchor=(0.3, 0.8))

    pass

def plot_marginal(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampledPoisson = core.get_data('fig4/fig4_finalMarginalPoisson.npy')
    target = core.get_data('fig4/fig4_targetMarginal.npy')
    sampledSon = core.get_data('fig4/fig4_finalMarginalSon.npy')


    aux.suppPlotThreeDistributions(ax,
                                   sampledPoisson,
                                   sampledSon,
                                   target,
                                   errorBar=True)
    #ax.legend(loc='upper right')
    ax.set_ylabel(r'$p_\mathrm{marginal}(z_i = 1)$', fontsize=12)
    #ax.set_ylim([0., 1.1])
    ax.set_xlabel('neuron id', fontsize=12)


    pass

def plot_boxAndWhiskers(ax):
    '''
        Plot box and whiskers plot of DKLs (joint) after training with Poisson
        over several sampled distributions
    '''

    core.show_axis(ax)
    core.make_spines(ax)

    # load the joint Poisson DKL from fig4
    DKLfinalPoisson = core.get_data('fig4/fig4_DKLtimeValuePoisson.npy')
    refDKLs = DKLfinalPoisson[:, -1]

    # load the data
    dataPoisson = core.get_data('figAppendix1/poissonJointsDKL.npy')

    # gather the data into an array
    data = [dataPoisson[i,:] for i in range(len(dataPoisson[:,0]))]
    indexToShow = [0, 1, 6, 7, 19]
    dataFiltered = [data[i] for i in indexToShow]
    data = [refDKLs] + dataFiltered

    # plot
    b1 = ax.boxplot(data[0],
                    sym='x',
                    positions=[0],
                    widths=0.5,
                    #flierprops={'markeredgecolor': 'tab:blue'},
                    boxprops={'facecolor': 'tab:blue'},
                    patch_artist=True)
    #for element in ['caps']:
    #    p.setp(b1[element], color='tab:blue')
    ax.boxplot(data[1:],
               sym='x',
               widths=0.5,
               positions=range(1,6))
    ax.set_xlim([-0.6, 5.6])
    ax.set_yscale('log')
    #ax.text( 0.1, 1e-3, 'data for\nFig. 4B', ha='center')
    ax.set_ylabel(
        r'$\mathregular{D}_\mathregular{KL} \left[ \, p(\mathbf{z}) \, || \, p\!^*(\mathbf{z}) \, \right]$', fontsize=12)
    ax.set_xlabel(r'# Distribution ID', fontsize=12)
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(["C","1","2","3","4","5"], fontsize=11)
    ax.set_ylim([7e-4, 2e0])

    return

colors = plt.cm.tab10(np.linspace(0, 1, 6))

def plot_otherDistr1(ax):

    core.show_axis(ax)
    core.make_spines(ax)
    distributions = core.get_data("fig4/fig4_otherDistr.npy")

    numDistrs = len(distributions[:,0])
    x = np.array(range(0, len(distributions[0,:])))
    width = 0.5

    
    ax.bar(x, distributions[0,:], width=width,
            label='ID {}'.format(1), bottom=1E-3,
            color=colors[0])

    ax.set_title('ID 1')
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)
    ax.set_ylabel(r'$p_\mathrm{joint}(\mathbf{z})$', fontsize=12)
    ax.set_xlim([min(x - .75), max(x + .35)])
    ax.set_xticks([])
    ax.set_yticks([0.1])

    return

def plot_otherDistr2(ax):

    core.show_axis(ax)
    core.make_spines(ax)
    distributions = core.get_data("fig4/fig4_otherDistr.npy")

    numDistrs = len(distributions[:,0])
    x = np.array(range(0, len(distributions[0,:])))
    width = 0.5

    
    ax.bar(x, distributions[1,:], width=width,
            label='ID {}'.format(2), bottom=1E-3,
            color=colors[1])

    ax.set_title('ID 2')
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)
    #ax.set_ylabel(r'$p_\mathrm{joint}(\mathbf{z})$', fontsize=12)
    ax.set_xlim([min(x - .75), max(x + .35)])
    ax.set_xticks([])
    ax.set_yticks([0.1])

    return

def plot_otherDistr3(ax):

    core.show_axis(ax)
    core.make_spines(ax)
    distributions = core.get_data("fig4/fig4_otherDistr.npy")

    numDistrs = len(distributions[:,0])
    x = np.array(range(0, len(distributions[0,:])))
    width = 0.5

    
    ax.bar(x, distributions[2,:], width=width,
            label='ID {}'.format(3), bottom=1E-3,
            color=colors[2])

    ax.set_title('ID 3')
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)
    #ax.set_ylabel(r'$p_\mathrm{joint}(\mathbf{z})$', fontsize=12)
    ax.set_xlim([min(x - .75), max(x + .35)])
    ax.set_xticks([])
    ax.set_yticks([0.1, 0.2, 0.3])

    return

def plot_otherDistr4(ax):

    core.show_axis(ax)
    core.make_spines(ax)
    distributions = core.get_data("fig4/fig4_otherDistr.npy")

    numDistrs = len(distributions[:,0])
    x = np.array(range(0, len(distributions[0,:])))
    width = 0.5

    
    ax.bar(x, distributions[3,:], width=width,
            label='ID {}'.format(4), bottom=1E-3,
            color=colors[3])

    ax.set_title('ID 4')
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)
    #ax.set_ylabel(r'$p_\mathrm{joint}(\mathbf{z})$', fontsize=12)
    ax.set_xlim([min(x - .75), max(x + .35)])
    ax.set_xticks([])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])

    return

def plot_otherDistr5(ax):

    core.show_axis(ax)
    core.make_spines(ax)
    distributions = core.get_data("fig4/fig4_otherDistr.npy")

    numDistrs = len(distributions[:,0])
    x = np.array(range(0, len(distributions[0,:])))
    width = 0.5

    
    ax.bar(x, distributions[4,:], width=width,
            label='ID {}'.format(5), bottom=1E-3,
            color=colors[4])

    ax.set_title('ID 5')
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)
    #ax.set_ylabel(r'$p_\mathrm{joint}(\mathbf{z})$', fontsize=12)
    ax.set_xlim([min(x - .75), max(x + .35)])
    ax.set_xticks([])
    ax.set_yticks([0.1])

    return

def plot_infSingleDKL(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    DKLtimeValuePoisson = core.get_data('fig4/fig5_dklTimeValuePoisson.npy')
    DKLtimeArrayPoisson = core.get_data('fig4/fig5_dklTimeArrayPoisson.npy')
    DKLtimeValueSon = core.get_data('fig4/fig5_dklTimeValueSon.npy')
    DKLtimeArraySon = core.get_data('fig4/fig5_dklTimeArraySon.npy')

    aux.suppPlotDklTime(ax, DKLtimeArrayPoisson, DKLtimeValuePoisson,
                        DKLtimeArraySon, DKLtimeValueSon)

    #ax.legend(loc='upper right')#, bbox_to_anchor=(0.3, 0.6))
    ax.set_ylim([8e-3,5.e0])
    ax.set_xlim([0., 5e5])

    pass

def plot_infProbDist(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampledPoisson = core.get_data('fig4/fig5_jointPoisson.npy')
    target = core.get_data('fig4/fig5_jointTarget.npy')
    sampledSon = core.get_data('fig4/fig5_jointSon.npy')

    aux.suppPlotThreeDistributions(ax,
                                   sampledPoisson,
                                   sampledSon,
                                   target,
                                   errorBar=True)
    #ax.legend(bbox_to_anchor=(0.5, 0.8))
    #ax.set_ylim([0., .8])
    ax.set_xlabel(r'$\mathbf{z}$, states', fontsize=12)


    pass

def plot_infMarginal(ax):

    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    sampledPoisson = core.get_data('fig4/fig5_margPoisson.npy')
    target = core.get_data('fig4/fig5_margTarget.npy')
    sampledSon = core.get_data('fig4/fig5_margSon.npy')

    aux.suppPlotThreeDistributions(ax,
                                   sampledPoisson,
                                   sampledSon,
                                   target,
                                   errorBar=True)
    #ax.legend(bbox_to_anchor=(0.75, 0.75))
    #ax.set_ylim([0., 1.05])
    ax.set_xlabel('neuron id', fontsize=12)
    ax.set_ylabel(r'$p_\mathrm{marginal}(z_i = 1)$', fontsize=12)

    pass