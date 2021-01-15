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


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(2, 1, hspace=0.65,
                          left=0.1, right=0.95, top=.95, bottom=0.16)
    gs_top = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.2,
                                        width_ratios=[1., 1.])
    gs_bottom = gs.GridSpecFromSubplotSpec(1, 2, gs_main[1, 0], wspace=0.1,
                                           width_ratios=[1., 1.])

    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "tracesToStates": gs_top[0, 0],

        "structure": gs_top[0, 1],

        "dklTime": gs_bottom[0, 0],

        "pspShapes": gs_bottom[0, 1],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
        "tracesToStates",
        "structure"
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "tracesToStates",
                         "structure",
                         "dklTime",
                         "pspShapes"
                     ],
                     label_ypos = {'pspShapes': 1.,
                                   'dklTime':1.},
                     label_xpos={'dummy2': 0.02,
                                 "dummy1": 0.02,
                                 'pspShapes':0.02,
                                 'structure':-0.05,
                                 }
                     )


def get_fig_kwargs():
    width = 6.
    alpha = 5./6.16
    return {"figsize": (width, alpha*width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#
def plot_tracesToStates(ax):
    volts = core.get_data('fig1_voltage_trace.npy')

    for i, v in enumerate(volts):
        ax.plot(v + i * 3. - 3., '-k')
    ax.text(170, -57.5, r't [a.u.]')

    ax.text(-80, -48.5, r'$u_3$')
    ax.text(-80, -51.5, r'$u_2$')
    ax.text(-80, -54.5, r'$u_1$')

    ax.text(130., -45.2, '101', color='r', rotation=90, fontsize=10)
    ax.text(280., -45.2, '100', color='r', rotation=90, fontsize=10)
    ax.text(-80., -46.4, 'z', color='r', fontsize=12)

    ax.axvline(150., ymin=.015, ymax=.78, color='r')
    ax.axvline(300., ymin=.015, ymax=.78, color='r')

    ax.set_ylim(-55.5, -44)


def plot_structure(ax):

    #
    #   This plot is skipped and will be
    #   created with tikz in the tex-file
    #

    pass


def plot_dklTime(ax):

    dkls = core.get_data('fig1_ll_dkls.npy')
    dkl_times = core.get_data('fig1_ll_dkl_times.npy')

    core.show_axis(ax)
    core.make_spines(ax)

    for dkl in dkls:
        ax.plot(dkl_times, dkl)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$t$ [ms]')
    ax.set_ylabel(
        r'$\mathregular{D}_\mathregular{KL} \left[ \, p(\mathbf{z}) \, || \, p\!^*(\mathbf{z}) \, \right]$')

    pass


def plot_pspShapes(ax):
    core.show_axis(ax)
    core.make_spines(ax)

    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Parameters to be used
    linewidth = 1.5
    weight_noise = 0.2
    v_rest = -55.
    dt = 0.1
    t_full = 28
    t_spike = 5
    t_syn = 10
    indSpike = int(t_spike / dt)
    indPSPends = int((t_spike + t_syn)/dt)
    rectangularWeight = Vpsp_curr_exp(weight=weight_noise,
                                      title='rec_weight')

    # array for the time axis
    timeArray = np.arange(0., t_full, dt)

    # Baseline at the resting potential
    baseLine = np.zeros(len(timeArray)) + v_rest

    # bio PSP
    bioPSP = np.zeros(len(timeArray)) + v_rest
    kernel = Vpsp_curr_exp(weight=weight_noise, tdur=t_full - t_spike)
    bioPSP[indSpike:indSpike + len(kernel)] += kernel

    # Rectangular PSP
    recPSP = np.zeros(len(timeArray)) + v_rest
    recPSP[indSpike:indSpike + int(t_syn / dt)] += rectangularWeight

    # Plot the PSP
    ax.plot(timeArray, baseLine, 'k', linewidth=1, alpha=0.5)
    ax.plot(timeArray, bioPSP, 'r', linewidth=linewidth, label='alpha PSP')
    ax.plot(timeArray, recPSP, 'b--', linewidth=linewidth, label='rectangular PSP')

    # Fill the areas to be integrated
    ax.fill_between(timeArray[indSpike:indPSPends],
                    baseLine[indSpike:indPSPends],
                    bioPSP[indSpike:indPSPends],
                    color='r',
                    alpha=0.5)
    ax.fill_between(timeArray[indSpike:indPSPends],
                    baseLine[indSpike:indPSPends],
                    recPSP[indSpike:indPSPends],
                    color='b',
                    alpha=0.5)
    ax.set_xlabel(r'$t$ [a.u.]')
    ax.set_ylabel(r'$\bar u_k$ [a.u.]')

    # Plot the arrow and the times
    ypos = max(bioPSP) + 0.005 # to be tuned for prettiness
    core.make_arrow(ax, (t_spike, ypos), (t_spike + 10., ypos))
    ax.text(t_spike + 5., ypos,
            r"$\tau_{\mathregular{ref}}\approx\tau_{\mathregular{syn}}$",
            color="r", va="bottom", ha="center")
    ax.legend(fontsize=8)

    pass


#######################
# Supplementary function to plot a realistic PSP
# Taken from: Petrovici et al. 2017 IJCNN (a.k.a. robustness from structure)
#######################

def Vpsp_curr_exp(weight=None, neuron_params=None, set_singlepara=None,
                  title=None, tdur=None, plot=False):
    '''
    Excitatory PSP of IF_curr_exp
    '''
    if neuron_params == None:
        neuron_params = {  # for LIF sampling
            'cm': .2,
            'tau_m': .1,  # 1.,
            'v_thresh': -50.,
            'tau_syn_E': 10.,  # 30.,
            'v_rest': -50.,
            'tau_syn_I': 10.,  # 30.,
            'v_reset': -50.01,  # -50.1,
            'tau_refrac': 10.,  # 30.
            "i_offset": 0.,
        }

    if set_singlepara != None:
        for i in range(len(set_singlepara) / 2):
            neuron_params[set_singlepara[i * 2]] = set_singlepara[i * 2 + 1]

    if weight == None:
        weight = .02

    c_m = neuron_params['cm']
    tau_m = neuron_params['tau_m']

    tau_syn = neuron_params['tau_syn_E']
    v_rest = neuron_params['v_rest']

    scalfac = 1.  # to mV scale factor
    if tdur != None:
        tdur = np.arange(0, tdur, 0.1)  # 0.1 ms
    else:
        tdur = np.arange(0, 150, 0.1)  # 0.1 ms

    A_se = weight * (1. / (c_m * (1. / tau_syn - 1. / tau_m))) * \
        (np.exp(- tdur / tau_m) - np.exp(- tdur / tau_syn)) * scalfac

    if title == 'rec_weight':  # return the rectangular psp weight which will create the same under PSP area within tau_syn as the double exponential PSP
        lif_weight = weight
        rec_weight = lif_weight * (1. / (c_m * (1. - tau_syn / tau_m))) * (
            tau_syn * (np.exp(-1) - 1) - tau_m * (np.exp(- tau_syn / tau_m) - 1))
        return rec_weight

    tmax = np.log(tau_syn / tau_m) / (1 / tau_m - 1 / tau_syn)
    if plot == True:
        p.ion()
        ax.plot(A_se)
        print('Theo tmax: ', tmax)
        print('Theo Amax: ', weight * (1. / (c_m * (1. / tau_syn - 1. / tau_m))) * (p.exp(- tmax / tau_m) - p.exp(- tmax / tau_syn)) * scalfac)
    return A_se
