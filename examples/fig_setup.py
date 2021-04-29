#!/usr/bin/env python
# encoding: utf-8
"""Example script for a plot using the gridspeccer module.

Neccessary functions are get_gridspec, adjust_axes, plot_labels and get_fig_kwargs.
The actual plot functions are below and are named in accordance to the names in
the get_gridspec function.
"""

from matplotlib import gridspec as gs
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from gridspeccer import core
from gridspeccer.core import log
from gridspeccer import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """

    gs_main = gs.GridSpec(1, 1,
                          left=0.05, right=0.97, top=0.95, bottom=0.08)

    gs_schematics = gs.GridSpecFromSubplotSpec(2, 1, gs_main[0, 0],
                                               height_ratios=[1.5, 2.0],
                                               hspace=0.2)
    gs_tikz = gs.GridSpecFromSubplotSpec(1, 2, gs_schematics[1, 0],
                                         width_ratios=[2, 1])
    gs_lower_row = gs.GridSpecFromSubplotSpec(1, 2, gs_schematics[0, 0],
                                              wspace=0.35,
                                              width_ratios=[1.5, 1])
    gs_membranes = gs.GridSpecFromSubplotSpec(2, 1, gs_lower_row[0, 1],
                                              hspace=0.2)

    return {
        # ### schematics
        "arch": (gs_tikz[0, 0], {'3d': True}),

        "coding": gs_tikz[0, 1],

        "psp_shapes": gs_lower_row[0, 0],

        "membrane_schematic_0": gs_membranes[0, 0],
        "membrane_schematic_1": gs_membranes[1, 0],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for axis in list(axes.values()):
        core.hide_axis(axis)

    for k in [
        "arch",
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    """Naming of gridspecs plus placement"""
    core.plot_labels(axes,
                     labels_to_plot=[
                         "psp_shapes",
                         "membrane_schematic_0",
                         "arch",
                         "coding",
                     ],
                     label_ypos={
                         "arch": 0.0,
                         "coding": 0.87,
                         "psp_shapes": 0.95,
                     },
                     label_xpos={
                         "arch": 0.0,
                     },
                     label_zpos={
                         "arch": 7.5,
                     },
                     )


def get_fig_kwargs():
    """figure specification"""
    width = 7.12
    alpha = 10. / 12. / 1.4 / 0.8
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(axis)
#
# axis is the Axes to plot into
#
xlim = (0, 100)

NAME_OF_VLEAK = "$E_{L}$"
NAME_OF_VTH = "$\\vartheta$"
EXAMPLECLASS = 1


def plot_arch(axis):
    """done with tex"""
    log.info("the architecture is done with 'tex' for %s", axis)
    print((aux.cm_gray_r))  # access aux once


def plot_coding(axis):
    """arrows for coding plot"""
    # make the axis
    core.show_axis(axis)
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_xlabel("time [a.u.]")
    axis.xaxis.set_label_coords(.5, -0.05)
    axis.set_ylabel("neuron id")
    axis.yaxis.set_ticks_position('none')
    axis.xaxis.set_ticks_position('none')
    axis.set_xticks([])
    axis.set_yticks([])

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    # #################### draw arrow instead of y axis to have arrow there
    # get width and height of axes object to compute
    # matching arrowhead length and width
    # fig = plt.gcf()
    # dps = fig.dpi_scale_trans.inverted()

    # bbox = axis.get_window_extent()  # .transformed(dps)
    # width, height = bbox.width, bbox.height

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # manual arrowhead width and length
    head_width = 1. / 20. * (ymax - ymin)
    head_length = 1. / 20. * (xmax - xmin)
    line_width = 0.5  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    # yhw = head_width / (ymax - ymin) * (xmax - xmin) * height / width
    # yhl = head_length / (xmax - xmin) * (ymax - ymin) * width / height

    axis.arrow(xmin, ymin, xmax - xmin, 0, fc='k', ec='k', lw=line_width,
               head_width=head_width, head_length=head_length, overhang=ohg,
               length_includes_head=True, clip_on=False)
    return 0


def plot_frame(axis):
    """Plotting frame around an axis, possibly with background fill"""
    extent_left = 0.05
    extent_right = 0.0
    extent_top = -0.010
    extent_bottom = 0.26
    fancybox = mpatches.Rectangle(
        (-extent_left, -extent_bottom),
        1 + extent_left + extent_right, 1 + extent_bottom + extent_top,
        facecolor="black", fill=True, alpha=0.07,  # zorder=zorder,
        transform=axis.transAxes)
    plt.gcf().patches.append(fancybox)


def membrane_schematic(axis, should_spike, x_annotated=False, y_annotated=False):
    """schematic of a membrane trace"""
    # make the axis
    core.show_axis(axis)
    core.make_spines(axis)

    ylim = (-0.2, 1.2)

    xvals = np.linspace(0., 100., 200)
    c_m = 0.2
    t_s = 10.
    t_m = t_s
    weight = 0.032
    t_i1 = 20.
    v_th = 1.

    def theta(value):
        return value > 0

    if should_spike:
        t_i2 = 30.
        t_spike = 35.

        def voltage(time, before_spike):
            return weight / c_m * (
                theta(time - t_i1) * np.exp(-(time - t_i1) / t_m) * (time - t_i1) +
                theta(time - t_i2) * np.exp(-(time - t_i2) / t_m) * (time - t_i2)
            ) * ((time < t_spike) * before_spike + (time > t_spike) * (1 - before_spike))

        axis.plot(xvals, voltage(xvals, True), color='black')
        axis.plot(xvals[xvals > t_spike], voltage(xvals[xvals > t_spike], False), color='black', linestyle="dotted")
    else:
        t_i2 = 35.

        def voltage(time):
            return weight / c_m * (
                theta(time - t_i1) * np.exp(-(time - t_i1) / t_m) * (time - t_i1) +
                theta(time - t_i2) * np.exp(-(time - t_i2) / t_m) * (time - t_i2)
            )

        axis.plot(xvals, voltage(xvals), color='black')

    axis.axhline(v_th, linewidth=1, linestyle='dashed', color='black', alpha=0.6)

    input_arrow_height = 0.17
    arrow_head_width = 2.47
    arrow_head_length = 0.04
    for spk in [t_i1, t_i2]:
        axis.arrow(spk, ylim[0], 0, input_arrow_height,
                   color="black",
                   head_width=arrow_head_width,
                   head_length=arrow_head_length,
                   length_includes_head=True,
                   zorder=-1)

    # output spike
    if should_spike:
        t_out = 34.5
        axis.arrow(t_out, ylim[1], 0, -input_arrow_height,
                   color="black",
                   head_width=arrow_head_width,
                   head_length=arrow_head_length,
                   length_includes_head=True,
                   zorder=-1)

    axis.set_yticklabels([])
    if y_annotated:
        axis.set_ylabel("membrane voltage")
        axis.yaxis.set_label_coords(-0.2, 1.05)

    axis.set_yticks([v_th, 0])
    axis.set_yticklabels([NAME_OF_VTH, NAME_OF_VLEAK])
    axis.set_ylim(ylim)

    if x_annotated:
        axis.set_xlabel("time [a. u.]")
        axis.xaxis.set_label_coords(.5, -0.15)
    axis.set_xticks([])


def plot_membrane_schematic_0(axis):
    """first membrane schematic"""
    membrane_schematic(axis, False, x_annotated=False, y_annotated=False)


def plot_membrane_schematic_1(axis):
    """second membrane schematic"""
    membrane_schematic(axis, True, x_annotated=True, y_annotated=True)


def plot_name(axis, name):
    """plot only a name"""
    core.show_axis(axis)
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['right'].set_visible(False)

    axis.yaxis.set_ticks_position('none')
    axis.xaxis.set_ticks_position('none')
    axis.set_xticks([])
    axis.set_yticks([])

    axis.set_xlabel(name, fontsize=13)
    axis.xaxis.set_label_coords(.30, 0.50)


def plot_psp_shapes(axis):
    "Plotting psp shapes"""
    # make the axis
    core.show_axis(axis)
    core.make_spines(axis)

    xvals = np.linspace(0., 100., 200)
    t_s = 10.
    t_i = 15.

    def theta(value):
        return value > 0

    def voltage(time, t_m):
        factor = 1.
        if t_m < t_s:
            factor = 6. / t_m

        if t_m != t_s:
            ret_val = factor * t_m * t_s / (t_m - t_s) * theta(time - t_i) * \
                (np.exp(-(time - t_i) / t_m) - np.exp(-(time - t_i) / t_s))
        else:
            ret_val = factor * theta(time - t_i) * np.exp(-(time - t_i) / t_m) * (time - t_i)

        ret_val[time < t_i] = 0.
        return ret_val

    taums = [100000000., 20, 10, 0.0001]
    taums_name = [r"\rightarrow \infty", "= 2", "= 1", r"\rightarrow 0"]
    colours = ['C7', 'C8', 'C9', 'C6']
    for t_m, t_m_name, col in zip(taums, taums_name, colours):
        # axis.set_xlabel(r'$\tau_\mathrm{m}$ [ms]')
        lab = "$" + r'\tau_\mathrm{{m}} / \tau_\mathrm{{s}} {}'.format(t_m_name) + "$"
        axis.plot(xvals, voltage(xvals, t_m), color=col, label=lab)

    axis.set_yticks([])
    axis.set_xticks([])

    axis.set_ylabel("PSPs [a. u.]")
    axis.yaxis.set_label_coords(-0.03, 0.5)

    axis.set_xlabel("time [a. u.]")
    axis.xaxis.set_label_coords(.5, -0.075)

    axis.legend()
