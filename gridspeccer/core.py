#!/usr/bin/env python2
# encoding: utf-8

"""
    Core utils for all figures.
"""

import itertools as it
import numpy as np
import os
import os.path as osp
import pylab as p
import string
import sys

np.random.seed(424242)


def make_log(name):
    import logging
    import os
    log = logging.Logger(name)
    if "DEBUG" in os.environ:
        log_level = logging.DEBUG
        log_formatter = logging.Formatter("%(asctime)s %(levelname)s: "
            "%(message)s", datefmt="%y-%m-%d %H:%M:%S")
    else:
        log_level = logging.INFO
        log_formatter = logging.Formatter("%(asctime)s %(levelname)s: "
            "%(message)s", datefmt="%y-%m-%d %H:%M:%S")
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(log_formatter)
    log.addHandler(log_handler)
    log.setLevel(log_level)
    return log
log = make_log("gridspeccer")

# in order to have consistent coloring, please have all colors be defined here
label_to_color = {
    # preliminary:
    "sw" : "k",
    "sw+hw" : "m", #"r",
    "hw_all" : "g",
    "hw_single" : "b",
}

layer_to_color = {
    "hidden" : "orange",
    "label" : "blue",
    "bad" : "red",
}
layer_alpha = 0.75

dataset_to_color = {
    "train" : "g",
    "test" : "b",
    "train_theo" : "darkgreen",
    "test_theo" : "#062A78",
    #  "hw_mean" : "deepskyblue", 
    #  "hw_span" : "aqua",
    "hw_mean" :  "#A67B5B",
    "hw_span" : "wheat",
}

def make_figure(name):
    log.info("--- Creating figure: {} ---".format(name))

    plotscript = get_plotscript(name)

    class FigureNotDone(Exception):
        pass

    def throw_figure_not_done():
        raise FigureNotDone

    try:
        gs = getattr(plotscript, "get_gridspec",
                lambda: throw_figure_not_done())()
    except FigureNotDone:
        log.error("Work on {} hasn't even started yet, sheesh!".format(name))
        return

    fig_kwargs = getattr(plotscript, "get_fig_kwargs", lambda: {})()

    fig, axes = make_axes(gs, fig_kwargs=fig_kwargs)

    # call possible axes adjustment script for figure
    getattr(plotscript, "adjust_axes", lambda axes: None)(axes)

    for k, ax in axes.items():
        log.info("Plotting subfigure: {}".format(k))
        getattr(plotscript, "plot_{}".format(k), lambda ax: log.warn(
            "Plotscript missing for subplot <{}> in figure <{}>!"
            "".format(k, name)))(ax)

    log.info("Plotting labels…")
    getattr(plotscript, "plot_labels", lambda axes: log.warn(
        "Not plotting labels for figure {}".format(name)))(axes)

    save_figure(fig, name)
    p.close(fig)


def get_plotscript(name):
    try:
        sys.path.append(os.getcwd())
        import importlib
        plotscript = importlib.import_module(name)
    except ImportError:
        log.error("Plotscript for figure {} not found!".format(name))
        raise
    sys.path.pop(-1)
    return plotscript


def save_figure(fig, name):
    if not osp.isdir(osp.join("..", "fig")):
        os.makedirs(osp.join("..", "fig"))
    fig.savefig(osp.join("..", "fig", f"{name}.pdf"))


def make_axes(gridspec, fig_kwargs=None):
    """
        Turn gridspec information into plots.
    """
    if fig_kwargs is None:
        fig_kwargs = {}

    fig = p.figure(**fig_kwargs)
    axes = {}

    for k, gs in gridspec.items():
        # we just add a label to make sure all axes are actually created
        log.debug("Creating subplot: {}".format(k))
        axes[k] = fig.add_subplot(gs, label=k)

    return fig, axes


def get_data(filename):
    return np.load(osp.join("..", "data", filename))

def plot_labels(axes, labels_to_plot, xpos_default=.04, ypos_default=.90,
        label_xpos={}, label_ypos={}, label_color={}, fontdict={}):

    for l, c in zip((l for l in labels_to_plot),
            string.ascii_lowercase):
        log.info("Subplot {0} receives label {1}".format( l, c ))
        c = c
        plot_caption(axes[l], "\\textbf{" + c + "}",
                label_xpos.get(l, xpos_default),
                label_ypos.get(l, ypos_default), label_color.get(l, "k"),
                fontdict=fontdict)

def plot_caption(ax, caption, xpos=.04, ypos=.88, color="k", fontdict={}):
    # find out how our caption will look in reality
    caption_args={
            "ha" : "left", "va" : "bottom",
            #"weight" : "bold",
            "style" : "normal",
            "size" : 16,
            "color": color,
            "zorder" : 1000,
        }
    # r = get_renderer(ax.figure)
    # bb = t.get_window_extent(renderer=r)

    # if fontdict is None:
        # fontdict = {"family": "Linux Biolinum Kb"}
        # size = caption_args["size"]
        # bbox = mpatches.FancyBboxPatch(ax.transAxes.transform((xpos, ypos)),
                # size *1.0, size*1.0, zorder=10,
                # edgecolor="k", facecolor="r", boxstyle="round")
        # ax.patches.append(bbox)

    t = ax.text(xpos, ypos, caption, fontdict=fontdict, transform=ax.transAxes,
            **caption_args )

def hide_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def show_axis(ax):
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

def hide_ticks(ax, axis='both', minormajor='both'):
    ax.tick_params(axis=axis, which=minormajor,length=0)

def make_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def make_arrow(ax, pos_from, pos_to, color="r", arrowstyle="<|-|>",
        shrinkA=0., shrinkB=0., transform="data"):
    ax.annotate("", xy=pos_to, xytext=pos_from,
            xycoords=transform, textcoords=transform,
            arrowprops=dict(arrowstyle=arrowstyle, color=color,
                shrinkA=shrinkA, shrinkB=shrinkB))

def make_arrow_lines(ax, xpos, xlength, ypos, color="r", arrowstyle="<|-|>",
        line_alpha=0.75, text_ypos_adjustment=0., text_va="center", text=""):

    make_arrow(ax, (xpos, ypos), (xpos+xlength, ypos), color=color, arrowstyle=arrowstyle)

    ax.text(xpos+xlength/2., ypos - text_ypos_adjustment, text,
            va=text_va, color="r", ha="center")

    ax.axvline(x=xpos, ls="-", alpha=line_alpha, color=color)
    ax.axvline(x=xpos+xlength, ls="-", alpha=line_alpha, color=color)


