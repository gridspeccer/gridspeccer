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
    gs_main = gs.GridSpec(2, 2, hspace=0.34, wspace=0.30,
            left=0.08, right=0.92, top=.92, bottom=0.08)

    return {
            # these are needed for proper labelling
            # core.make_axes takes care of them

            "jointDklsP" : gs_main[0, 0],

            "inferenceDklsP" : gs_main[1, 0],

            "jointDklsS" : gs_main[0, 1],

            "inferenceDklsS" : gs_main[1, 1],
        }

def adjust_axes(axes):
    """
        Settings for all plots.
    """
    # TODO: Uncomment & decide for each subplot!
    for ax in axes.itervalues():
        core.hide_axis(ax)

    #for k in [
    #        "dummy1",
    #        "dummy2"
    #    ]:
    #    axes[k].set_frame_on(False)

def plot_labels(axes):
    core.plot_labels(axes,
        labels_to_plot=[
            "jointDklsP",
            "inferenceDklsP",
            "jointDklsS",
            "inferenceDklsS",
        ],
    #    label_ypos = {'delays_all': 0.95},
        label_xpos = {}
        )

def get_fig_kwargs():
    width = 12.
    alpha = 0.68
    return {"figsize": (width, alpha*width)}



###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

def plot_jointDklsP(ax):

    # set up the axes
    core.show_axis(ax)
    core.make_spines(ax)

    # load the joint Poisson DKL from fig4
    DKLfinalPoisson = core.get_data('fig4/fig4_DKLtimeValuePoisson.npy')
    refDKLs = DKLfinalPoisson[:, -1]

    # load the data
    dataPoisson = core.get_data('figAppendix1/poissonJointsDKL.npy')

    # gather the data into an array
    data = [dataPoisson[i,:] for i in range(len(dataPoisson[:,0]))]
    data = [refDKLs] + data

    # plot
    aux.plotBoxPlot(ax, data)
    #ax.text( 0.1, 1e-3, 'data for\nFig. 4B', ha='center')
    ax.set_xticks([])
    #ax.set_xticklabels(['fig 4. B'])
    ax.set_title('Poisson, joint', fontweight='bold')
    ax.set_ylim([7e-4, 2e0])

def plot_inferenceDklsP(ax):

    # set up the axes
    core.show_axis(ax)
    core.make_spines(ax)

    # load the joint Poisson DKL from fig4
    DKLfinalPoisson = core.get_data('fig4/fig5_dklTimeValuePoisson.npy')
    refDKLs = DKLfinalPoisson[:, -1]

    # load the data
    dataPoisson = core.get_data('figAppendix1/poissonInfDKL.npy')

    # gather the data into an array
    data = [dataPoisson[i,:] for i in range(len(dataPoisson[:,0]))]
    data = [refDKLs] + data

    # plot
    aux.plotBoxPlot(ax, data)
    #ax.text( 0.1, 3e-3, 'data for\nFig. 4E', ha='center')
    ax.set_xticks([])
    ax.set_title('Poisson, inference', fontweight='bold')
    ax.set_ylim([7e-4, 2e0])
    #ax.set_xticklabels(['fig4B'] + range(1,21))

def plot_jointDklsS(ax):


    # set up the axes
    core.show_axis(ax)
    core.make_spines(ax)

    # load the joint Poisson DKL from fig4
    DKLfinalPoisson = core.get_data('fig4/fig4_DKLtimeValueSon.npy')
    refDKLs = DKLfinalPoisson[:, -1]

    # load the data
    dataPoisson = core.get_data('figAppendix1/randomJointsDKL.npy')

    # gather the data into an array
    data = [dataPoisson[i,:] for i in range(len(dataPoisson[:,0]))]
    data = [refDKLs] + data

    # plot
    aux.plotBoxPlot(ax, data)
    #ax.text( 0.1, 4e-3, 'data for\nFig. 4B', ha='center')
    ax.set_xticks([])
    ax.set_title('RN, joint', fontweight='bold')
    ax.set_ylim([7e-4, 2e0])

def plot_inferenceDklsS(ax):

    # set up the axes
    core.show_axis(ax)
    core.make_spines(ax)

    # load the joint Poisson DKL from fig4
    DKLfinalPoisson = core.get_data('fig4/fig5_dklTimeValueSon.npy')
    refDKLs = DKLfinalPoisson[:, -1]

    # load the data
    dataPoisson = core.get_data('figAppendix1/randomInfDKL.npy')

    # gather the data into an array
    data = [dataPoisson[i,:] for i in range(len(dataPoisson[:,0]))]
    data = [refDKLs] + data

    # plot
    aux.plotBoxPlot(ax, data)
    #ax.text( 0.1, 5e-3, 'data for\nFig. 4E', ha='center')
    ax.set_xticks([])
    ax.set_title('RN, inference', fontweight='bold')
    ax.set_ylim([7e-4, 2e0])

