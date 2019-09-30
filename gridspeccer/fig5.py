#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib import gridspec as gs
from matplotlib import collections as coll
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
import matplotlib.pyplot as plt
import pylab as p
import copy
import numpy as np
from scipy.misc import imresize

from . import core
from .core import log
from . import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """
    # TODO: Adjust positioning
    gs_main = gs.GridSpec(3, 1, hspace=0.3,
                          height_ratios=[1.,1.,.7],
                          left=0.04, right=0.96, top=.97, bottom=0.05)
    gs_minor1 = gs.GridSpecFromSubplotSpec(1, 4, gs_main[0, 0], wspace=0.5,
                                         hspace=0.2,
                                         width_ratios=[1.,1.,1.,1.])
    gs_minor2 = gs.GridSpecFromSubplotSpec(1, 4, gs_main[1, 0], wspace=0.5,
                                         hspace=0.2,
                                         width_ratios=[1.,1.,1.,1.])
    gs_minorDummy = gs.GridSpecFromSubplotSpec(1, 2, gs_main[2, 0], wspace=0.1,
                                         hspace=0.2,
                                         width_ratios=[1.2,1.])
    #gs_minor3 = gs.GridSpecFromSubplotSpec(2, 10, gs_main[2, 0], wspace=0.25,
    #                                     hspace=0.03,
    #                                     width_ratios=[1.2,1.2,1.2,0.3,1.2,1.2,1.2,0.3,10.,1.])
    gs_minor3 = gs.GridSpecFromSubplotSpec(2, 8, gs_minorDummy[0, 0], wspace=0.28,
                                        hspace=0.2,
                                        width_ratios=[1.2,1.2,1.2,0.3,1.2,1.2,1.2,0.3])
    gs_minor4 = gs.GridSpecFromSubplotSpec(2, 2, gs_minorDummy[0, 1], wspace=0.04,
                                        hspace=0.01,
                                        height_ratios=[.5,1.],
                                        width_ratios=[1.,.01])
    #gs_minor6 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[4, 0], wspace=0.3,
    #                                    hspace=0.2,
    #                                    width_ratios=[1.,1.])
    #gs_minor7 = gs.GridSpecFromSubplotSpec(1, 2, gs_main[5, 0], wspace=0.3,
    #                                    hspace=0.2,
    #                                    width_ratios=[1.,1.])
    return {
        # these are needed for proper labelling
        # core.make_axes takes care of them

        "mnistExample": gs_minor1[0, 0],
        "fashionExample": gs_minor2[0, 0],

        "mnistIterError": gs_minor1[0,1],
        "fashionIterError": gs_minor2[0,1],

        "mnistOrigSandP": gs_minor3[0, 0],
        "mnistClampSandP": gs_minor3[0, 1],
        "mnistNetworkSandP": gs_minor3[0, 2],
        "mnistLabelSandP": gs_minor3[0,3],

        "mnistOrigPatch": gs_minor3[1, 0],
        "mnistClampPatch": gs_minor3[1, 1],
        "mnistNetworkPatch": gs_minor3[1, 2],
        "mnistLabelPatch": gs_minor3[1, 3],

        "fashionOrigSandP": gs_minor3[0, 4],
        "fashionClampSandP": gs_minor3[0, 5],
        "fashionNetworkSandP": gs_minor3[0, 6],
        "fashionLabelSandP": gs_minor3[0, 7],

        "fashionOrigPatch": gs_minor3[1, 4],
        "fashionClampPatch": gs_minor3[1, 5],
        "fashionNetworkPatch": gs_minor3[1, 6],
        "fashionLabelPatch": gs_minor3[1, 7],

        "compPics": gs_minor4[0,0],
        "compLabel": gs_minor4[1,0],
        "compColorBar": gs_minor4[0:,1],

        "mseMnist": gs_minor1[0,2],
        "mseFashion": gs_minor2[0,2],

        "errorMnist": gs_minor1[0,3],
        "errorFashion": gs_minor2[0,3],
    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.itervalues():
        core.hide_axis(ax)

    for k in [
        'mnistExample',
        'fashionExample',
        'compPics',
        'compColorBar',
    ]:
        axes[k].set_frame_on(False)

    # Share the y-axes of specific subplots
    for pair in [
    ]:
        axes[pair[0]].get_shared_x_axes().join(axes[pair[0]], axes[pair[1]])



def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "mnistExample",
                         "fashionExample",

                         "mnistIterError",
                         "fashionIterError",

                         "mseMnist",
                         "mseFashion",

                         "errorMnist",
                         "errorFashion",

                         "mnistOrigSandP",
                         "compPics",
                     ],
                     label_ypos={'mnistExample': .95,
                                 'fashionExample': .95,
                                 "mnistOrigSandP": .95,
                                 "compPics": .95},
                     label_xpos={'mnistExample': -.15,
                                 'fashionExample': -.25,
                                 "mnistOrigSandP": -.45,
                                 "compPics": -.08}
                     )


def get_fig_kwargs():
    width = 12.
    alpha = .7
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

def plot_mnistExample(ax):

    # load the data
    original = core.get_data('fig5/mnist_test_red.npy')

    # Do the actual plotting
    aux.plotExamplePictures(ax, original, (12,12), (28,28), (2,4),
                            indices=[3,4,0,2,5,13,1,6])
    ax.text(120., 20., 'original', rotation=90, fontsize=12)
    ax.text(120., 80., 'reduced', rotation=90, fontsize=12)

    ax.text(-28., 35., 'MNIST', weight='bold', fontsize=15, rotation=90)

    return

def plot_fashionExample(ax):

    # load the data
    original = core.get_data('fig5/fashion_test_red.npy')

    # Do the actual plotting
    aux.plotExamplePictures(ax, original, (12,12), (28,28), (2,3),
                            indices=[0,1,5,6,2,3])

    ax.text(-30., 30.,
            'F-MNIST',
            weight='bold',
            fontsize=15,
            rotation=90)
    ax.text(90., 20., 'original', rotation=90, fontsize=12)
    ax.text(90., 80., 'reduced', rotation=90, fontsize=12)

    return

def plot_fashionIterError(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # Load the data
    abstractRatio = core.get_data('fig5/fashionAbstract.npy')
    classRatio = core.get_data('fig5/fashionClassRatios.npy')
    iterNumb = core.get_data('fig5/fashionClassRatiosArray.npy')

    # Do the plot
    aux.plotITLTrainingError(ax, abstractRatio, classRatio, iterNumb)
    ax.set_ylim([-.04, .25])
    ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.01),
               fontsize=9)

    # Add inset with mixture matrix
    #iax = plt.axes([0, 0, 1, 1])
    #ip = InsetPosition(ax, [0.45, 0.25, 0.3, 0.7]) #posx, posy, width, height
    iax = inset_axes(ax, width = "50%", height= "50%", loc=1)

    # Load the data
    mixtureMatrix = core.get_data('fig5/fashionConfMatrix.npy')
    print("Mixture matrix of the fMNIST dataset: {}".format(mixtureMatrix))
    labels = ['T', 'Tr', 'S']

    # Do the plot
    iax.set_frame_on(True)
    aux.plotMixtureMatrix(iax, mixtureMatrix, labels)

    return

def plot_mnistIterError(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # Load the data
    abstractRatio = core.get_data('fig5/mnistAbstract.npy')
    classRatio = core.get_data('fig5/mnistClassRatios.npy')
    iterNumb = core.get_data('fig5/mnistClassRatiosArray.npy')

    # Do the plot
    aux.plotITLTrainingError(ax, abstractRatio, classRatio, iterNumb)
    ax.set_ylim([-.04, .25])
    ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.01),
               fontsize=9)

    # Add inset with mixture matrix
    #iax = plt.axes([0, 0, 1, 1])
    #ip = InsetPosition(ax, [0.45, 0.25, 0.3, 0.7]) #posx, posy, width, height
    #iax.set_axes_locator(ip)
    iax = inset_axes(ax, width = "50%", height= "50%", loc=1)

    # Load the data
    mixtureMatrix = core.get_data('fig5/mnistConfMatrix.npy')
    labels = [0, 1, 4, 7]

    # Do the plot
    iax.set_frame_on(True)
    aux.plotMixtureMatrix(iax, mixtureMatrix, labels)

    return

def plot_compPics(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)
    respImages = core.get_data('fig5/686respPatchMnistDyn.npy')

    # set up the parameters
    dt = 2
    tMin = 100.
    tMax = 350.
    tMinIndex = int(tMin/dt)
    tMaxIndex = int(tMax/dt)
    N_vertical = 1
    N_horizontal = 10
    N_all = N_vertical * N_horizontal
    half = 0
    frame = 1
    picSize = (24, 24)
    picSizeOrig = (12, 12)
    indices = np.floor(np.linspace(tMinIndex,tMaxIndex, N_all)).astype(int)
    respImages = respImages[indices,:]

    # Plot the upper 8 examples (originals)
    pic = np.ones((( N_vertical + 1) * frame + N_vertical * picSize[0] + half,
                   (N_horizontal + 1) * frame + N_horizontal * picSize[1])) * 255.
    for counter in xrange(N_vertical * N_horizontal):
        j = counter % N_horizontal
        i = int(counter / N_horizontal)
        picVec = respImages[counter, :]
        picCounter = np.reshape(picVec, picSizeOrig)
        picCounter = imresize(picCounter, picSize, interp='nearest')


        pic[(i + 1) * frame + i * picSize[0]: (i + 1) * frame + (i + 1) * picSize[0],
            (j + 1) * frame + j * picSize[1]: (j + 1) * frame + (j + 1) * picSize[1]] = picCounter

    ax.imshow(pic, cmap='Greys', aspect='equal')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_ylabel('visible', fontsize=10, labelpad=5)

    pass

def plot_compLabel(ax):

    core.show_axis(ax)
    labels = [0,1,4,7]
    N_labels = len(labels)
    dt = 2
    tMin = 100.
    tMax = 350.
    tMinIndex = int(tMin/dt)
    tMaxIndex = int(tMax/dt)

    labelImage = core.get_data('fig5/686labelPatchMnistDyn.npy')[tMinIndex:tMaxIndex,:]
    ax.imshow(np.flipud(labelImage.T),
             cmap=cm.gray_r,
             vmin=0., vmax=1.,
             aspect='auto',
             interpolation='nearest',
             extent=(-50., 200., -.5,N_labels - .5))
             #extent=(-.5, N_labels - .5,200.,-50.))
    ax.set_yticks(range(N_labels))
    ax.set_yticklabels(labels)
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(11)
    ax.tick_params(width=0, length=0)
    ax.set_xlabel(r'$t$ [ms]', fontsize=12)
    ax.set_ylabel('label', labelpad=0, fontsize=12)
    ax.axvline(x=0, ymin=-.02, ymax=1.58, color='tab:green', linestyle='--',
               clip_on=False, linewidth=2)

def plot_compColorBar(ax):

    cax = inset_axes(ax,
                 width="100%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc=3,
                 bbox_to_anchor=(0., 0.03, 1, .92),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
    cmap = mpl.cm.Greys
    norm = mpl.colors.Normalize(vmin=0., vmax=1.0)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                #norm=norm,
                                ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.])

    return

def plot_mseMnist(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    timeArray = core.get_data('fig5/fashionTimeArray.npy')
    mseMnistSandp = core.get_data('fig5/mnistMseSandp.npy')
    mseMnistPatch = core.get_data('fig5/mnistMsePatch.npy')
    abstractMnistSandp = core.get_data('fig5/mnistAbstractMseSandpMseArray.npy')
    abstractMnistPatch = core.get_data('fig5/mnistAbstractMsePatchMseArray.npy')

    aux.plotMse(ax, timeArray, mseMnistPatch, mseMnistSandp,
                abstractMnistPatch, abstractMnistSandp)

    ax.set_ylim([0.0, 0.56])

    return

def plot_mseFashion(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    timeArray = core.get_data('fig5/fashionTimeArray.npy')
    mseFashionSandp = core.get_data('fig5/fashionMseSandp.npy')
    mseFashionPatch = core.get_data('fig5/fashionMsePatch.npy')
    abstractFashionSandp = core.get_data('fig5/fashionAbstractMseSandpMseArray.npy')
    abstractFashionPatch = core.get_data('fig5/fashionAbstractMsePatchMseArray.npy')


    aux.plotMse(ax, timeArray, mseFashionPatch, mseFashionSandp,
                abstractFashionPatch, abstractFashionSandp)
    ax.set_ylim([0.0, 0.62])

    return

def plot_errorMnist(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    timeArray = core.get_data('fig5/fashionTimeArray.npy')
    errorMnistSandp = core.get_data('fig5/mnistAccSandp.npy')
    errorMnistPatch = core.get_data('fig5/mnistAccPatch.npy')
    errorMnistRef = core.get_data('fig5/mnistAccHW.npy')
    abstractMnistSandp = core.get_data('fig5/mnistAbstractMseSandp.npy')
    abstractMnistPatch = core.get_data('fig5/mnistAbstractMsePatch.npy')

    aux.plotErrorTime(ax, timeArray,
                          errorMnistPatch,
                          errorMnistSandp,
                          errorMnistRef,
                          abstractMnistPatch,
                          abstractMnistSandp)

    ax.set_ylim([0.0, 0.85])

    return

def plot_errorFashion(ax):

    # Set up the plot
    core.show_axis(ax)
    core.make_spines(ax)

    # load the data
    timeArray = core.get_data('fig5/fashionTimeArray.npy')
    errorFashionSandp = core.get_data('fig5/fashionAccSandp.npy')
    errorFashionPatch = core.get_data('fig5/fashionAccPatch.npy')
    errorFashionRef = core.get_data('fig5/fashionAccHW.npy')
    abstractFashionSandp = core.get_data('fig5/fashionAbstractMseSandp.npy')
    abstractFashionPatch = core.get_data('fig5/fashionAbstractMsePatch.npy')

    aux.plotErrorTime(ax, timeArray,
                          errorFashionPatch,
                          errorFashionSandp,
                          errorFashionRef,
                          abstractFashionPatch,
                          abstractFashionSandp)
    ax.set_ylim([0.0, 0.78])

    return


# MNIST
def plot_mnistOrigSandP(ax):

    image = core.get_data('fig5/145origSandpMnist.npy')
    aux.plotVisible(ax, image, (12,12), '')

def plot_mnistClampSandP(ax):

    image = core.get_data('fig5/145clampSandpMnist.npy')
    aux.plotClamping(ax, image, (12,12), '', mode='sandp')

def plot_mnistNetworkSandP(ax):

    image = core.get_data('fig5/145respSandpMnist.npy')
    aux.plotClamping(ax, image, (12,12), '')

def plot_mnistLabelSandP(ax):

    image = core.get_data('fig5/145labelSandpMnist.npy')
    aux.plotLabel(ax, image, [7,4,1,0],'')


def plot_mnistOrigPatch(ax):

    image = core.get_data('fig5/27origPatchMnist.npy')
    aux.plotVisible(ax, image, (12,12), 'O')

def plot_mnistClampPatch(ax):

    image = core.get_data('fig5/27clampPatchMnist.npy')
    aux.plotClamping(ax, image, (12,12), 'C', mode='patch')

def plot_mnistNetworkPatch(ax):

    image = core.get_data('fig5/27respPatchMnist.npy')
    aux.plotClamping(ax, image, (12,12), 'R')

def plot_mnistLabelPatch(ax):

    image = core.get_data('fig5/27labelPatchMnist.npy')
    aux.plotLabel(ax, image, [7,4,1,0],'L')


# fashion

def plot_fashionOrigSandP(ax):

    image = core.get_data('fig5/37origSandpFashion.npy')
    aux.plotVisible(ax, image, (12,12), '')

def plot_fashionClampSandP(ax):

    image = core.get_data('fig5/37clampSandpFashion.npy')
    aux.plotClamping(ax, image, (12,12), '', mode='sandp')

def plot_fashionNetworkSandP(ax):

    image = core.get_data('fig5/37respSandpFashion.npy')
    aux.plotClamping(ax, image, (12,12), '')

def plot_fashionLabelSandP(ax):

    image = core.get_data('fig5/37labelSandpFashion.npy')
    aux.plotLabel(ax, image, ['S', 'Tr', 'T'],'')


def plot_fashionOrigPatch(ax):

    image = core.get_data('fig5/4origPatchFashion.npy')
    aux.plotVisible(ax, image, (12,12), 'O')

def plot_fashionClampPatch(ax):

    image = core.get_data('fig5/4clampPatchFashion.npy')
    aux.plotClamping(ax, image, (12,12), 'C', mode='patch')

def plot_fashionNetworkPatch(ax):

    image = core.get_data('fig5/4respPatchFashion.npy')
    aux.plotClamping(ax, image, (12,12), 'R')

def plot_fashionLabelPatch(ax):

    image = core.get_data('fig5/4labelPatchFashion.npy')
    aux.plotLabel(ax, image, ['S', 'Tr', 'T'],'L')
