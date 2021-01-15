#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib import gridspec as gs
from matplotlib import collections as coll
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
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
	gs_main = gs.GridSpec(3, 1, hspace=0.3, height_ratios=[2.,3., 5],
						  left=0.07, right=0.97, top=.95, bottom=0.05)
	gs_top = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.3,
										width_ratios=[1., 1.5])
	gs_mid = gs.GridSpecFromSubplotSpec(2, 4, gs_main[1, 0], wspace=0.5,
										width_ratios=[.7, 1.5, .4, .7])
	gs_b = gs.GridSpecFromSubplotSpec(4, 8, gs_main[2, 0], wspace=0.25,
										 hspace=0.2,
										 width_ratios=[1.2,1.2,1.2,0.3,1.2,1.2,1.2,0.3])

	return {
		# these are needed for proper labelling
		# core.make_axes takes care of them

		"classRatio": gs_top[0, 0],

		"mseTime": gs_top[0, 1],

		"orig": gs_mid[0,0],
		"clamped": gs_mid[1,0],
		"resp": gs_mid[:2,1:3],
		"label": gs_mid[:2,3],

		"mnistOrig1": gs_b[0, 0],
		"mnistClamp1": gs_b[0, 1],
		"mnistNetwork1": gs_b[0, 2],
		"mnistLabel1": gs_b[0,3],

		"mnistOrig2": gs_b[0, 4],
		"mnistClamp2": gs_b[0, 5],
		"mnistNetwork2": gs_b[0, 6],
		"mnistLabel2": gs_b[0,7],

		"mnistOrig3": gs_b[1, 0],
		"mnistClamp3": gs_b[1, 1],
		"mnistNetwork3": gs_b[1, 2],
		"mnistLabel3": gs_b[1,3],

		"mnistOrig4": gs_b[1, 4],
		"mnistClamp4": gs_b[1, 5],
		"mnistNetwork4": gs_b[1, 6],
		"mnistLabel4": gs_b[1,7],

		"fashionOrig1": gs_b[2, 0],
		"fashionClamp1": gs_b[2, 1],
		"fashionNetwork1": gs_b[2, 2],
		"fashionLabel1": gs_b[2,3],

		"fashionOrig2": gs_b[2, 4],
		"fashionClamp2": gs_b[2, 5],
		"fashionNetwork2": gs_b[2, 6],
		"fashionLabel2": gs_b[2,7],

		"fashionOrig3": gs_b[3, 0],
		"fashionClamp3": gs_b[3, 1],
		"fashionNetwork3": gs_b[3, 2],
		"fashionLabel3": gs_b[3,3],

		"fashionOrig4": gs_b[3, 4],
		"fashionClamp4": gs_b[3, 5],
		"fashionNetwork4": gs_b[3, 6],
		"fashionLabel4": gs_b[3,7],
	}


def adjust_axes(axes):
	"""
		Settings for all plots.
	"""
	for ax in axes.values():
		core.hide_axis(ax)

	for k in [
		'resp'
	]:
		axes[k].set_frame_on(False)


def plot_labels(axes):
	core.plot_labels(axes,
					 labels_to_plot=[
						 "classRatio",
						 "mseTime",
						 "orig",
						 "mnistOrig1",
					 ],
					 label_ypos={'classRatio': 1.1,
					 			 'mseTime': 1.1,
					 			 'orig': 1.1,
					 			 'mnistOrig1': 1.1},
					 label_xpos={'classRatio': -.2,
					             'mseTime': -.2,
					             'orig': -.45,
					             'mnistOrig1': -.6}
					 )


def get_fig_kwargs():
	width = 6.
	alpha = 9. / 6.16
	return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#

def plot_classRatio(ax):

	# Set up the plot
	core.show_axis(ax)
	core.make_spines(ax)

	# load the data
	mnistAbstract = core.get_data('figPattern/mnistAbstract.npy')
	mnistHWRef = core.get_data('figPattern/mnistHWRef.npy')
	mnistSandp = core.get_data('figPattern/mnistSandp.npy')
	mnistPatch = core.get_data('figPattern/mnistPatch.npy')
	fashionAbstract = core.get_data('figPattern/fashionAbstract.npy')
	fashionHWRef = core.get_data('figPattern/fashionHWRef.npy')
	fashionSandp = core.get_data('figPattern/fashionSandp.npy')
	fashionPatch = core.get_data('figPattern/fashionPatch.npy')

	# Set up the arrays to plot the bar diagram
	gapSize = 0.4
	x = [np.array([0.,2.]) + gapSize*x for x in range(4)]
	y = [np.array([np.mean(mnistAbstract), np.mean(fashionAbstract)]),
		 np.array([mnistHWRef, fashionHWRef]),
		 np.array([mnistSandp, fashionSandp]),
		 np.array([mnistPatch, fashionPatch])
		]
	colors = ['blue', 'red', 'green', 'xkcd:mustard']
	labels = ['software', 'hardware', 'S&P', 'Patch']

	for index in range(4):
		ax.bar(x[index], y[index], width=0.35,
			   label=labels[index], color=colors[index])

	# Set up the labels
	ax.set_ylabel('classification ratio [1]')
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	ax.tick_params(length=0., pad=5)
	#ax.set_xticks([0.5*(x[1][0] + x[2][0]), 0.5*(x[1][1] + x[2][1])])
	ax.set_xticks([x[1][0] + 0.5 * gapSize, x[1][1] + 0.5 * gapSize])
	ax.set_xticklabels(['MNIST', 'fashion\nMNIST'], fontsize=10)
	ax.tick_params(axis='both', which='minor')
	ax.set_xlim([-.4, 3.6])
	ax.axhline(y=1.,
               xmin=-.4,
               xmax=3.6,
               linewidth=.2,
               linestyle='--',
               color='k')
	ax.set_ylim([0., 1.2])
	center = 0.5 * (x[3][0] + 0.35 + x[0][1])
	ax.legend(loc='upper center', fontsize=8, ncol=2, bbox_to_anchor=(.5, 1.2))

	pass


def plot_mseTime(ax):

	# Set up the plot
	core.show_axis(ax)
	core.make_spines(ax)

	# load the data
	timeArray = core.get_data('figPattern/fashionTimeArray.npy')
	mseFashionSandp = core.get_data('figPattern/fashionMseSandp.npy')
	mseFashionPatch = core.get_data('figPattern/fashionMsePatch.npy')
	mseMnistSandp = core.get_data('figPattern/mnistMseSandp.npy')
	mseMnistPatch = core.get_data('figPattern/mnistMsePatch.npy')

	# set up the data
	datas = [mseFashionPatch, mseFashionSandp, mseMnistPatch, mseMnistSandp]
	labels = ['fashion Patch', 'fashion S&P', 'MNIST Patch', 'MNIST S&P']
	colors = ['xkcd:crimson', 'xkcd:forest green', 'xkcd:coral', 'xkcd:lime']
	timeArray = timeArray - 150.

	# do the plotting
	for index in range(4):
		data = datas[index]

		median = np.median(data, axis=0)
		value75 = np.percentile(data, 75, axis=0)
		value25 = np.percentile(data, 25, axis=0)
		ax.plot(timeArray, median, linewidth=1.5, color=colors[index],
			label=labels[index])
		ax.fill_between(timeArray,
						value25,
						value75,
						color=colors[index],
						alpha=0.2,
						linewidth=0.0,
						)

	# annotate the plot
	ax.set_xlabel(r'$t$ [ms]', labelpad=5)
	ax.set_ylabel('mean squared\nerror [1]')
	ax.set_xlim([-100., 220.])
	ax.legend(fontsize=8)

	pass


def plot_orig(ax):

	image = core.get_data('figPattern/686origPatchMnistDyn.npy')
	aux.plotVisible(ax, image, (12,12), 'original')

def plot_clamped(ax):

	image = core.get_data('figPattern/686clampPatchMnistDyn.npy')
	aux.plotClamping(ax, image, (12,12), 'clamping')

def plot_resp(ax):

	# Set up the plot
	core.show_axis(ax)
	core.make_spines(ax)
	respImages = core.get_data('figPattern/686respPatchMnistDyn.npy')

	# set up the parameters
	dt = 2
	tMin = 100.
	tMax = 350.
	tMinIndex = int(tMin/dt)
	tMaxIndex = int(tMax/dt)
	N_vertical = 5
	N_horizontal = 8
	N_all = N_vertical * N_horizontal
	half = 0
	frame = 1
	picSize = (24, 24)
	picSizeOrig = (12, 12)
	indices = np.round(np.linspace(tMinIndex,tMaxIndex), N_all).astype(int)
	respImages = respImages[indices,:]

	# Plot the upper 8 examples (originals)
	pic = np.ones((( N_vertical + 1) * frame + N_vertical * picSize[0] + half,
				   (N_horizontal + 1) * frame + N_horizontal * picSize[1])) * 255.
	for counter in range(N_vertical * N_horizontal):
		j = counter % N_horizontal
		i = int(counter / N_horizontal)
		picVec = respImages[counter, :]
		picCounter = np.reshape(picVec, picSizeOrig)
		picCounter = imresize(picCounter, picSize, interp='nearest')


		pic[(i + 1) * frame + i * picSize[0]: (i + 1) * frame + (i + 1) * picSize[0],
			(j + 1) * frame + j * picSize[1]: (j + 1) * frame + (j + 1) * picSize[1]] = picCounter

	ax.imshow(pic, cmap='Greys', aspect='auto')
	ax.set_xticks([], [])
	ax.set_yticks([], [])
	ax.set_xlabel('network response', labelpad=5)
	
	pass

def plot_label(ax):

	core.show_axis(ax)
	labels = [0,1,4,7]
	N_labels = len(labels)
	dt = 2
	tMin = 100.
	tMax = 350.
	tMinIndex = int(tMin/dt)
	tMaxIndex = int(tMax/dt)

	labelImage = core.get_data('figPattern/686labelPatchMnistDyn.npy')[tMinIndex:tMaxIndex,:]
	ax.imshow(labelImage,
			 cmap=cm.gray_r,
			 vmin=0., vmax=1.,
			 aspect='auto',
			 interpolation='nearest',
			 extent=(-.5, N_labels - .5,200.,-50.))
	ax.set_xticks(list(range(N_labels)))
	ax.set_xticklabels(labels)
	for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
	ax.tick_params(width=0, length=0)
	ax.set_ylabel(r'$t$ [ms]')
	ax.set_xlabel('label response', labelpad=0)

# MNIST
def plot_mnistOrig1(ax):

	image = core.get_data('figPattern/145origSandpMnist.npy')
	aux.plotVisible(ax, image, (12,12), '')

def plot_mnistClamp1(ax):

	image = core.get_data('figPattern/145clampSandpMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistNetwork1(ax):

	image = core.get_data('figPattern/145respSandpMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistLabel1(ax):

	image = core.get_data('figPattern/145labelSandpMnist.npy')
	aux.plotLabel(ax, image, [7,4,1,0],'')


def plot_mnistOrig2(ax):

	image = core.get_data('figPattern/75origSandpMnist.npy')
	aux.plotVisible(ax, image, (12,12), '')

def plot_mnistClamp2(ax):

	image = core.get_data('figPattern/75clampSandpMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistNetwork2(ax):

	image = core.get_data('figPattern/75respSandpMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistLabel2(ax):

	image = core.get_data('figPattern/75labelSandpMnist.npy')
	aux.plotLabel(ax, image, [7,4,1,0],'')


def plot_mnistOrig3(ax):

	image = core.get_data('figPattern/27origPatchMnist.npy')
	aux.plotVisible(ax, image, (12,12), '')

def plot_mnistClamp3(ax):

	image = core.get_data('figPattern/27clampPatchMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistNetwork3(ax):

	image = core.get_data('figPattern/27respPatchMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistLabel3(ax):

	image = core.get_data('figPattern/27labelPatchMnist.npy')
	aux.plotLabel(ax, image, [7,4,1,0],'')


def plot_mnistOrig4(ax):

	image = core.get_data('figPattern/110origPatchMnist.npy')
	aux.plotVisible(ax, image, (12,12), '')

def plot_mnistClamp4(ax):

	image = core.get_data('figPattern/110clampPatchMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistNetwork4(ax):

	image = core.get_data('figPattern/110respPatchMnist.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_mnistLabel4(ax):

	image = core.get_data('figPattern/110labelPatchMnist.npy')
	aux.plotLabel(ax, image, [7,4,1,0],'')


# fashion

def plot_fashionOrig1(ax):

	image = core.get_data('figPattern/37origSandpFashion.npy')
	aux.plotVisible(ax, image, (12,12), '')

def plot_fashionClamp1(ax):

	image = core.get_data('figPattern/37clampSandpFashion.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_fashionNetwork1(ax):

	image = core.get_data('figPattern/37respSandpFashion.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_fashionLabel1(ax):

	image = core.get_data('figPattern/37labelSandpFashion.npy')
	aux.plotLabel(ax, image, ['S', 'Tr', 'T'],'')

def plot_fashionOrig2(ax):

	image = core.get_data('figPattern/7origSandpFashion.npy')
	aux.plotVisible(ax, image, (12,12), '')

def plot_fashionClamp2(ax):

	image = core.get_data('figPattern/7clampSandpFashion.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_fashionNetwork2(ax):

	image = core.get_data('figPattern/7respSandpFashion.npy')
	aux.plotClamping(ax, image, (12,12), '')

def plot_fashionLabel2(ax):

	image = core.get_data('figPattern/7labelSandpFashion.npy')
	aux.plotLabel(ax, image, ['S', 'Tr', 'T'],'')

def plot_fashionOrig3(ax):

	image = core.get_data('figPattern/4origPatchFashion.npy')
	aux.plotVisible(ax, image, (12,12), 'O')

def plot_fashionClamp3(ax):

	image = core.get_data('figPattern/4clampPatchFashion.npy')
	aux.plotClamping(ax, image, (12,12), 'C')

def plot_fashionNetwork3(ax):

	image = core.get_data('figPattern/4respPatchFashion.npy')
	aux.plotClamping(ax, image, (12,12), 'R')

def plot_fashionLabel3(ax):

	image = core.get_data('figPattern/4labelPatchFashion.npy')
	aux.plotLabel(ax, image, ['S', 'Tr', 'T'],'L')

def plot_fashionOrig4(ax):

	image = core.get_data('figPattern/53origPatchFashion.npy')
	aux.plotVisible(ax, image, (12,12), 'O')

def plot_fashionClamp4(ax):

	image = core.get_data('figPattern/53clampPatchFashion.npy')
	aux.plotClamping(ax, image, (12,12), 'C')

def plot_fashionNetwork4(ax):

	image = core.get_data('figPattern/53respPatchFashion.npy')
	aux.plotClamping(ax, image, (12,12), 'R')

def plot_fashionLabel4(ax):

	image = core.get_data('figPattern/53labelPatchFashion.npy')
	aux.plotLabel(ax, image, ['S', 'Tr', 'T'],'L')