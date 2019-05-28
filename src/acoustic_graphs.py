#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

import numpy as np

import scipy.fftpack

import os
from os import listdir
from os.path import isfile, join

import cv2

def plot_acoustic_signal_and_spectrum(df, chip_size, i, basedir, col = 'acoustic_data', for_humans = False):
	counter = int(i / chip_size)
	chip = df.iloc[i:i+chip_size, :]

	#print(chip)
	print(i, len(chip))

	#basedir  = '/mnt/ros-data/datasets/LANL-Earthquake-Prediction/acoustic-graphs-for-conv2D-chip-size-' + str(len(chip.index))
	plot_acoustic_signal	(chip, counter, basedir, col, for_humans)
	plot_acoustic_spectrum	(chip, counter, basedir, col, for_humans)

	print(f'Created graph no. {counter} of {int(len(df.index)/chip_size)}')

	'''
	Parallel(n_jobs=8*4, prefer='threads')(
		delayed(_append_features_wrapper)(data, aggregate_length, do_fft, do_stft, i, stat_summary, include_y, windows_list)
		for i in range(0, len(training_set), chip_size))
	'''

	'''
	for i in range(0, len(training_set), chip_size):
		abs_counter = int(i / chip_size)
		chip = df.iloc[i:i+chip_size, :]
		print(chip)
		print(i)
		print(len(chip))

		plot_acoustic_signal_and_spectrum(chip, abs_counter)

		print(f'Graph no. {abs_counter}')
		abs_counter += 1
	'''

def plot_acoustic_signal(chip, counter, basedir, col = 'acoustic_data', for_humans = False):
	#ax = plt.gca()
	ax = plt.axes()

	if for_humans:
		ax.set_xlabel("Test Sample")
		ax.set_ylabel("Value")
		ax.legend(col)
		plt.grid(True)
	else:
		plt.axis('off')
		ax.legend().remove()

	ax.set_ylim((-4000, 4000))

	chip[col].plot(kind='line', lw=1, ax=ax, sharex=True)
	#ax = plt.gca()

	if for_humans:
		chip.iloc[ : , -1].plot(kind='line', lw=1, ax=ax, sharex=True)
		mng = plt.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())
	filename = basedir + '/lanl-acoustic-signal-{:06d}-ttf-{:02.5f}.png'.format(counter, chip.iloc[-1, -1])
	plt.savefig(filename, dpi=300)
	plt.close()

def plot_acoustic_spectrum(chip, counter, basedir, col = 'acoustic_data', for_humans = False):
	#ax = plt.gca()
	ax = plt.axes()

	if for_humans:
		ax.set_xlabel("Test Sample")
		ax.set_ylabel("Value")
		ax.legend(col)
		plt.grid(True)
	else:
		plt.axis('off')
		ax.legend().remove()

	ax.set_ylim((0, 100000))
	ax.set_xlim((10, 2000000))
	yf = scipy.fft(chip['acoustic_data'].values)
	#xf = scipy.fftpack.fftfreq(yf.size, 1 / 25e3)		# up to 12.5khz
	xf = scipy.fftpack.fftfreq(yf.size, 1 / 4e6)
	plt.plot(xf[:xf.size//2], abs(yf)[:yf.size//2], lw=1)
	if for_humans:
		mng = plt.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())
	filename = basedir + '/lanl-acoustic-spectrum-{:06d}-ttf-{:02.5f}.png'.format(counter, chip.iloc[-1, -1])
	plt.savefig(filename, dpi=300)
	plt.close()


def load_images(basedir, signal_or_spectrum='signal'):
	filelist = [f for f in listdir(basedir) if isfile(join(basedir, f)) and signal_or_spectrum in join(basedir, f)]

	dataset = []
	labels  = []

	filelist.sort()
	for fname in filelist:
		image   = cv2.imread(basedir + '/' + fname, cv2.IMREAD_GRAYSCALE)
		image   = cv2.resize(image, (int(image.shape[0]/4), int(image.shape[1]/4)))
		#print(image[:100])
		#print(image.shape)
		image = np.array(image, dtype="float") / 255.0
		#image.reshape(image.shape[0], image.shape[1], 1)
		#print(image.shape)
		#print(image[:100])
		#print(image.shape)
		dataset.append(image)
		#fname   = f.split(os.path.sep)	# e.g. lanl-acoustic-spectrum-004183-ttf-10.16080.png
		counter = fname[21:21+6]
		ttf     = fname[32:-4]
		print(f'Reading {signal_or_spectrum} image: {counter} with time to failure: {ttf}')
		labels.append(float(ttf))
		'''
		if int(counter) >= 99:
			break
		'''

	dataset = np.array(dataset)
	labels  = np.array(labels)
	print(f'Finished reading images, dataset has shape: {dataset.shape}')
	return dataset, labels


