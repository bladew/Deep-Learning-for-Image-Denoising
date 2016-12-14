import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '..')
import util
import os
import random
from random import randint

'''
	5 convolutional layers 
	filter size: 5 * 5 * 24
	24 * 26 + (8 * 25 + 1) * 24 * 3 + 25 * 24 + 1 = 15697
'''

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name)

def conv2d(x, W, training):
	return tf.cond(training,
		lambda: tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID'),
		lambda: tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))


def get_next_batch(images, hidden_layer_size, batch_size=6):
	final_patch_size = 26 - (hidden_layer_size+1) * 4
	X, y = [], []
	for image in random.sample(images, batch_size):
		corrupted, original = image[0][0], image[1][0]
		nRow, nCol = corrupted.shape
		m = randint(0, nRow - 26)
		n = randint(0, nCol - 26)	
		X.append(corrupted[m:m+26,n:n+26].reshape(26 * 26))
		y.append(original[m + 13 - final_patch_size/2: m + 13 + final_patch_size/2,
				n + 13 - final_patch_size/2: n + 13 + final_patch_size/2]
				.reshape(final_patch_size * final_patch_size))
	return X, y

def read_images():
	images = []
	for image in util.ImReader("../images/train").read_mat():
		images.append(image)
	for image in util.ImReader("../images/test").read_mat():
		images.append(image)
	return images
