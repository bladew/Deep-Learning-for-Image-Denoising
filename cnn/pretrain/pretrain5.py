import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '..')
import util
import os
import random
from random import randint

import train_utils

'''
	5 convolutional layers 
	filter size: 5 * 5 * 24
	24 * 26 + (8 * 25 + 1) * 24 * 3 + 25 * 24 + 1 = 15697
'''

# Set up placeholders
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)
vertical = tf.placeholder(tf.bool)
x = tf.cond(training, lambda: tf.placeholder(tf.float32, shape=[None, 26 * 26]), lambda: tf.placeholder(tf.float32, shape=[None, 321 * 481]))
y_ = tf.cond(training, lambda: tf.placeholder(tf.float32, shape=[None, 6 * 6]), lambda: tf.placeholder(tf.float32, shape=[None, 321 * 481]))

# First layer
x_image = tf.cond(training,
	lambda: tf.reshape(x, [-1, 26, 26, 1]),
	lambda: tf.cond(vertical,
		lambda: tf.reshape(x, [-1, 481, 321, 1]),
		lambda: tf.reshape(x, [-1, 321, 481, 1])))

W_conv1 = train_utils.weight_variable([5, 5, 1, 24], name='W_conv1')
b_conv1 = train_utils.bias_variable([24], name='b_conv1')
h_conv1 = tf.sigmoid(train_utils.conv2d(x_image, W_conv1, training) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

# Second layer
W_conv2 = train_utils.weight_variable([5, 5, 24, 24], name='W_conv2')
b_conv2 = train_utils.bias_variable([24], name='b_conv2')
h_conv2 = tf.sigmoid(train_utils.conv2d(h_conv1_drop, W_conv2, training) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)

# Third layer
W_conv3 = train_utils.weight_variable([5, 5, 24, 24], name='W_conv3')
b_conv3 = train_utils.bias_variable([24], name='b_conv3')
h_conv3 = tf.sigmoid(train_utils.conv2d(h_conv2_drop, W_conv3, training) + b_conv3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

# Fourth layer
W_conv4 = train_utils.weight_variable([5, 5, 24, 24], name='W_conv4')
b_conv4 = train_utils.bias_variable([24], name='b_conv4')
h_conv4 = tf.sigmoid(train_utils.conv2d(h_conv3_drop, W_conv4, training) + b_conv4)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)

# Fifth layer
W_conv5 = train_utils.weight_variable([5, 5, 24, 24], name='W_conv5')
b_conv5 = train_utils.bias_variable([24], name='b_conv5')
h_conv5 = tf.sigmoid(train_utils.conv2d(h_conv4_drop, W_conv5, training) + b_conv5)

# Last layer
W_conv6 = train_utils.weight_variable([5, 5, 24, 1], name='W_conv6')
b_conv6 = train_utils.bias_variable([1], name='b_conv6')
y_image = tf.sigmoid(train_utils.conv2d(h_conv5, W_conv6, training) + b_conv6)

y = tf.cond(training,
	lambda: tf.reshape(y_image, [-1, 6 * 6]),
	lambda: tf.reshape(y_image, [-1, 321 * 481]))

# Calculate the loss function by mean square cost
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

restorer = tf.train.Saver({
	'W_conv1': W_conv1,
	'b_conv1': b_conv1,
	'W_conv2': W_conv2,
	'b_conv2': b_conv2,
	'W_conv3': W_conv3,
	'b_conv3': b_conv3,
	'W_conv4': W_conv4,
	'b_conv4': b_conv4,
})

saver = tf.train.Saver({
	'W_conv1': W_conv1,
	'b_conv1': b_conv1,
	'W_conv2': W_conv2,
	'b_conv2': b_conv2,
	'W_conv3': W_conv3,
	'b_conv3': b_conv3,
	'W_conv4': W_conv4,
	'b_conv4': b_conv4,
	'W_conv5': W_conv5,
	'b_conv5': b_conv5,
	'W_conv6': W_conv6,
	'b_conv6': b_conv6,
})

def train(images, restore_path):
	'''
		apply result of third pretrain for training
		use images in ./train and ./test as training dataset
	'''
	restorer.restore(sess, restore_path)

	for epoch in range(8000):
		tol_err = 0
		for step in range(len(images) / 6 + 1):
			trainX, trainY = train_utils.get_next_batch(images, hidden_layer_size = 4)
			_, err = sess.run([train_step, loss], feed_dict={x: trainX, y_: trainY, training: True, vertical: None, keep_prob: 8.0/24.0})
			tol_err += err

		print epoch, tol_err / step

def save():
	# save pretrained result of layer 1
	save_path = saver.save(sess, "./models/model-5")
	print("Model saved in file: %s" % save_path)
	return save_path
