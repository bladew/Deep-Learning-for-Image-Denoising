import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '..')
import util
import os
import random
from random import randint
# sys.path.pop(0)
import train_utils

'''
	5 convolutional layers 
	filter size: 5 * 5 * 24
	24 * 26 + (8 * 25 + 1) * 24 * 3 + 25 * 24 + 1 = 15697
'''

# Set up placeholders
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

# Last layer
W_conv5 = train_utils.weight_variable([5, 5, 24, 1], name='W_conv5')
b_conv5 = train_utils.bias_variable([1], name='b_conv5')
y_image = tf.sigmoid(train_utils.conv2d(h_conv1, W_conv5, training) + b_conv5)

y = tf.cond(training,
	lambda: tf.reshape(y_image, [-1, 18 * 18]),
	lambda: tf.reshape(y_image, [-1, 321 * 481]))

# Calculate the loss function by mean square cost
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver({
	'W_conv1': W_conv1,
	'b_conv1': b_conv1,
})

def train(images):
	'''
		use images in ./train and ./test as training dataset
	'''
	for epoch in range(10):
		tol_err = 0
		for step in range(len(images) / 6 + 1):
			trainX, trainY = train_utils.get_next_batch(images, 1)
			_, err = sess.run([train_step, loss], feed_dict={x: trainX, y_: trainY, training: True, vertical: None})
			tol_err += err

		print epoch, tol_err / step
 
def save():
	'''
		save pretrained result of layer 1
	'''
	save_path = saver.save(sess, "./models/with_pretrain/model-1")
	print("Model saved in file: %s" % save_path)
	return save_path
