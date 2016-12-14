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


def get_next_batch(batch_size):
	X, y = [], []
	for image in random.sample(images, batch_size):
		corrupted, original = image[0][0], image[1][0]
		nRow, nCol = corrupted.shape
		m = randint(0, nRow - 26)
		n = randint(0, nCol - 26)	
		X.append(corrupted[m:m+26,n:n+26].reshape(26 * 26))
		y.append(original[m+10:m+16,n+10:n+16].reshape(6 * 6))
	return X, y

def get_next_batch22(batch_size):
	X, y = [], []
	for image in random.sample(images, batch_size):
		corrupted, original = image[0][0], image[1][0]
		nRow, nCol = corrupted.shape
		m = randint(0, nRow - 22)
		n = randint(0, nCol - 22)	
		X.append(corrupted[m:m+22,n:n+22].reshape(22 * 22))
		y.append(original[m+8:m+14,n+8:n+14].reshape(6 * 6))
	return X, y

images = []
for image in util.ImReader("../images/train").read_mat():
	images.append(image)
for image in util.ImReader("../images/test").read_mat():
	images.append(image)


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

W_conv1 = weight_variable([5, 5, 1, 24], name='W_conv1')
b_conv1 = bias_variable([24], name='b_conv1')
h_conv1 = tf.sigmoid(conv2d(x_image, W_conv1, training) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

# Second layer
W_conv2 = weight_variable([5, 5, 24, 24], name='W_conv2')
b_conv2 = bias_variable([24], name='b_conv2')
h_conv2 = tf.sigmoid(conv2d(h_conv1_drop, W_conv2, training) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)

# Third layer
W_conv3 = weight_variable([5, 5, 24, 24], name='W_conv3')
b_conv3 = bias_variable([24], name='b_conv3')
h_conv3 = tf.sigmoid(conv2d(h_conv2_drop, W_conv3, training) + b_conv3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

# Fourth layer
W_conv4 = weight_variable([5, 5, 24, 24], name='W_conv4')
b_conv4 = bias_variable([24], name='b_conv4')
h_conv4 = tf.sigmoid(conv2d(h_conv3_drop, W_conv4, training) + b_conv4)

# Last layer
W_conv5 = weight_variable([5, 5, 24, 1], name='W_conv5')
b_conv5 = bias_variable([1], name='b_conv5')
y_image = tf.sigmoid(conv2d(h_conv4, W_conv5, training) + b_conv5)

y = tf.cond(training,
	lambda: tf.reshape(y_image, [-1, 6 * 6]),
	lambda: tf.reshape(y_image, [-1, 321 * 481]))

# Calculate the loss function by mean square cost
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

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
	'b_conv5': b_conv5
})
saver.restore(sess, "models/model-4")


with open('psnr/four_layer_with_pretrain_psnr.csv', mode='wb') as f:
	count = 0
	for image in util.ImReader("../images/val").read_mat():
		corrupted, original = image[0][0], image[1][0]
		recovered = sess.run(y_image, feed_dict={x: corrupted.reshape(1, 321*481), y_: original.reshape(1, 321 * 481), vertical: corrupted.shape == (481, 321), training: False, keep_prob: 1.0})
		recovered = recovered.reshape(321, 481) if corrupted.shape == (321, 481) else recovered.reshape(481, 321)
		corruptedPSNR = util.calcPSNR(corrupted, original)
		recoveredPSNR = util.calcPSNR(recovered, original)
		#util.imsave(original, "../images/result3/"+str(count)+"_original.PNG")
		#util.imsave(corrupted, "../images/result3/"+str(count)+"_corrupted.PNG")
		util.imsave(recovered, "result/four_layer_with_pretraining/"+str(count)+"_recovered.PNG")
		f.write(str(count) + ',' + str(corruptedPSNR) + ',' + str(recoveredPSNR) + '\n')
		print count, corruptedPSNR, recoveredPSNR
		count += 1

#for x, y in util.ImReader("../images/val").read_mat():
#	print util.calcPSNR(x[0], y[0])
