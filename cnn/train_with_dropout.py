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

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def get_next_batch1(batch_size):
	X, y = [], []
	indices = random.sample([x for x in xrange(num_imgs1)], batch_size)
	i = 0
	for image in util.ImReader("../images/train").read_mat():
		if i in indices: 
			corrupted, original = image[0][0], image[1][0]
			nRow, nCol = corrupted.shape
			m = randint(0, nRow - 26)
			n = randint(0, nCol - 26)	
			X.append(corrupted[m:m+26,n:n+26].reshape(26 * 26))
			y.append(original[m+10:m+16,n+10:n+16].reshape(6 * 6))
		i += 1
	return X, y

def get_next_batch2(batch_size):
	X, y = [], []
	indices = random.sample([x for x in xrange(num_imgs2)], batch_size)
	i = 0
	for image in util.ImReader("../images/test").read_mat():
		if i in indices:
			corrupted, original = image[0][0], image[1][0]
			nRow, nCol = corrupted.shape
			m = randint(0, nRow - 26)
			n = randint(0, nCol - 26)	
			X.append(corrupted[m:m+26,n:n+26].reshape(26 * 26))
			y.append(original[m+10:m+16,n+10:n+16].reshape(6 * 6))
		i += 1
	return X, y

num_imgs1 = 0
num_imgs2 = 0
for image in util.ImReader("../images/train").read_mat():
	num_imgs1 += 1
for image in util.ImReader("../images/test").read_mat():
	num_imgs2 += 1


# Set up placeholders
x = tf.placeholder(tf.float32, shape=[None, 26 * 26])
y_ = tf.placeholder(tf.float32, shape=[None, 6 * 6])
keep_prob = tf.placeholder(tf.float32)

# First layer
x_image = tf.reshape(x, [-1, 26, 26, 1])
W_conv1 = weight_variable([5, 5, 1, 24], name='W_conv1')
b_conv1 = bias_variable([24], name='b_conv1')
h_conv1 = tf.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

# Second layer
W_conv2 = weight_variable([5, 5, 24, 24], name='W_conv2')
b_conv2 = bias_variable([24], name='b_conv2')
h_conv2 = tf.sigmoid(conv2d(h_conv1_drop, W_conv2) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)

# Third layer
W_conv3 = weight_variable([5, 5, 24, 24], name='W_conv3')
b_conv3 = bias_variable([24], name='b_conv3')
h_conv3 = tf.sigmoid(conv2d(h_conv2_drop, W_conv3) + b_conv3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

# Fourth layer
W_conv4 = weight_variable([5, 5, 24, 24], name='W_conv4')
b_conv4 = bias_variable([24], name='b_conv4')
h_conv4 = tf.sigmoid(conv2d(h_conv3_drop, W_conv4) + b_conv4)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)

# Last layer
W_conv5 = weight_variable([5, 5, 24, 1], name='W_conv5')
b_conv5 = bias_variable([1], name='b_conv5')
y_image = tf.sigmoid(conv2d(h_conv4_drop, W_conv5) + b_conv5)
y = tf.reshape(y_image, [-1, 6 * 6])

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

# use images in ./train and ./test as training dataset
for epoch in range(100):
	print epoch
	for step in range((num_imgs1 + num_imgs2) / 6):
		trainX, trainY = get_next_batch1(6)
		_, err = sess.run([train_step, loss], feed_dict={x: trainX, y_: trainY, keep_prob: 8.0 / 24})

		trainX, trainY = get_next_batch2(6)
		_, err = sess.run([train_step, loss], feed_dict={x: trainX, y_: trainY, keep_prob: 8.0 / 24})

		if step % 10 == 0:
			print err
			
save_path = saver.save(sess, "model")
print("Model saved in file: %s" % save_path)
# saver.restore(sess, "model")


count = 0
for image in util.ImReader("../images/val").read_mat():
	corrupted, original = image[0][0], image[1][0]
	recovered = sess.run(y_image, feed_dict={x: corrupted.reshape(1, 321*481), y_: original.reshape(1, 321 * 481), vertical: corrupted.shape == (321, 481), keep_prob: 1.0})
	recovered = recovered.reshape(321, 481) if corrupted.shape == (321, 481) else recovered.reshape(481, 321)
	util.imsave(original, "../images/result3/"+str(count)+"_original.PNG")
	util.imsave(corrupted, "../images/result3/"+str(count)+"_corrupted.PNG")
	util.imsave(recovered, "../images/result3/"+str(count)+"_recovered.PNG")
	print count, "corrupted:", util.calcPSNR(corrupted, original), "recovered:", util.calcPSNR(recovered, original)
	count += 1
