import tensorflow as tf
import sys
sys.path.insert(0, '..')
import util
import os

'''
	5 convolutional layers 
	filter size: 5 * 5 * 24
	24 * 26 + (8 * 25 + 1) * 24 * 3 + 25 * 24 + 1 = 15697
'''

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# setup placeholders based on image orientation
x = tf.placeholder(tf.float32, shape=[None, 321*481])
vertical = tf.placeholder(tf.bool)

# reshape image based on orientation
x_image = tf.cond(vertical, lambda: tf.reshape(x, [-1,321,481,1]), lambda: tf.reshape(x, [-1, 481, 321, 1]))

# setup deopout
keep_prob = tf.placeholder(tf.float32)

# first layer
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
# h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.sigmoid(conv2d(h_conv1_drop, W_conv2) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)

# third layer
W_conv3 = weight_variable([5, 5, 32, 4])
b_conv3 = bias_variable([4])
h_conv3 = tf.sigmoid(conv2d(h_conv2_drop, W_conv3) + b_conv3)
# h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)
'''
# fourth layer
W_conv4 = weight_variable([5, 5, 24, 24])
b_conv4 = bias_variable([24])
h_conv4 = tf.sigmoid(conv2d(h_conv3_drop, W_conv4) + b_conv4)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)

# last layer
W_conv5 = weight_variable([5, 5, 24, 1])
b_conv5 = bias_variable([1])
y_image = tf.sigmoid(conv2d(h_conv4_drop, W_conv5) + b_conv5)
'''
h_conv3_flat = tf.reshape(h_conv3, [-1, 321 * 481 * 4])
# h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
# h_pool1 = max_pool_2x2(h_conv1)

# Feed-Forward Layer 1 (with ReLU Activation) - 1024 Units

W_fc1 = weight_variable([321 * 481 * 4, 128])
b_fc1 = bias_variable([128])
h_fc1 = tf.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 321 * 481])
b_fc2 = bias_variable([321 * 481])
h_fc2 = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

y_image = tf.cond(vertical, lambda: tf.reshape(h_fc2, [-1,321,481,1]), lambda: tf.reshape(h_fc2, [-1, 481, 321, 1]))

# Feed-Forward Layer 2 (for final Logits with softmax) - 10 Units
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Calculate the loss function by mean square cost
loss = tf.reduce_mean(tf.reduce_sum(tf.square(h_fc2 - x)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# num of epoches = 10
for _ in range(1):
	step = 0
	for image in util.ImReader("../images/train").read():
		sess.run(train_step, feed_dict={x: image.reshape(1, 321 * 481), vertical: image.shape == (321, 481), keep_prob: 1})
		step += 1
		if step % 10 == 0:
			print sess.run(loss, feed_dict={x: image.reshape(1, 321 * 481), vertical: image.shape == (321, 481), keep_prob: 1})

	for image in util.ImReader("../images/test").read():
		sess.run(train_step, feed_dict={x: image.reshape(1, 321 * 481), vertical: image.shape == (321, 481), keep_prob: 1})
		step += 1
		if step % 10 == 0:
			print sess.run(loss, feed_dict={x: image.reshape(1, 321 * 481), vertical: image.shape == (321, 481), keep_prob: 1})

count = 0
for image in util.ImReader("../images/val").read():
	im = sess.run(y_image, feed_dict={x: image.reshape(1, 321*481), vertical: image.shape == (321, 481), keep_prob: 1.0})
	im = im.reshape(321, 481) if image.shape == (321, 481) else im.reshape(481, 321)
	util.imsave(im, "../images/result/"+str(count)+".PNG")
	count += 1
