import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ../util

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 8 * Wx + b, all 8 randomly chosen feature map share the same bias
# 24 * 26 + (8 * 25 + 1) * 24 * 3 + 25 * 24 + 1 = 15697

# setup placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# TODO: reshape the image based on its original dimension
x_image = tf.reshape(x, [-1,28,28,1])

# setup deopout
keep_prob = tf.placeholder(tf.float32)

# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
# h_pool1 = max_pool_2x2(h_conv1)

# second layer
# W_conv2 = weight_variable([6, 6, 6, 6])
# b_conv2 = bias_variable([6])
# h_conv2 = tf.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

#TODO: add three more conv layers

# Feed-Forward Layer 1 (with ReLU Activation) - 1024 Units
W_fc1 = weight_variable([28 * 28 * 32, 28*28])
b_fc1 = bias_variable([28*28])

h_conv1_drop_flat = tf.reshape(h_conv1_drop, [-1, 28*28*32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_drop_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
y = tf.reshape([28, 28])

# Feed-Forward Layer 2 (for final Logits with softmax) - 10 Units
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#TODO: need three(?) full connection layers in total???

# Calculate the loss function by mean square cost
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - x)))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# TODO: need to invoke util.read_data

# Train for 2000 batches, with a batch size of 50.
for i in range(2000):
	# TODO: need to get batch from data
	# batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
