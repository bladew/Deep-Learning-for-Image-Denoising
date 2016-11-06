# import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

class DenoisingAutoEncoder(object):
    """Denoising Auto-Encoder"""
    def __init__(self, train_reader, batch_sz = 1, learning_rate = 1e-4):
        super(DenoisingAutoEncoder, self).__init__()
        self.train = train_reader
        self.batch_sz = batch_sz

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, self.train.features])
        self.input_original = tf.placeholder(tf.float32, [None, self.train.features])

        # Create param
        self.w = tf.variable(tf.truncated_normal([self.train.features, 100], stddev = 0.1))
        self.w_ = tf.variable(tf.truncated_normal([100, self.train.features], stddev = 0.1))
        self.b = tf.variable(tf.constant(0.1, shape = [1, 100]))
        self.b_ = tf.variable(tf.constant(0.1, shape = [1, self.train.features]))

        # Feed-forward step
        self.h = tf.nn.sigmoid(tf.matmul(self.input_corrupt, self.w) + self.b)
        self.h_ = tf.nn.sigmoid(tf.matmul(self.h, self.w_) + self.b_)

        # Loss func
        self.error = tf.sqrt(tf.reduce_mean(tf.square(self.input_original - self.h_)))

        # Train
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.error)

        # Initialization
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)


    def train(self, epoch = 50):
        for i in xrange(epoch):
            for x,y in self.train.read_mat(self, batch_sz = self.batch_sz, vector = True):
                feed = {self.input_corrupt: x, self.input_original: y}
                error, _ = self.sess.run([self.error, self.train_step], feed_dict = feed)
                print "error is %g"%(error)

    def test(self, test_reader):
        for x,y in test_reader.read_mat(self, batch_sz = self.batch_sz, vector = True):
            feed = {self.input_corrupt: x, self.input_original: y}
            cleared = self.sess.run([self.h_], feed_dict = feed)
            print "PSNR change %g -> %g"%(util.calcPSNR(x, y), util.calcPSNR(cleared, y))


if __name__ == '__main__':
    train_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    da = DenoisingAutoEncoder(train_reader)
    da.train()