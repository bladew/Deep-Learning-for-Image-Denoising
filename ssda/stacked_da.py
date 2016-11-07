# import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

from denoising_auto_decoder import DenoisingAutoEncoder

class StackedDenoisingAutoEncoder(object):
    """StackedDenoisingAutoEncoder using DenoisingAutoEncoder"""
    def __init__(self, train_reader, layer = 10, batch_sz = 1, learning_rate =1e-4, pre_train = False):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.train = train_reader
        self.layer = layer
        self.batch_sz = batch_sz

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, self.train.features])
        self.input_original = tf.placeholder(tf.float32, [None, self.train.features])

        # Create das
        self.das = [DenoisingAutoEncoder(self.train, self.batch_sz, learning_rate) for i in xrange(self.layer)]

        # Pretrain
        if pre_train:
            self.pretrain()

        # Create model param
        self.w = [da.w for da in self.das]
        self.w_ = [da.w_ for da in self.das]
        self.b = [da.b for da in self.das]
        self.b_ = [da.b_ for da in self.das]

        # Feed-forward step
        self.layer_nodes = []
        next_input = self.input_corrupt
        for i in xrange(len(self.w)):
            h = tf.nn.sigmoid(tf.matmul(self.input_corrupt, self.w[i]) + self.b[i])
            self.layer_nodes.append(h)
            h_ = tf.nn.sigmoid(tf.matmul(h, self.w_[i]) + self.b_[i])
            self.layer_nodes.append(h_)
            next_input = h_

        self.h_ = self.layer_nodes[-1]

        # Loss func
        self.error = tf.sqrt(tf.reduce_mean(tf.square(self.input_original - self.h_)))

        # Train
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.error)

        # Initialization
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)


    def pretrain(self, epoch = 30):
        for i in xrange(len(self.das)):
            for _ in xrange(epoch):
                for x,y in self.train.read_mat(batch_sz = 20, vector = True):
                    next_input = [x,y]
                    for j in xrange(i):
                        da = self.das[j]
                        next_input[0] = da.sess.run([da.h], feed_dict = {da.input_corrupt: next_input[0]})
                        next_input[1] = da.sess.run([da.h], feed_dict = {da.input_corrupt: next_input[1]})
                    da = self.das[i]
                    da.sess.run([da.train_step], feed_dict = {
                        da.input_corrupt: next_input[0], 
                        da.input_original: next_input[1]})



    def train(self, epoch = 50):
        for i in xrange(epoch):
            for x,y in self.train.read_mat(batch_sz = self.batch_sz, vector = True):
                feed = {self.input_corrupt: x, self.input_original: y}
                error, _ = self.sess.run([self.error, self.train_step], feed_dict = feed)
                print "error is %g"%(error)

    def test(self, test_reader):
        for x,y in test_reader.read_mat(self, batch_sz = self.batch_sz, vector = True):
            feed = {self.input_corrupt: x, self.input_original: y}
            cleared = self.sess.run([self.h_], feed_dict = feed)
            print "PSNR change %g -> %g"%(util.calcPSNR(x, y), util.calcPSNR(cleared, y))

		