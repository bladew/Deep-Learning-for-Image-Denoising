import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

from denoising_auto_encoder import DenoisingAutoEncoder

class StackedDenoisingAutoEncoder(object):
    """StackedDenoisingAutoEncoder using DenoisingAutoEncoder"""
    def __init__(self, train_readers, layer = 2, batch_sz = 10, learning_rate =1e-3, pre_train = False):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.train_readers = train_readers
        self.layer = layer
        self.batch_sz = batch_sz
        features = self.train_readers[0].features

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, features])
        self.input_original = tf.placeholder(tf.float32, [None, features])

        # Create das
        self.das = [DenoisingAutoEncoder(self.train_readers, self.batch_sz, learning_rate) for i in xrange(self.layer)]

        # Pretrain
        if pre_train:
            self.pretrain(10)

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

        self.h_ = next_input

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
                print "start %d da %d epoch"%(i, _)
                for reader in self.train_readers:
                    for x,y in reader.read_mat(10, True):
                        next_input = [x,y]
                        for j in xrange(i):
                            da = self.das[j]
                            next_input[0] = da.sess.run([da.h_], feed_dict = {da.input_corrupt: next_input[0]})[0]
                            next_input[1] = da.sess.run([da.h_], feed_dict = {da.input_corrupt: next_input[1]})[0]
                        da = self.das[i]
                        da.sess.run([da.train_step], feed_dict = {da.input_corrupt: next_input[0], da.input_original: next_input[1]})


    def train(self, epoch = 10):
        errors = []
        for i in xrange(epoch):
            for label, reader in enumerate(self.train_readers):
                if label >= len(errors):
                    errors.append([])
                count = 0
                for x,y in reader.read_mat(self.batch_sz, True):
                    feed = {self.input_corrupt: x, self.input_original: y}
                    error, _ = self.sess.run([self.error, self.train_step], feed_dict = feed)
                    if count >= len(errors[label]):
                        errors[label].append(error)
                        print "batch %d-%d, error is %g"%(label, count, error)
                    else:
                        print "batch %d-%d, error changes from %g -> %g"%(label, count, errors[label][count], error)
                        errors[label][count] = error
                    count += 1


    def test(self, test_reader):
        count = 0
        for x,y in test_reader.read_mat(1, True):
            feed = {self.input_corrupt: x[0], self.input_original: y[0]}
            cleared = self.sess.run([self.h_], feed_dict = feed)
            print "image %d, PSNR change %g -> %g"%(count, util.calcPSNR(x[0], y[0]), util.calcPSNR(cleared, y[0]))
            count += 1


    # def save_model(self):



if __name__ == '__main__':
    train_reader1 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    train_reader2 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    sda = StackedDenoisingAutoEncoder([train_reader1, train_reader2], pre_train = False)
    sda.train()
    sda.test(test_reader)