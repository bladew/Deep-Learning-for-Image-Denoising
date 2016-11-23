import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

from denoising_auto_encoder import DenoisingAutoEncoder

class StackedDenoisingAutoEncoder(object):
    """StackedDenoisingAutoEncoder using DenoisingAutoEncoder"""
    def __init__(self, train_readers, layer = 2, batch_sz = 20, patch_sz = 6, learning_rate =1e-5, pre_train = False):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.train_readers = train_readers
        self.layer = layer
        self.batch_sz = batch_sz
        self.patch_sz = patch_sz
        features = min(self.train_readers[0].features, self.patch_sz * self.patch_sz)

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, features])
        self.input_original = tf.placeholder(tf.float32, [None, features])

        # Create das
        self.das = [DenoisingAutoEncoder(self.train_readers, self.batch_sz, self.patch_sz, learning_rate, model = 'da5') for i in xrange(self.layer)]

        # Pretrain
        if pre_train:
            self.pretrain(1)

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


    def pretrain(self, epoch = 1):
        for i in xrange(len(self.das)):
            for _ in xrange(epoch):
                print "start %d da %d epoch"%(i, _)
                for reader in self.train_readers:
                    for x,y in reader.read_patch(self.batch_sz, self.patch_sz):
                        next_input = [x,y]
                        for j in xrange(i):
                            da = self.das[j]
                            next_input[0] = da.sess.run([da.h], feed_dict = {da.input_corrupt: next_input[0]})[0]
                            next_input[1] = da.sess.run([da.h], feed_dict = {da.input_corrupt: next_input[1]})[0]
                        da = self.das[i]
                        da.sess.run([da.train_step], feed_dict = {da.input_corrupt: next_input[0], da.input_original: next_input[1]})


    def train(self, epoch = 1):
        # errors = []
        for i in xrange(epoch):
            for label, reader in enumerate(self.train_readers):
                # if label >= len(errors):
                #     errors.append([])
                count = 0
                for x,y in reader.read_patch(self.batch_sz, self.patch_sz):
                    feed = {self.input_corrupt: x, self.input_original: y}
                    error, _ = self.sess.run([self.error, self.train_step], feed_dict = feed)
                    print "batch %d-%d, error is %g"%(label, count, error)
                    # if count >= len(errors[label]):
                    #     errors[label].append(error)
                    #     print "batch %d-%d, error is %g"%(label, count, error)
                    # else:
                    #     print "batch %d-%d, error changes from %g -> %g"%(label, count, errors[label][count], error)
                    #     errors[label][count] = error
                    count += 1


    def test(self, test_reader):
        count = 0
        res = []
        for corrupt,original in test_reader.read_mat(1):
            x = test_reader.im2col(corrupt[0], self.patch_sz).T
            y = test_reader.im2col(original[0], self.patch_sz).T
            feed = {self.input_corrupt: x, self.input_original: y}
            cleared_patch = self.sess.run([self.h_], feed_dict = feed)
            cleared = test_reader.reconstruct(cleared_patch[0].T, self.patch_sz, original[0].shape)
            res.append(util.calcPSNR(cleared, original[0]))
            print "%d, PSNR change %g -> %g"%(count, util.calcPSNR(corrupt[0], original[0]), res[-1])
            count += 1

        print "avg %g, std %g"%(np.mean(res), np.std(res))
        

    def save_model(self, prefix):
        for i,da in enumerate(self.das):
            da.save_model(prefix + '_da' + str(i))



if __name__ == '__main__':
    train_reader1 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    train_reader2 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    sda = StackedDenoisingAutoEncoder([train_reader1, train_reader2], pre_train = False)
    sda.train(1)
    sda.save_model('sda')
    sda.test(test_reader)