import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

from denoising_auto_encoder import DenoisingAutoEncoder

class StackedDenoisingAutoEncoder(object):
    """StackedDenoisingAutoEncoder using DenoisingAutoEncoder"""
    def __init__(self, train_readers, layer = 2, batch_sz = 20, patch_sz = 6, learning_rate =1e-6, regular = 1e-4, pre_train = False, model = None):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.train_readers = train_readers
        self.layer = layer
        self.batch_sz = batch_sz
        self.patch_sz = patch_sz
        features = min(self.train_readers[0].features, self.patch_sz * self.patch_sz)

        # Session
        self.sess = tf.Session()

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, features])
        self.input_original = tf.placeholder(tf.float32, [None, features])

        # Create das
        self.das = [DenoisingAutoEncoder(self.train_readers, self.batch_sz, self.patch_sz, learning_rate) for i in xrange(self.layer)]

        # Pretrain
        if pre_train:
            self.pretrain(1)
        
        # Create model param
        self.ws, self.w_s, self.bs, self.b_s = [], [], [], []
        for i,da in enumerate(self.das):
            params = da.get_param()
            self.ws.append(tf.Variable(params['w']))
            self.bs.append(tf.Variable(params['b']))
            self.w_s.append(tf.Variable(params['w_']))
            self.b_s.append(tf.Variable(params['b_']))
        self.saver = tf.train.Saver(max_to_keep=30)

        # Feed-forward step
        self.layer_nodes = []
        next_input = self.input_corrupt
        for i in xrange(len(self.ws)):
            h = tf.nn.sigmoid(tf.matmul(next_input, self.ws[i]) + self.bs[i])
            self.layer_nodes.append(h)
            h_ = tf.nn.sigmoid(tf.matmul(h, self.w_s[i]) + self.b_s[i])
            self.layer_nodes.append(h_)
            next_input = h_

        self.h_ = next_input

        # Loss func
        self.w_sum = 0
        for w in self.ws:
            self.w_sum += tf.reduce_sum(tf.square(w))
        for w_ in self.w_s:
            self.w_sum += tf.reduce_sum(tf.square(w_))

        self.error = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_original - self.h_), 1)) + regular / 2 * self.w_sum

        # Train
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.error)

        # Initialization
        init = tf.initialize_all_variables()
        self.sess.run(init)

        if model != None:
            self.saver.restore(self.sess, model)


    def pretrain(self, epoch = 1):
        for i in xrange(1, len(self.das)):
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
        errors = []
        temp = 0
        for i in xrange(epoch):
            count = 0
            for label, reader in enumerate(self.train_readers):
                for x,y in reader.read_patch(self.batch_sz, self.patch_sz):
                    feed = {self.input_corrupt: x, self.input_original: y}
                    error, _ = self.sess.run([self.error, self.train_step], feed_dict = feed)
                    print "batch %d-%d, error is %g"%(label, count, error)
                    count += 1
                    temp += error

                    if count % 1000 == 0:
                        errors.append([count, temp / 1000])
                        temp = 0
        
            np.savetxt('sda_no_pre_training_err' + str(i) + '.csv', errors, fmt = '%f', delimiter=",")
            self.save_model('sda_no_pre_training_' + str(i))


    def test(self, test_reader):
        count = 0
        res = []
        for corrupt,original in test_reader.read_mat(1):
            x = test_reader.im2col(corrupt[0], self.patch_sz).T
            y = test_reader.im2col(original[0], self.patch_sz).T
            feed = {self.input_corrupt: x, self.input_original: y}
            cleared_patch = self.sess.run([self.h_], feed_dict = feed)
            cleared = np.clip(test_reader.reconstruct(cleared_patch[0].T, self.patch_sz, original[0].shape), 0, 1)
            res.append(util.calcPSNR(cleared, original[0]))
            print "%d, PSNR change %g -> %g"%(count, util.calcPSNR(corrupt[0], original[0]), res[-1])
            util.imsave(cleared, str(count)  + '_pretraining')
            count += 1

        print "avg %g, std %g"%(np.mean(res), np.std(res))
        

    def save_model(self, path):
        self.saver.save(self.sess, path)


    def load_model(self, path):
        self.saver.restore(self.sess, path)


if __name__ == '__main__':
    train_reader1 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    train_reader2 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    # test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/classic")
    sda = StackedDenoisingAutoEncoder([train_reader1, train_reader2], pre_train = False, model = 'sda')
    sda.train(2)
    sda.save_model('sda')
    sda.test(test_reader)