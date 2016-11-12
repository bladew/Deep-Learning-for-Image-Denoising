import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

class DenoisingAutoEncoder(object):
    """Denoising Auto-Encoder"""
    def __init__(self, train_readers, batch_sz = 20, patch_sz = 10, learning_rate = 1e-3):
        super(DenoisingAutoEncoder, self).__init__()
        self.train_readers = train_readers
        self.batch_sz = batch_sz
        self.patch_sz = patch_sz
        features = min(self.train_readers[0].features, patch_sz * patch_sz)

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, features])
        self.input_original = tf.placeholder(tf.float32, [None, features])

        # Create param
        self.w = tf.Variable(tf.truncated_normal([features, 100], stddev = 0.1))
        self.w_ = tf.Variable(tf.truncated_normal([100, features], stddev = 0.1))
        self.b = tf.Variable(tf.constant(0.1, shape = [1, 100]))
        self.b_ = tf.Variable(tf.constant(0.1, shape = [1, features]))

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
        errors = []
        for i in xrange(epoch):
            for label, reader in enumerate(self.train_readers):
                if label >= len(errors):
                    errors.append([])
                count = 0
                for x,y in reader.read_patch(self.batch_sz, self.patch_sz):
                    feed = {self.input_corrupt: x, self.input_original: y}
                    h_, error, _ = self.sess.run([self.h_, self.error, self.train_step], feed_dict = feed)
                    print "%d batch, error is %g"%(count, error)
                    # if count >= len(errors[label]):
                    #     errors[label].append([error])
                    # else:
                    #     errors[label][count].append(error)
                    # print errors[label][count]
                    count += 1
                    

    def test(self, test_reader):
        count = 0
        # for x,y in test_reader.read_mat(1, True):
        for corrupt, original in test_reader.read_mat(1):
            x = test_reader.im2col(corrupt[0], self.patch_sz).T
            y = test_reader.im2col(original[0], self.patch_sz).T
            feed = {self.input_corrupt: x, self.input_original: y}
            cleared_patch = self.sess.run([self.h_], feed_dict = feed)
            cleared = test_reader.reconstruct(cleared_patch, self.patch_sz, original[0].shape)
            print "%d, PSNR change %g -> %g"%(count, util.calcPSNR(corrupt[0], original[0]), util.calcPSNR(cleared, original[0]))
            count += 1


    # def save_model(self, path = None):




if __name__ == '__main__':
    train_reader1 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    train_reader2 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    da = DenoisingAutoEncoder([train_reader1, train_reader2])
    # da.train(10)
    da.test(test_reader)