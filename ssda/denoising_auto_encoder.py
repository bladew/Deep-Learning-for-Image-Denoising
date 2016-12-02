import tensorflow as tf
import sys
import numpy as np

sys.path.insert(0, '..')
import util

class DenoisingAutoEncoder(object):
    """Denoising Auto-Encoder"""
    def __init__(self, train_readers, batch_sz = 20, patch_sz = 6, learning_rate = 1e-4, regular = 1e-4, ryo = 0.05, beta = 0.01, model = None):
        super(DenoisingAutoEncoder, self).__init__()
        self.train_readers = train_readers
        self.batch_sz = batch_sz
        self.patch_sz = patch_sz
        features = min(self.train_readers[0].features, patch_sz * patch_sz)

        # Create placeholder
        self.input_corrupt = tf.placeholder(tf.float32, [None, features])
        self.input_original = tf.placeholder(tf.float32, [None, features])

        # Create param
        self.w = tf.Variable(tf.truncated_normal([features, features], stddev = 0.1))
        self.w_ = tf.Variable(tf.truncated_normal([features, features], stddev = 0.1))
        self.b = tf.Variable(tf.constant(0.1, shape = [1, features]))
        self.b_ = tf.Variable(tf.constant(0.1, shape = [1, features]))

        # Var saver
        self.saver = tf.train.Saver({
            'w': self.w,
            'w_': self.w_,
            'b': self.b,
            'b_': self.b_}, max_to_keep=30)

        # Feed-forward step
        self.h = tf.nn.sigmoid(tf.matmul(self.input_corrupt, self.w) + self.b)
        self.h_ = tf.nn.sigmoid(tf.matmul(self.h, self.w_) + self.b_)

        # Loss func
        self.sparsity = tf.map_fn(lambda x: ryo * tf.log(ryo / x) + (1 - ryo) * tf.log((1 - ryo) / (1 - x)) ,tf.reduce_mean(self.h, 0))
        self.error = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_original - self.h_), 1)) + regular / 2 * (tf.reduce_sum(tf.square(self.w)) + tf.reduce_sum(tf.square(self.w_))) + beta * tf.reduce_sum(self.sparsity)

        # Train
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.error)

        # Create session
        self.sess = tf.Session()

        # Initialization
        init = tf.initialize_all_variables()
        self.sess.run(init)
        if model != None:
            self.saver.restore(self.sess, model)


    def train(self, epoch = 50):
        errors = []
        temp = 0
        for i in xrange(epoch):
            count = 0
            for label, reader in enumerate(self.train_readers):
                for x,y in reader.read_patch(self.batch_sz, self.patch_sz):
                    feed = {self.input_corrupt: x, self.input_original: y}
                    h_, error, _ = self.sess.run([self.h_, self.error, self.train_step], feed_dict = feed)
                    print "%d batch, error is %g"%(count, error)
                    count += 1
                    temp += error

                    if count % 1000 == 0:
                        errors.append([count, temp / 1000])
                        temp = 0
        
        np.savetxt('da_training_err.csv', errors, fmt = '%f', delimiter=",")

    def test(self, test_reader):
        count = 0
        res = []
        for corrupt, original in test_reader.read_mat(1):
            x = test_reader.im2col(corrupt[0], self.patch_sz).T
            y = test_reader.im2col(original[0], self.patch_sz).T
            feed = {self.input_corrupt: x, self.input_original: y}
            cleared_patch = self.sess.run([self.h_], feed_dict = feed)
            cleared = np.clip(test_reader.reconstruct(cleared_patch[0].T, self.patch_sz, original[0].shape), 0, 1)
            res.append(util.calcPSNR(cleared, original[0]))
            print "%d, PSNR change %g -> %g"%(count, util.calcPSNR(corrupt[0], original[0]), res[-1])
            util.imsave(cleared, str(count))
            count += 1

        print "avg %g, std %g"%(np.mean(res), np.std(res))


    def save_model(self, path, sess = None):
        if sess is None:
            sess = self.sess

        self.saver.save(sess, path)


    def load_model(self, path, sess = None):
        if sess is None:
            sess = self.sess

        self.saver.restore(sess, path)


    def get_param(self):
        return {'w': self.w.eval(session = self.sess),
                'b': self.b.eval(session = self.sess),
                'w_': self.w_.eval(session = self.sess),
                'b_': self.b_.eval(session = self.sess)}


if __name__ == '__main__':
    train_reader1 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    train_reader2 = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    # test_reader = util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/classic")
    da = DenoisingAutoEncoder([train_reader1, train_reader2], model = 'model_da')
    da.train(1)
    da.test(test_reader)
    da.save_model('model_da')