#import tensorflow as tf
import sys
import numpy as np
sys.path.insert(0, '..')

import util

class DenoisingAutoEncoder(object):
    """Denoising Auto-Encoder"""
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()
        self.arg = arg


if __name__ == '__main__':
	for result in util.ImReader("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test").read():
		print result
		break