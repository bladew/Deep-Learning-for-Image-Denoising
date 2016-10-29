import numpy as np
from PIL import Image
import os

def imread(imName):
    '''
    Read in an image and return its normalized gray-scale pixel values
    '''
    im = Image.open(imName).convert('L')
    W, H = im.size
    result = np.zeros((H, W))
    imPix = im.load()
    for i in xrange(W):
        for j in xrange(H):
            result[j, i] = imPix[i, j]
    result /= 255.0
    return result


def imsave(im, imName):
    '''
    Save a 2D numpy array as a PNG-format image
    '''
    im[im<0.0] = 0.0
    im[im>1.0] = 1.0
    I = Image.fromarray(np.uint8(np.round(im * 255)))
    I.save(imName, 'PNG')


def calcPSNR(gt, I):
    return 20 * np.log10(1.0 / np.std(gt - I))


def convert(dir_path):
    '''
    Convert all images under given path into grey-scale and save in the same path
    '''
    for file in os.listdir(dir_path):
        if file.endswith(".jpg"):
            name = dir_path+'/'+file
            res = imread(name)
            imsave(res, name.replace(".jpg", ".png"))


def select_img(pic_list, dir_path):
    '''
    Select images from the given path according to the list
    UNDER CONSTRUCTION

    pic_list: the txt filename containing target image names
    dir_path: the path holding images
    '''
    if not os.path.exists(dir_path + "/" + "vol68"):
        os.makedirs(dir_path + "/" + "vol68")

    with open(pic_list, 'r') as list_file:
        for img in list_file:
            print 


class ImReader(object):
    """
    Iterator for reading in images from the given path 
    """
    def __init__(self, path):
        super(ImReader, self).__init__()
        if not os.path.exists(path):
            print "no such path"
            return

        self.path = path


    def read(self, batch_sz = 1):
        for file in os.listdir(self.path):
            if file.endswith(".png"):
                yield imread(self.path + '/' + file)
        


if __name__ == '__main__':
    pass
    # convert("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    # convert("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    # convert("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    # select_img("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val68.txt", "/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")