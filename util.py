import numpy as np
import scipy.io
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
    I.save(imName)


def mat_save(filename, img, original = None):
    if original is None:
        scipy.io.savemat(filename, {'img':img})
    else:
        scipy.io.savemat(filename, {'corrupted':img, 'original':original})


def mat_load(filename):
    return scipy.io.loadmat(filename)


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


def corrupt(dir_path):
    '''
    Corrupt the png file and save the pair in mat 
    '''
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            name = dir_path+'/'+file
            img = imread(name)
<<<<<<< HEAD
            mat_save(name.replace(".png", ".mat"), img + np.random.normal(0, 50/255.0, img.shape), img)
=======
            mat_save(name.replace(".png", ".mat"), img + np.random.normal(0, 50.0/255.0, img.shape), img)


def select_img(pic_list, dir_path):
    '''
    Select images from the given path according to the list
    UNDER CONSTRUCTION

    pic_list: the txt filename containing target image names
    dir_path: the path holding images
    '''
    # if not os.path.exists(dir_path + "/" + "vol68"):
    #     os.makedirs(dir_path + "/" + "vol68")

    # with open(pic_list, 'r') as list_file:
    #     for img in list_file:
    #         print 
    pass
>>>>>>> cnn


class ImReader(object):
    '''
    Iterator for reading in images from the given path 
    '''
    def __init__(self, path):
        super(ImReader, self).__init__()
        if not os.path.exists(path):
            print "no such path"
            return

        self.path = path
        self.w = 0
        self.h = 0
        for sample in self.read():
            self.h, self.w = sample[0].shape
            break
        self.features = self.w * self.h


    def read(self, batch_sz = 1, vector = False):
        batch = []
        for file in os.listdir(self.path):
            if file.endswith(".png"):
                if vector:
                    batch.append(np.reshape(imread(self.path + '/' + file), -1))
                else:
                    batch.append(imread(self.path + '/' + file))
                
                if len(batch) == batch_sz:
                    res = batch
                    batch = []
                    yield res


    def read_mat(self, batch_sz = 1, vector = False):
        x_batch, y_batch = [], []
        for file in os.listdir(self.path):
            if file.endswith(".mat"):
                d = mat_load(self.path + '/' + file)
                if vector:
                    x_batch.append(np.reshape(d['corrupted'], -1))
                    y_batch.append(np.reshape(d['original'], -1))
                else:
                    x_batch.append(d['corrupted'])
                    y_batch.append(d['original'])
                if len(x_batch) == batch_sz:
                    res_x, res_y = x_batch, y_batch
                    x_batch, y_batch = [], []
                    yield res_x, res_y


    def read_patch(self, batch_sz = 1, patch_sz = 6):
        x_batch, y_batch = [], []
        for file in os.listdir(self.path):
            if file.endswith(".mat"):
                d = mat_load(self.path + '/' + file)
                corrupted = self.im2col(d['corrupted'], patch_sz)
                original = self.im2col(d['original'], patch_sz)
                for i in xrange(batch_sz - 1, corrupted.shape[1], batch_sz):
                    yield corrupted[:, i - batch_sz + 1: i + 1].T, original[:, i - batch_sz + 1: i + 1].T


    def corrupt_and_read(self, batch_sz = 1, mean = 0, sigma = 25, vector = False):
        x_batch, y_batch = [], []
        for file in os.listdir(self.path):
            if file.endswith(".png"):
                original = imread(self.path + '/' + file)
                if vector:
                    x_batch.append(np.reshape(self.gaussian_noise(original, mean, sigma), -1))
                    y_batch.append(np.reshape(original, -1))
                else:
                    x_batch.append(self.gaussian_noise(original, mean, sigma))
                    y_batch.append(original)
                if len(x_batch) == batch_sz:
                    res_x, res_y = x_batch, y_batch
                    x_batch, y_batch = [], []
                    yield res_x, res_y

    
    def gaussian_noise(self, img, mean = 0, sigma = 25):
        '''
        Apply Gaussian noise to the img.

        img: original image
        mean: mean of the Gaussian func
        sigma: sigma of the Gaussian func

        return: corrupted image
        '''
        return img + np.random.normal(mean, sigma / 255.0, img.shape)


    def im2col(self, I, patchSize, stride=1):
        '''
        im2col (sliding) on gray-scale image
        '''
        if type(patchSize) is int:
            patchSize = [patchSize, patchSize]
        # Parameters
        I = I.T
        M, N = I.shape
        col_extent = N - patchSize[1] + 1
        row_extent = M - patchSize[0] + 1
        # Get Starting block indices
        start_idx = np.arange(patchSize[0])[:,None]*N + np.arange(patchSize[1])
        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(0,row_extent,stride)[:,None]*N + np.arange(0,col_extent,stride)
        # Get all actual indices & index into input array for final output
        idx = start_idx.ravel()[:,None] + offset_idx.ravel()
        out = np.take(I, idx)
        return out


    def reconstruct(self, cols, patch_sz, original_sz):
        '''
        reconstruct cols of patches into an image
        '''
        mm, nn = original_sz
        t = np.reshape(np.arange(mm * nn),(mm, nn))
        temp = self.im2col(t, [patch_sz, patch_sz]).flatten()
        I = np.bincount(temp, weights=cols.flatten())
        I /= np.bincount(temp)
        I = np.reshape(I, (mm, nn))
        return I

if __name__ == '__main__':
<<<<<<< HEAD
    pass
=======
    pass
    # convert("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/train")
    # convert("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/test")
    # convert("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    # select_img("/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val68.txt", "/home/zwang32/course/cs295k/Deep-Learning-for-Image-Denoising/images/val")
    corrupt("./images/train")
    corrupt("./images/test")
    corrupt("./images/val")



>>>>>>> cnn
