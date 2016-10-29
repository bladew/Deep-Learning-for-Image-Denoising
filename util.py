import numpy as np
from PIL import Image

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

