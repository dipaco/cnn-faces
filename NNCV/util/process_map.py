import numpy as np
import pickle
from PIL import Image
from matplotlib.pyplot import show, imshow, colorbar
from matplotlib.image import imread
from scipy.misc import imresize
from scipy import ndimage
from math import ceil, floor
from skimage import measure

from util.rwmap import bin2img


def process_map(filen, ind, dirf):
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/ndimage.html
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.label.html
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.center_of_mass.html
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.find_objects.html
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    pass


def make_strc(data):
    pass


def mergemaps(maps):
    pass


if __name__ == '__main__':
    pass
