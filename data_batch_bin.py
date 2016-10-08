import numpy as np
from PIL import Image

def binary_file(bin_filename,ls_image,ls_label, size = (32,32)):
    """
    https://www.cs.toronto.edu/~kriz/cifar.html
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
    with 6000 images per class. There are 50000 training images and 10000 test images. 
	
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 
         colour image. The first 1024 entries contain the red channel values, 
         the next 1024 the green, and the final 1024 the blue. The image is stored 
         in row-major order, so that the first 32 entries of the array are the red channel 
         values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i 
         indicates the label of the ith image in the array data.

    Format:
        <1 x label><3072 x pixel>
        ...
        <1 x label><3072 x pixel>

     The first byte is the label of the first image, which is a 
     number in the range 0-9. The next 3072 bytes are the values of 
     the pixels of the image. The first 1024 bytes are the red channel 
     values, the next 1024 the green, and the final 1024 the blue. 
     The values are stored in row-major order, so the first 32 bytes 
     are the red channel values of the first row of the image. 
    """
    ls = []
    for k,fn in enumerate(ls_image):
        i = Image.open(fn)
        
        i.thumbnail(size, Image.ANTIALIAS)        
        padding = Image.new('RGBA', size, (0, 0, 0, 0))
        padding.paste(i,(int((size[0] - i.size[0]) / 2), int((size[1] - i.size[1]) / 2)))
        im = padding

        arr = np.array(im)        
        lbl = [ls_label[k]]
        r = arr[:,:,0].flatten()
        g = arr[:,:,0].flatten()
        b = arr[:,:,0].flatten()
        r = np.array(list(lbl) + list(r) + list(g) + list(b), np.uint8)
        ls.append(r)

    output = np.array(ls)
    output.dump(bin_filename)




