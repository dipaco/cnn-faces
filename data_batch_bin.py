import numpy as np
from PIL import Image

def progressB(step, total, bar_len, info = '', symb_f = '\u2588', symb_e = '-'):
    import sys
    from math import ceil

    progress = (step/total) * 100.0
    step_bar = int(ceil(bar_len * (step / total)))
    bar = (symb_f * step_bar) + (symb_e * (bar_len - step_bar))

    sys.stdout.write('\r|%s| %.1f%s %s'%(bar,progress,'%',info))
    sys.stdout.flush()
	


def binary_file(bin_filename,ls_image,ls_label, size = (32,32), dump = 0):
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
        progressB(k, len(ls_image), bar_len = 60, info = fn)
	
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

    if dump:
        output.dump(bin_filename)
    else:
        output.tofile(bin_filename)

def binary_file(bin_filename,image):
    pass

def read_binary_file(bin_filename, dump = 0):
    if dump:
        def unpickle(file):
            import _pickle as cPickle
            fo = open(file, 'rb')
            dict = cPickle.load(fo)
            fo.close()
            return dict
        return unpickle(bin_filename)
    else:
        return np.fromfile(bin_filename,dtype = np.uint8)


def arr_image(filename, height = 32 ,width = 32, hwp = False):
    # Image:
    #   +----+----+----+----+
    #   | 0  | 1  | 2  | 3  |
    #   +----+----+----+----+
    #   | 4  | 5  | 6  | 7  |
    #   +----+----+----+----+
    # 
    with Image.open(filename) as im:
        imgwidth, imgheight = im.size
        grid = []
        if hwp:
            height = width = int((imgwidth + imgheight) / 8)
        for h in range(0, imgheight, height):
            for w in range(0,imgwidth,width):
                grid.append(im.crop((w, h, w+width, h+height)))
        return grid       

def binary_file(bin_filename,im_filename, size = (32,32), dump = 0):
    ls = []
    ls_images = arr_image(im_filename, hwp = True)
    ls_label = [0 for i in range(0,len(ls_images))]
    
    for k,im in enumerate(ls_images): 
        im.thumbnail(size, Image.ANTIALIAS)        
        padding = Image.new('RGBA', size, (0, 0, 0, 0))
        padding.paste(im,(int((size[0] - im.size[0]) / 2), int((size[1] - im.size[1]) / 2)))
        im2 = padding        
        
        arr = np.array(im2)
        lbl = [ls_label[k]]
        r = arr[:,:,0].flatten()
        g = arr[:,:,0].flatten()
        b = arr[:,:,0].flatten()
        r = np.array(list(lbl) + list(r) + list(g) + list(b), np.uint8)
        ls.append(r)   

        progressB(k+1, len(ls_images), bar_len = 60, info = 'im_%d'%k)

    output = np.array(ls)
    if dump:
        output.dump(bin_filename)
    else:
        output.tofile(bin_filename)

        
if __name__ == '__main__':
    import sys    
    if len(sys.argv) > 2:
        bin_filename = sys.argv[1]
        im_filename = sys.argv[2]
        binary_file(bin_filename,im_filename)
    
   