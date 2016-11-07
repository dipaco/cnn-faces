import numpy as np
from PIL import Image
import sys


def progressbar(step, total, bar_len, info='', symb_f='\u2588', symb_e='-'):
    from math import ceil

    progress = (step/total) * 100.0
    step_bar = int(ceil(bar_len * (step / total)))
    bar = (symb_f * step_bar) + (symb_e * (bar_len - step_bar))

    sys.stdout.write('\r|%s| %.1f%s %s'%(bar,progress,'%',info))
    sys.stdout.flush()


def binary_file(bin_filename, ls_image, ls_label, size=(32, 32), dump=0):
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

    for k, fn in enumerate(ls_image):
        progressbar(k + 1, len(ls_image), bar_len=60, info=fn)

        i = Image.open(fn)
        i.thumbnail(size, Image.ANTIALIAS)
        padding = Image.new('RGBA', size, (0, 0, 0, 0))
        padding.paste(i, (int((size[0] - i.size[0]) / 2), int((size[1] - i.size[1]) / 2)))
        im = padding

        arr = np.array(im)
        lbl = [ls_label[k]]
        r = arr[:, :, 0].flatten()
        g = arr[:, :, 0].flatten()
        b = arr[:, :, 0].flatten()
        r = np.array(list(lbl) + list(r) + list(g) + list(b), np.uint8)
        ls.append(r)

    output = np.array(ls)

    if dump:
        output.dump(bin_filename)
    else:
        output.tofile(bin_filename)


def read_binary_file(bin_filename, dump=0, typea=np.uint8):
    if dump:
        def unpickle(file):
            import _pickle as cpickle
            fo = open(file, 'rb')
            dict = cpickle.load(fo)
            fo.close()
            return dict
        return unpickle(bin_filename)
    else:
        return np.fromfile(bin_filename, dtype=typea)


def image2grid(img_filename, s=10, height=32, width=32, save_img=False, limit_save_img=-1):
    # Image:
    #   +----+----+----+----+
    #   | 0  | 1  | 2  | 3  |
    #   +----+----+----+----+
    #   | 4  | 5  | 6  | 7  |
    #   +----+----+----+----+
    # grid:
    #   +----+----+----+----+----+----+----+----+
    #   | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
    #   +----+----+----+----+----+----+----+----+
    #
    with Image.open(img_filename) as im:
        imgwidth, imgheight = im.size
        grid = []
        k = 0
        ch = int((height - 1)/2)
        cw = int((width - 1)/2)
        for h in range(0, imgheight, s):
            for w in range(0, imgwidth, s):
                grid.append(im.crop((w - cw, h - ch, w - cw + width, h - ch + height)))
                if save_img:
                    grid[k].save('image_%d.jpg' % k)
                    k += 1
                    if k == limit_save_img:
                        save_img = False
        return grid


def binary_file(bin_filename, img_filename, step=10, size=(32, 32), box=[81, 81], save_img=False, limit_save=-1, dump=0):
    ls = []
    ls_images = image2grid(img_filename=img_filename,
                           s=step,
                           height=box[0],
                           width=box[1],
                           save_img=save_img,
                           limit_save_img=limit_save)
    ls_label = np.zeros(len(ls_images), dtype=int)

    for k, im in enumerate(ls_images):
        progressbar(k + 1, len(ls_images), bar_len=60, info='im_%d' % k)
        im.thumbnail(size, Image.ANTIALIAS)
        padding = Image.new('RGBA', size, (0, 0, 0, 0))
        padding.paste(im, (int((size[0] - im.size[0]) / 2), int((size[1] - im.size[1]) / 2)))
        im2 = padding

        arr = np.array(im2)
        lbl = [ls_label[k]]
        r = arr[:, :, 0].flatten()
        g = arr[:, :, 0].flatten()
        b = arr[:, :, 0].flatten()
        r = np.array(list(lbl) + list(r) + list(g) + list(b), np.uint8)
        ls.append(r)
    print()
    output = np.array(ls)
    if dump:
        output.dump(bin_filename)
    else:
        output.tofile(bin_filename)


def bin2img(bin_filename, img_filename, step=10):
    from matplotlib.pyplot import show, imshow, colorbar
    from scipy.misc import imresize
    from matplotlib.image import imread

    r = np.load(bin_filename)
    I = imread(img_filename)
    # r = np.load('logits_3_face.bin')
    # I = imread('3_face.jpg')

    def ishape(size=(350, 590), stp=5):
        k = 0
        j = 0
        for i in range(0, size[1], stp):
            j += 1
        for i in range(0, size[0], stp):
            k += 1
        return k, j

    # a = [r[i][0] for i in range(0, r.shape[0])]
    # ar = np.array(a)
    ar = r[:, 0]
    current_step = step
    s = ishape(stp=current_step)
    # s = (s[0] + 40 / current_step, s[1] + 40 / current_step)
    print(s)
    f = np.reshape(ar, s)

    f = (255 * (f - f.min()) / (f.max() - f.min())).astype('uint8')
    f = imresize(f, I.shape[:2])
    F = np.stack((f, f, f), 2)
    # rn = f.max() - f.min()

    # imshow(f > 0.9*rn + f.min(), cmap='gray')
    F = (0.8 * F + 0.2 * I).astype('uint8')
    imshow(F)
    colorbar()
    show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Programa para generar los data_batch.bin de una imagen')
    group = parser.add_argument_group(title='Argumentos requeridos')
    group.add_argument('--file', help='Nombre del archivo .bin', required=True)
    group.add_argument('--image', help='Nombre de la imagen', required=True)

    exgroup = group.add_mutually_exclusive_group(required=True)
    exgroup.add_argument('--img2bin', action='store_true', help='Genera un data_bach a partir de una imagen')
    exgroup.add_argument('--bin2img', action='store_true', help='Genera una imagen a partir de un archivo .bin')

    parser.add_argument('--step', help='Desplazamiento de box')
    parser.add_argument('--box', nargs=2, help='Dimensiones de box')
    parser.add_argument('--save_img', action='store_true', help='Guarda las img generadas por crop')
    parser.add_argument('--limit', help='Numero maximo de img a guardar por --save_img')

    args = parser.parse_args()

    st = None
    if args.step is not None:
        st = int(args.step)

    if args.img2bin:
        b = []
        l = -1
        if args.box is not None:
            b.append(int(args.box[0]))
            b.append(int(args.box[0]))
        if args.limit is not None:
            l = int(args.limit)

        if st is None and b == []:
            binary_file(bin_filename=args.file, img_filename=args.image, save_img=args.save_img, limit_save=l)
        elif st is not None and b == []:
            binary_file(bin_filename=args.file, img_filename=args.image, step=st, save_img=args.save_img, limit_save=l)
        elif st is None and len(b) > 0:
            binary_file(
                bin_filename=args.file,
                img_filename=args.image,
                box=b,
                save_img=args.save_img,
                limit_save=l
            )
        elif st is not None and len(b) > 0:
            binary_file(
                bin_filename=args.file,
                img_filename=args.image,
                step=st,
                box=b,
                save_img=args.save_img,
                limit_save=l
            )
    elif args.bin2img:
        if st is None:
            bin2img(bin_filename=args.file, img_filename=args.image)
        else:
            bin2img(bin_filename=args.file, img_filename=args.image, step=st)
        

