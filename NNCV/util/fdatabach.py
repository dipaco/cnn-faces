import sys
import numpy as np
from PIL import Image
from math import ceil


def progressbar(step, total, bar_len, info='', symb_f='\u2588', symb_e='-'):
    # Muestra por pantalla una barra de progreso
    # total: Limite de la barra
    # step: progreso actual de la tarea en ejecucion
    # bar_len: longitud en caracteres de la barra
    # info: informacion adicional mostrada al lado del progreso
    # symb_f y symb_e: Simbolos usados para representar el progreso
    progress = (step / total) * 100.0
    step_bar = int(ceil(bar_len * (step / total)))
    bar = (symb_f * step_bar) + (symb_e * (bar_len - step_bar))

    sys.stdout.write('\r|%s| %.1f%s %s' % (bar, progress, '%', info))
    sys.stdout.flush()


def binary_file_dataset(bin_filename, ls_image, ls_label, size=(32, 32), dump=0):
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
            dicts = cpickle.load(fo)
            fo.close()
            return dicts

        return unpickle(bin_filename)
    else:
        return np.fromfile(bin_filename, dtype=typea)


if __name__ == '__main__':
    import argparse
    from os import listdir, path

    parser = argparse.ArgumentParser(description='Programa para generar los data_batch.bin de imagenes')
    group = parser.add_argument_group(title='Argumentos requeridos')
    group.add_argument('--file', help='Nombre del archivo .bin', required=True)
    group.add_argument('--path', help='Nombre de la carpeta contenedora', required=True)
    group.add_argument('--flabel', help='Archivo con la lista de etiquetas del dataset', required=True)

    parser.add_argument('--size', nargs=2, help='Dimensiones de la imagen reducida')

    args = parser.parse_args()

    limag = []
    llabel = []
    size = (32, 32)

    if args.size is not None:
        size = [
            int(args.size[0]),
            int(args.size[0])
        ]

    r = listdir(path=args.path)
    for item in r:
        if item.endswith((".jpg", ".png", ".gif", ".bmp", ".tif")):
            limag.append(path.join(args.path, item))

    with open(args.flabel, mode='r', encoding='utf8') as fl:
        for line in fl:
            if len(line) == 0:
                continue
            if line[-1] in ('\n', '\r'):
                line = line[:-1]
            llabel.append(line)

    if len(limag) != len(llabel):
        print("Error: len(img):{} != len(label):{}", len(limag), len(llabel))
        exit()

    binary_file_dataset(
        bin_filename=args.file,
        ls_image=limag,
        ls_label=llabel,
        size=size
    )
