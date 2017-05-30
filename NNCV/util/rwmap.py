import sys
import numpy as np
from PIL import Image
from matplotlib.pyplot import show, imshow, colorbar
from math import ceil, floor


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


def modsize(imgwidth, imgheight):

    # determinar las dimensiones de la imagen resultado
    # ============================================================
    hght = wght = 0     
    if imgwidth == imgheight:
        hght = wght = 100
    elif imgwidth > imgheight:
        wght = 100
        hght = floor(100 * imgheight / float(imgwidth))          
    else:
        hght = 100
        wght = floor(100 * imgwidth / float(imgheight))            
    
    # aumento de las dimensiones para generar una division entera
    # ============================================================
    # width
    m = imgwidth % wght
    if m != 0:
        a = wght - m
        imgwidth += a
    # height
    m = imgheight % hght
    if m != 0:
        a = hght - m
        imgheight += a

    # calculo del desplazamiento de la ventana
    # ============================================================
    sw = int(imgwidth / float(wght))
    sh = int(imgheight / float(hght))
    
    return imgwidth, imgheight, sw, sh


def image2grid(img_filename, height=81, width=81, factor=1, save_img=False, limit_save_img=-1):
    
    # Imagen
    #   +----+----+----+----+   
    #   | 0  | 1  | 2  | 3  |         +----+----+----+----+----+----+----+----+
    #   +----+----+----+----+   =>    | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
    #   | 4  | 5  | 6  | 7  |         +----+----+----+----+----+----+----+----+
    #   +----+----+----+----+
    # 
    
    with Image.open(img_filename) as im:
        
        # Modifica las dimensiones de la imagen
        # ============================================================
        if factor != 1:
            _width, _height = im.size
            _width *= factor
            _height *= factor
            im = im.resize((floor(_width), floor(_height)))

        imgwidth, imgheight = im.size        
        
        imgwidth, imgheight, sw, sh = modsize(imgwidth=imgwidth, imgheight=imgheight)

        # genera la particiones de la imagen 
        # ============================================================
        grid = []
        k = 0
        ch = int((height - 1) / 2)
        cw = int((width - 1) / 2)
        for h in range(0, imgheight, sh):
            for w in range(0, imgwidth, sw):                
                grid.append(im.crop((w - cw, h - ch, w + cw, h + ch)))
                # save_img: guarda las imagenes generadas por crop
                # limit_save_img: opcional, numero total de imagenes a disco
                if save_img:
                    grid[k].save('image_%d.jpg' % k)
                    k += 1
                    if k == limit_save_img:
                        save_img = False

        return grid


def binary_file(bin_filename, img_filename, size=(32, 32), box=(81, 81),
                factor=1, save_img=False, limit_save=-1, dump=0):
    ls = []
    ls_images = image2grid(img_filename=img_filename,                           
                           height=box[0],
                           width=box[1],
                           factor=factor,
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

    from os.path import splitext
    file, ext = splitext(bin_filename)
    file += '_' + str(len(ls_images))
    if dump:
        output.dump(file + ext)
    else:
        output.tofile(file + ext)


def bin2img(bin_filename, img_filename, ishow=True):
    # carga la imagen y los logits
    r = np.load(bin_filename)    
    I = Image.open(img_filename)

    imgwidth, imgheight = I.size    

    imgwidth, imgheight, sw, sh = modsize(imgwidth=imgwidth, imgheight=imgheight)
    
    width = len(range(0, imgwidth, sw))
    height = len(range(0, imgheight, sh))

    ar = r[:, 0]
    f = np.reshape(ar, (height, width))
    f[f < 0] = 0

    # f = (255 * (f - f.min()) / (f.max() - f.min())).astype('uint8')

    if ishow:
        f2 = (255 * (f - f.min()) / (f.max() - f.min())).astype('uint8')
        F = np.stack((f2, f2, f2), 2)
        imshow(F)
        colorbar()
        show()

    return f
