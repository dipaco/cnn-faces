import sys
import numpy as np
from PIL import Image
from matplotlib.pyplot import show, imshow, colorbar
from matplotlib.image import imread
from scipy.misc import imresize
from scipy import ndimage
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


def binary_file(bin_filename, img_filename, size=(32, 32), box=[81, 81],
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


def bin2img(bin_filename, img_filename, fade=0.8, step=10, factor=1, ishow = True):
    # carga la imagen y los logits
    r = np.load(bin_filename)
    I = imread(img_filename)
    
    # almacena las dimensiones original de la imagen, se usa para retornar el mapa
    # con igual dimension a la imagen procesada
    IShape = I.shape
    
    # Modifica las dimensiones de la imagen
    if factor != 1:
        _height, _width, _ = I.shape
        _width *= factor
        _height *= factor
        I = imresize(I, (int(_height), int(_width)))
        
       
    # Proceso para calcular los bordes agregados a la imagen en la etapa
    # de generacion del data_bach. Necesario para la correcta redimension 
    # de los logits
    im = Image.fromarray(I)
    imgwidth, imgheight = im.size

    modw = (imgwidth + 1) % step
    modh = (imgheight + 1) % step    
    
    # bordes laterales
    if modw != 0:
        it = int(ceil(imgwidth/float(step)) * step + 1 - imgwidth)
        itrl = it        
        imgwidth += it
        
    # bordes superior e inferior 
    if modh != 0:        
        it = int(ceil(imgheight / float(step)) * step + 1 - imgheight)
        ittd = it        
        imgheight += it

    
    # obtiene las dimensiones necesarias para la redimension de los logits
    def ishape(size, stp=10):
        return len(range(0, size[0], stp)), len(range(0, size[1], stp))

    ar = r[:, 0]
    s = ishape(size=(imgheight, imgwidth), stp=step) 
    f = np.reshape(ar, s)
    f = (255 * (f - f.min()) / (f.max() - f.min())).astype('uint8')
    
    f = imresize(f, I.shape[:2])

    # muestra el mapa generado
    if ishow:
        F = np.stack((f, f, f), 2)

        F = (fade * F + (1.0 - fade) * I).astype('uint8')
        imshow(F)
        colorbar()
        show()

    '''
    f2 = f.copy()
    while np.max(f2) != np.min(f2):
        m = np.max(f2)
        i, j = np.where(f2 == m)
        y = np.zeros(f2.shape)
        y[i[0] - 40 : i[0] + 40 + 1, j[0] - 40 : j[0] + 40 + 1] = 255

        F = np.stack((y, y, y), 2)

        F = (fade * F + (1.0 - fade) * I).astype('uint8')
        imshow(F)
        colorbar()
        show()

        f2[i[0], j[0]] = np.min(f2)
    '''

    return imresize(f, IShape[:2])


def mergemaps(maps):
    mpfm = np.maximum(maps[0], maps[1])
    mpfm = np.maximum(mpfm, maps[2])
    mpfm = np.maximum(mpfm, maps[3])
    # np.maximum([2, 3, 4], [1, 5, 2]) => array([2, 5, 4])
    return mpfm

def mulprom(filen, ind, dirf, dirh, threshold=100):  
    from skimage import measure
    import pickle
    
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/ndimage.html    
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.label.html
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.center_of_mass.html
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.find_objects.html
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    
    # mapa de las caras
    fmaps = []
    fmaps.append(bin2img(bin_filename='%s/logits_1#1_%d.bin'%(dirf, ind), img_filename=filen, step=10, ishow=False))
    fmaps.append(bin2img(bin_filename='%s/logits_2#3_%d.bin'%(dirf, ind), img_filename=filen, step=10, factor=2/3, ishow=False))
    fmaps.append(bin2img(bin_filename='%s/logits_4#9_%d.bin'%(dirf, ind), img_filename=filen, step=10, factor=4/9, ishow=False))
    fmaps.append(bin2img(bin_filename='%s/logits_3#2_%d.bin'%(dirf, ind), img_filename=filen, step=10, factor=3/2, ishow=False))
    fmf = mergemaps(fmaps)    
    imshow(fmf)
    colorbar()
    show()
    
    # mapa de los sombreros
    hmaps = []
    hmaps.append(bin2img(bin_filename='%s/logits_1#1_%d.bin'%(dirh, ind), img_filename=filen, step=10, ishow=False))
    hmaps.append(bin2img(bin_filename='%s/logits_2#3_%d.bin'%(dirh, ind), img_filename=filen, step=10, factor=2/3, ishow=False))
    hmaps.append(bin2img(bin_filename='%s/logits_4#9_%d.bin'%(dirh, ind), img_filename=filen, step=10, factor=4/9, ishow=False))
    hmaps.append(bin2img(bin_filename='%s/logits_3#2_%d.bin'%(dirh, ind), img_filename=filen, step=10, factor=3/2, ishow=False))
    hmf = mergemaps(hmaps)    
    imshow(hmf)
    colorbar()
    show()
    
    # Calculo del promedio de los mapas de caras y sombreros
    prom = np.mean(np.array([fmf, hmf]), axis=0).astype('uint8')
    
    # Calculo de la multiplicacion de los mapas de caras y sombreros
    im1 = (1 * (fmf - fmf.min()) / (fmf.max() - fmf.min()))
    im2 = (1 * (hmf - hmf.min()) / (hmf.max() - hmf.min()))
    mult = (im1 * im2)
    mult = (255 * (mult - mult.min()) / (mult.max() - mult.min())).astype('uint8')
    
    #imshow(prom)
    #colorbar()
    #show()    
    #imshow(mult)
    #colorbar()
    #show()
    
    # Umbralizacion 
    mp_caras_seg = (fmf > threshold).astype('uint8')
    mp_sombr_seg = (hmf > threshold).astype('uint8')    
    ims_prom = (prom > threshold).astype('uint8')
    ims_mult = (mult > threshold).astype('uint8')
    
    #imshow(mp_caras_seg, cmap='gray')
    #show()    
    #imshow(mp_sombr_seg, cmap='gray')
    #show()    
    #imshow(ims_prom, cmap='gray')
    #show()    
    #imshow(ims_mult, cmap='gray')
    #show()
    
    # etiquetado
    labeled_img_caras, num_features_caras = ndimage.label(mp_caras_seg)
    labeled_img_sombr, num_features_sombr = ndimage.label(mp_sombr_seg)
    print('num_label_caras: ', num_features_caras)
    print('num_label_sombr: ', num_features_sombr)
    
    #iml = (255 * (labeled_img_caras - labeled_img_caras.min()) / (labeled_img_caras.max() - labeled_img_caras.min())).astype('uint8')
    #imshow(iml)
    #show()
    #imk = (255 * (labeled_img_sombr - labeled_img_sombr.min()) / (labeled_img_sombr.max() - labeled_img_sombr.min())).astype('uint8')
    #imshow(imk)
    #show()
      
    # Slices correspondiente al minimo Paralelepipedo que contiene a cada objecto descrito por las etiquetas
    #pos_lprom = ndimage.find_objects(labeled_img_prom)
    #pos_lmult = ndimage.find_objects(labeled_img_mult)
    #--------------------------------------------------------
    # centros de masa
    #cmc = ndimage.center_of_mass(mp_caras_seg, labels=labeled_img_caras,  index=range(1, num_features_caras + 1))
    #cms = ndimage.center_of_mass(mp_sombr_seg, labels=labeled_img_sombr,  index=range(1, num_features_sombr + 1))

    # Measure properties of labeled image regions:
    # - area:        Number of pixels of region
    # - centroid:    Centroid coordinate tuple (row, col)
    # - label:       The label in the labeled input image
    # - coords:      Coordinate list (row, col) of the region
    # - filled_area: Number of pixels of filled region.
    # - image:       Sliced binary region image which has the same size as bounding box
    properties_list_caras = measure.regionprops(labeled_img_caras)
    properties_list_sombr = measure.regionprops(labeled_img_sombr)
    print('num_properties_caras: ', len(properties_list_caras))
    print('num_properties_sombr: ', len(properties_list_sombr))
    
    # Diferencia
    npcmc = np.array([cmc.centroid for cmc in properties_list_caras])
    npcms = np.array([cms.centroid for cms in properties_list_sombr])
    diferencia = np.array([(npcmc - cmi)**2 for cmi in npcms])
    
    # Intersección de regiones
    b_label_caras = [labeled_img_caras == x for x in range(1, num_features_caras + 1)]
    b_label_sombr = [labeled_img_sombr == x for x in range(1, num_features_sombr + 1)]    
    mat_interseccion = [l1 & l2 for l1 in b_label_caras for l2 in b_label_sombr]    
    val_interseccion = [True in mt for mt in mat_interseccion] 
    
    '''
    Ejemplo:
    --------------------------------------------
    
    a = np.array(([0,0,0,0],
                  [1,1,0,0],
                  [1,1,0,1],
                  [0,0,0,0],
                  [0,1,1,1]))

    b = np.array(([0,1,1,0],
                  [0,1,0,0],
                  [0,0,0,0],
                  [0,0,1,1],
                  [0,0,1,1]))
              
    lbla, nfa  = ndimage.label(a)
    lblb, nfb  = ndimage.label(b)
    
    lbla
    array([[0, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 0],
           [0, 3, 3, 3]])
           
    lblb
    array([[0, 1, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 2, 2],
           [0, 0, 2, 2]])
       
    y1 = [lbla == x for x in range(1, nfa + 1)]
    y2 = [lblb == x for x in range(1, nfb + 1)]
    
    y1
    [array([[False, False, False, False],
           [ True,  True, False, False],
           [ True,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False,  True],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False,  True,  True,  True]], dtype=bool)]
        
    y2
    [array([[False,  True,  True, False],
           [False,  True, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False,  True,  True],
           [False, False,  True,  True]], dtype=bool)]
    
    ma = [v1 & v2 for v1 in y1 for v2 in y2]
    ma
    [array([[False, False, False, False],
           [False,  True, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool), 
    array([[False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False, False, False],
           [False, False,  True,  True]], dtype=bool)]
    
    val = [True in m for m in ma]    
    val
    [True, False, False, False, False, True]
    
    '''
        
    # Areas
    area_inter = [np.count_nonzero(mti) for mti in mat_interseccion]
    
    
    
    # estructura
    strc = {
        'image': filen, # imagen original
        'threshold' : threshold, # valor usado para la umbralizacion
        'mapf' : fmf,       # mapa de caras
        'maph' : hmaps,     # mapa de sombreros
        'prom' : prom,      # promedio de los dos mapas
        'mult' : mult,      # multiplicacion de los dos mapas
        'lblf' : (labeled_img_caras, num_features_caras),   # etiquetado del mapa de caras
        'lblh' : (labeled_img_sombr, num_features_sombr),   # etiquetado del mapa de sombreros
        'propf' : properties_list_caras,    # lista de propiedades de cada etiqueta en el mapa de caras
        'proph' : properties_list_sombr,    # lista de propiedades de cada etiqueta en el mapa de sombreros
        'dis2c' : diferencia,  # distancia al cuadrado entre cada centroide de las etiquetas de los dos mapas
        'inters' : {           # Interseccion de los dos mapas
        #    'bmati' : mat_interseccion,     # matriz de intersecciones entre las regiones de las etiquetas de los dos mapas
            'bvali' : val_interseccion,     # valor True/False de la interseccion
            'areai' : area_inter            # numero de pixel que conforman las intersecciones
        }
    }
    
    # clave, descripcion
    keys = [
        ('image', 'imagen original'),
        ('threshold', 'valor usado para la umbralizacion'),
        ('mapf' , 'mapa de caras'),
        ('maph' , 'mapa de sombreros'),
        ('prom' , 'promedio de los dos mapas'),
        ('mult' , 'multiplicacion de los dos mapas'),
        ('lblf' , 'etiquetado del mapa de caras'),
        ('lblh' , 'etiquetado del mapa de sombreros'),
        ('propf' , 'lista de propiedades de cada etiqueta en el mapa de caras'),
        ('proph' , 'lista de propiedades de cada etiqueta en el mapa de sombreros'),
        ('dis2c' , 'distancia al cuadrado entre cada centroide de las etiquetas de los dos mapas'),
        ('inters' , 'Interseccion de los dos mapas', [           
            ('bmati' , 'matriz de intersecciones entre las regiones de las etiquetas de los dos mapas'),
            ('bvali' , 'valor True/False de la interseccion'),
            ('areai' , 'numero de pixel que conforman las intersecciones')
        ])
    ]
    
        
    from os.path import splitext
    file, ext = splitext(filen)
    with open(filen + '.d', 'wb') as f:
        data = (strc, keys)    
        pickle.dump(data, f)
    '''
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
       
    print('\n{:<5} {:<5} ({:<8},{:<8}) ({:<8},{:<8}) ({:<10},{:<10})'.format(
        'Lbl1',
        'Lbl2',
        'CMx L1',
        'CMy L1',
        'CMx L2',
        'CMy L2',
        'Dif x',
        'Dif y'
    ))
    print('-' * (len(formato) + 10))
    for i in range(1, 5 + 1):#num_features_prom + 1):
        for j in range(1, 3 + 1):#num_features_mult + 1):
            print(formato.format(
                i, 
                j, 
                cmp[i - 1][1], 
                cmp[i - 1][0], 
                cmm[j - 1][1], 
                cmm[j - 1][0], 
                diferencia[j - 1][i - 1][1], 
                diferencia[j - 1][i - 1][0])
            )'''

# python data_batch.py --image=img1.jpg --file=1 --mulprom --threshold=190
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Programa para generar los data_batch.bin de una imagen')
    group = parser.add_argument_group(title='Argumentos requeridos')
    group.add_argument('--file', help='Nombre del archivo .bin', required=True)
    group.add_argument('--image', help='Nombre de la imagen', required=True)

    exgroup = group.add_mutually_exclusive_group(required=True)
    exgroup.add_argument('--img2bin', action='store_true', help='Genera un data_bach a partir de una imagen')
    exgroup.add_argument('--bin2img', action='store_true', help='Genera una imagen a partir de un archivo .bin')
    exgroup.add_argument('--mulprom', action='store_true', help='')

    parser.add_argument('--box', nargs=2, help='Dimensiones de box')
    parser.add_argument('--save_img', action='store_true', help='Guarda las img generadas por crop')
    parser.add_argument('--limit', help='Numero maximo de img a guardar por --save_img')
    parser.add_argument('--pyramid', action='store_true', help='')
    parser.add_argument('--factor', help='')
    
    parser.add_argument('--threshold', help='')
    #parser.add_argument('--dirf', help='')
    #parser.add_argument('--dirh', help='')

    args = parser.parse_args()

    if args.img2bin:
        bx = [81, 81]
        l = -1
        if args.box is not None:
            bx.append(int(args.box[0]))
            bx.append(int(args.box[0]))
        if args.limit is not None:
            l = int(args.limit)
        # num#den_batch_ind_bsize.bin
        binary_file(
            bin_filename='1#1_' + args.file,
            img_filename=args.image,            
            box=bx,
            save_img=args.save_img,
            limit_save=l
        )
        if args.pyramid:            
            binary_file(
                bin_filename='2#3_' + args.file,
                img_filename=args.image,                
                box=bx,
                factor=2/3,
                save_img=args.save_img,
                limit_save=l
            )
            binary_file(
                bin_filename='4#9_' + args.file,
                img_filename=args.image,                
                box=bx,
                factor=4/9,
                save_img=args.save_img,
                limit_save=l
            )
            binary_file(
                bin_filename='3#2_' + args.file,
                img_filename=args.image,                
                box=bx,
                factor=3/2,
                save_img=args.save_img,
                limit_save=l
            )
    elif args.bin2img:
        pyrm = 1
        if args.factor is not None:
            pyrm = args.factor
            if pyrm == '2/3':
                bin2img(bin_filename=args.file, img_filename=args.image, step=st, factor=2/3)
            elif pyrm == '3/2':
                bin2img(bin_filename=args.file, img_filename=args.image, step=st, factor=3/2)
            elif pyrm == '4/9':
                bin2img(bin_filename=args.file, img_filename=args.image, step=st, factor=4/9)
        else:
            bin2img(bin_filename=args.file, img_filename=args.image, step=st)
    elif args.mulprom:
        mulprom(
            filen=args.image, 
            ind=int(args.file), 
            dirf='faces', 
            dirh='hat',
            threshold=int(args.threshold)
        )
  
