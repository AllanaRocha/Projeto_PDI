import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from filtros import *
# from otsu import otsu

def show_imgs(res, plt):
    cv2.imshow('image',res)
    cv2.waitKey(0)
    plt.show()
    
def save_imgs(folder_path, namefile, res, plt=''):
    print("Processando -> ", namefile)
    cv2.imwrite(folder_path+"result_"+namefile, res)
    if plt != '':
        plt.savefig(folder_path+"result_histograma_"+namefile)

def remover_fundo(path_img):
    img = cv2.imread(path_img, 0)#gray
    tamanho = img.shape
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
    
    img_bin_otsu = binarizacao_otsu(img)
    img_mediana = filtro_mediana(img_bin_otsu, 3)

    #concatenando as imagens
    res = cv2.hconcat([img, img_mediana])
    
    #montando o histograma
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    #TODO: OS HISTOGRAMAS ESTAO SENDO PLOTADOS ERRADOS 
    # histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    # axes[0].plot(histogram)
    # axes[0].set_title('Img Original - Hitograma')
    # axes[0].set_xlabel('Valor do Pixel')
    # axes[0].set_ylabel('Pixels')
    
    # histogram_proc, bin_edges_proc = np.histogram(img_mediana, bins=256, range=(0, 256))
    # axes[1].plot(histogram_proc)
    # axes[1].set_title('Img Processada - Hitograma')
    # axes[1].set_xlabel('Valor do Pixel')
    # axes[1].set_ylabel('Pixels')
    
    return res, plt

def segmentar_cachorro(path_img):
    img = cv2.imread(path_img, 0)#gray
    tamanho = img.shape
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
    
    mask = binarizacao_otsu(img)
    
    # mascara negativa
    mask = 255 - mask
    
    # trecho linear para que 127.5 vá para 0, mas 255 permaneca 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    
    #Aplicando operacao "AND" entre mask e img
    result = mask & img
    result = filtro_dilatacao(result)

    cv2.imshow('image',result)
    cv2.waitKey(0)
    plt.show()
    
    #concatenando as imagens
    res = cv2.hconcat([img, result])
    
    #montando o histograma
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    #TODO: OS HISTOGRAMAS ESTAO SENDO PLOTADOS ERRADOS 
    # histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    # axes[0].plot(histogram)
    # axes[0].set_title('Img Original - Hitograma')
    # axes[0].set_xlabel('Valor do Pixel')
    # axes[0].set_ylabel('Pixels')
    
    # histogram_proc, bin_edges_proc = np.histogram(img_mediana, bins=256, range=(0, 256))
    # axes[1].plot(histogram_proc)
    # axes[1].set_title('Img Processada - Hitograma')
    # axes[1].set_xlabel('Valor do Pixel')
    # axes[1].set_ylabel('Pixels')
    
    return res, plt

def remover_ruido(path_img):
    img = cv2.imread(path_img, 0)#gray
    tamanho = img.shape
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
    
    img_mediana = filtro_mediana(img, 3)
    img_mediana = filtro_mediana(img_mediana, 3)
    
    img_dilation = filtro_dilatacao(img_mediana)
    img_dilation = filtro_mediana(img_dilation, 3)
    
    img_erosion = filtro_erosao(img_dilation, k=5)
    
    img_dilation = filtro_dilatacao(img_erosion)

    #concatenando as imagens
    res = cv2.hconcat([img, img_dilation])
    
    #montando o histograma
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    #TODO: OS HISTOGRAMAS ESTAO SENDO PLOTADOS ERRADOS 
    # histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    # axes[0].plot(histogram)
    # axes[0].set_title('Img Original - Hitograma')
    # axes[0].set_xlabel('Valor do Pixel')
    # axes[0].set_ylabel('Pixels')
    
    # histogram_proc, bin_edges_proc = np.histogram(img_mediana, bins=256, range=(0, 256))
    # axes[1].plot(histogram_proc)
    # axes[1].set_title('Img Processada - Hitograma')
    # axes[1].set_xlabel('Valor do Pixel')
    # axes[1].set_ylabel('Pixels')
    
    return res, plt

def preencher_logo(path_img):
    img = cv2.imread(path_img, 0)#gray
  
    img_erosion = filtro_erosao(img, k=5)
    img_closing = fechamento(img_erosion)

    res = cv2.hconcat([img, img_closing])
    
    #montando o histograma
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    #TODO: OS HISTOGRAMAS ESTAO SENDO PLOTADOS ERRADOS 
    # histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    # axes[0].plot(histogram)
    # axes[0].set_title('Img Original - Hitograma')
    # axes[0].set_xlabel('Valor do Pixel')
    # axes[0].set_ylabel('Pixels')
    
    # histogram_proc, bin_edges_proc = np.histogram(img_mediana, bins=256, range=(0, 256))
    # axes[1].plot(histogram_proc)
    # axes[1].set_title('Img Processada - Hitograma')
    # axes[1].set_xlabel('Valor do Pixel')
    # axes[1].set_ylabel('Pixels')
    
    return res

def mirror_mermaid(path):
    mermaid = cv2.imread(path + 'image_03a.png')
    fin = cv2.imread(path + 'image_03b.png')
    shark = cv2.imread(path + 'image_03c.png')
    castle = cv2.imread(path + 'image_03d.png')
    triangle = cv2.imread(path + 'image_03e.png')
    human = cv2.imread(path + 'image_03f.png')

    human_half = or_op(human,triangle)
    not_fin = not_op(fin)
    shark = xor_op(not_fin,shark)
    shark = not_op(shark)
    not_castle = not_op(castle)
    fish_half = or_op(shark,not_castle)
    mirror_mermaid = and_op(human_half,fish_half)
    mermaid_n_mirror_mermaid = and_op(mermaid,mirror_mermaid)
    return mermaid_n_mirror_mermaid

if __name__ == '__main__':
    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    path_database_imgs = os.path.join(pathname, 'imgs', '')
    path_database_result = os.path.join(pathname, 'result', '')
    
    #------------- SEPARAR TEXTO -----------------#
    # res_1, plt_1 = remover_fundo(path_database_imgs+"image_01.png")
    # show_imgs(res_1, plt_1)
    # save_imgs(path_database_result, "image_01.png", res_1, plt_1)
    
   #------------- SEGMENTAR CACHORRO -----------------#
    # res_2, plt_2 = segmentar_cachorro(path_database_imgs+"image_02.png")
    # show_imgs(res_2, plt_2)
    # save_imgs(path_database_result, "image_01.png", res_1, plt_1)
    
    #------------- REMOVER RUIDO COELHO -----------------#
    # res_3, plt_3 = remover_ruido(path_database_imgs+"image_04.png")
    # show_imgs(res_3, plt_3)
    # save_imgs(path_database_result, "image_01.png", res_1, plt_1)
    
    #------------- PREENCHER LOGO -----------------#
    res_4 = preencher_logo(path_database_imgs+"image_05.png")
    # show_imgs(res_4, plt_4)
    save_imgs(path_database_result, "image_05.png", res_4)

    #------------- SEREIA -----------------#
    # res_5 = mirror_mermaid(path_database_imgs)
    # save_imgs(path_database_result,'image_03.png',res_5)