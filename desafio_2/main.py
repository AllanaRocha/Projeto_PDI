import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from filtros import *
from conversores import *

def show_imgs(res, plt):
    cv2.imshow('image',res)
    cv2.waitKey(0)
    plt.show()
    
def save_imgs(folder_path, namefile, res, plt):
    print("Processando -> ", namefile)
    cv2.imwrite(folder_path+"result_"+namefile, res)
    plt.savefig(folder_path+"result_histograma_"+namefile)

def clarear(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
        
    #convertendo do espaco de cores RGB para HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    #equalizando o histograma no canal V    
    img_hsv[:, :, 2] = equalizar_histograma(img_hsv[:, :, 2])
    
    #convertendo de volta de HSV para RGB 
    img_equ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    #concatenando as imagens
    res = cv2.hconcat([img, img_equ])
    
    #Montando os histogramas
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0, 256))
        axes[0].plot(bin_edges[0:-1], histogram, color=c)
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Valor do Pixel')
    axes[0].set_ylabel('Pixels')
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img_equ[:, :, channel_id], bins=256, range=(0, 256))
        axes[1].plot(bin_edges[0:-1], histogram, color=c)
    axes[1].set_title('Img Clareada - Hitograma')
    axes[1].set_xlabel('Valor do Pixel')
    axes[1].set_ylabel('Pixels')

    return res, plt
    
def escurecer(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
        
    #convertendo do espaco de cores RGB para HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    #equalizando o histograma no canal V    
    img_hsv[:, :, 2] = equalizar_histograma(img_hsv[:, :, 2])
    
    #convertendo de volta de HSV para RGB 
    img_equ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    #concatenando as imagens
    res = cv2.hconcat([img, img_equ])
    
    #Montando os histogramas
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0, 256))
        axes[0].plot(bin_edges[0:-1], histogram, color=c)
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Valor do Pixel')
    axes[0].set_ylabel('Pixels')
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img_equ[:, :, channel_id], bins=256, range=(0, 256))
        axes[1].plot(bin_edges[0:-1], histogram, color=c)
    axes[1].set_title('Img Escurecida - Hitograma')
    axes[1].set_xlabel('Valor do Pixel')
    axes[1].set_ylabel('Pixels')

    return res, plt

def remover_ruido(path_img, canal_rgb):
    img = cv2.imread(path_img)
    tamanho = img.shape
    kernel_size = 3
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
    
    img_copy = img.copy()
    
    if canal_rgb == 1:
        img_copy = filtro_passa_baixa_verde(img_copy)
    elif canal_rgb == 2:
        img_copy = filtro_passa_baixa_vermelho(img_copy,value=0)

    img_copy[:,:, canal_rgb] = filtro_mediana(img_copy[:,:, canal_rgb],ksize=kernel_size)
    img_copy[:,:, canal_rgb] = filtro_mediana(img_copy[:,:, canal_rgb],ksize=kernel_size)
    
    #concatenando as imagens
    res = cv2.hconcat([img, img_copy])
    
    #Montando os histogramas
    colors = ("blue", "green", "red")
    channel_ids = (0, 1, 2)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0, 256))
        axes[0].plot(bin_edges[0:-1], histogram, color=c)
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Valor do Pixel')
    axes[0].set_ylabel('Pixels')
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img_copy[:, :, channel_id], bins=256, range=(0, 256))
        axes[1].plot(bin_edges[0:-1], histogram, color=c)
    axes[1].set_title('Img Modificada - Hitograma')
    axes[1].set_xlabel('Valor do Pixel')
    axes[1].set_ylabel('Pixels')

    return res, plt

def agucar(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 700 or tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) 
    
    img_agucada = img.copy()

    for canal in range(3): #BGR
        img_agucada[:,:, canal] = filtro_nitidez(img_agucada[:,:, canal]) 
        img_agucada[:,:, canal] = equalizar_histograma(img_agucada[:,:, canal])
        
    #concatenando as imagens
    res = cv2.hconcat([img, img_agucada])
    
    #Montando os histogramas
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0, 256))
        axes[0].plot(bin_edges[0:-1], histogram, color=c)
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Valor do Pixel')
    axes[0].set_ylabel('Pixels')
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img_agucada[:, :, channel_id], bins=256, range=(0, 256))
        axes[1].plot(bin_edges[0:-1], histogram, color=c)
    axes[1].set_title('Img Aguçada - Hitograma')
    axes[1].set_xlabel('Valor do Pixel')
    axes[1].set_ylabel('Pixels')

    return res, plt

def melhorar_imagem(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    kernel_size = 3
    
    if tamanho[0] >= 600 and tamanho[1] >= 600:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
            
    img_melhorada = img.copy()

    img_melhorada[:,:, 2] = filtro_mediana(img_melhorada[:,:, 2],ksize=kernel_size)  
    
    for canal in range(3): #BGR
        img_melhorada[:,:, canal] = filtro_media(img_melhorada[:,:, canal])
        if canal==0 or canal==1:
            img_melhorada[:,:, canal] = filtro_nitidez(img_melhorada[:,:, canal]) 
        
    #concatenando as imagens
    res = cv2.hconcat([img, img_melhorada])
    
    #Montando os histogramas
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True, tight_layout=True)
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0, 256))
        axes[0].plot(bin_edges[0:-1], histogram, color=c)
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Valor do Pixel')
    axes[0].set_ylabel('Pixels')
    
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img_melhorada[:, :, channel_id], bins=256, range=(0, 256))
        axes[1].plot(bin_edges[0:-1], histogram, color=c)
    axes[1].set_title('Img Melhorada - Hitograma')
    axes[1].set_xlabel('Valor do Pixel')
    axes[1].set_ylabel('Pixels')

    return res, plt

if __name__ == '__main__':
    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    path_database_imgs = os.path.join(pathname, 'imgs', '')
    path_database_result = os.path.join(pathname, 'result', '')
    
    #------------- Clarear -----------------#
    res_1, plt_1 = clarear(path_database_imgs+"Clarear_1.png")
    # show_imgs(res_1, plt_1)
    save_imgs(path_database_result, "Clarear_1.png", res_1, plt_1)
    
    res_2, plt_2 = clarear(path_database_imgs+"Clarear_2.png")
    # show_imgs(res_2, plt_2)
    save_imgs(path_database_result, "Clarear_2.png", res_2, plt_2)

    # #------------- Escurecer -----------------#
    res_3, plt_3 = escurecer(path_database_imgs+"Escurecer_1.png")
    # show_imgs(res_3, plt_3)
    save_imgs(path_database_result, "Escurecer_1.png", res_3, plt_3)
    
    res_4, plt_4 = escurecer(path_database_imgs+"Escurecer_2.png")
    # show_imgs(res_4, plt_4)
    save_imgs(path_database_result, "Escurecer_2.png", res_4, plt_4)
    
    # #------------- Remover Ruido -----------------#
    res_5, plt_5 = remover_ruido(path_database_imgs+"Ruido_1.png", canal_rgb=1) #BGR
    # show_imgs(res_5, plt_5)
    save_imgs(path_database_result, "Ruido_1.png", res_5, plt_5)
    
    res_6, plt_6 = remover_ruido(path_database_imgs+"Ruido_2.png", canal_rgb=2) #BGR
    # show_imgs(res_5, plt_5)
    save_imgs(path_database_result, "Ruido_2.png", res_6, plt_6)
    
    # #------------- Agucar -----------------#
    res_8, plt_8 = agucar(path_database_imgs+"Agucar.png")
    # show_imgs(res_8, plt_8)
    save_imgs(path_database_result, "Agucar_1.png", res_8, plt_8)
        
    # #------------- Melhorar Imagem -----------------#
    res_10, plt_10 = melhorar_imagem(path_database_imgs+"Guardachuvas.png")
    # show_imgs(res_10, plt_10)
    save_imgs(path_database_result, "Image_1.jpg", res_10, plt_10)
    
    print("Processo concluido")
