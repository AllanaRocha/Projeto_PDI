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
        # kernel_size = 5
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
    
    img_gaussiano = img.copy()
    
    # if not canal_rgb:
    #     for i in range(3):
    #         #passa o filtro de mediana nos 3 canais (RGB)
    #         img_mediana[:,:, i] = filtro_mediana(img_mediana[:,:, i], ksize=kernel_size) 
    # else:
    # img_mediana[:,:, canal_rgb] = filtro_mediana(img_mediana[:,:, canal_rgb], ksize=kernel_size) 
    # cv2.imshow('image',img_mediana[:,:, canal_rgb])
    # cv2.waitKey(0)
    # plt.show()
    img_gaussiano[:,:, canal_rgb] = filtro_gaussiano(img_gaussiano[:,:, canal_rgb], ksize=kernel_size) 
    cv2.imshow('image',img_gaussiano[:,:, canal_rgb])
    cv2.waitKey(0)
    plt.show()
    
    #concatenando as imagens
    res = cv2.hconcat([img, img_gaussiano])
    
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
        histogram, bin_edges = np.histogram(img_mediana[:, :, channel_id], bins=256, range=(0, 256))
        axes[1].plot(bin_edges[0:-1], histogram, color=c)
    axes[1].set_title('Img Modificada - Hitograma')
    axes[1].set_xlabel('Valor do Pixel')
    axes[1].set_ylabel('Pixels')

    return res, plt

def remover_revoada_1(path_img):
    img = cv2.imread(path_img,0)
    tamanho = img.shape
    
    if tamanho[0] >= 1000 or tamanho[1] >= 1000:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) 
    
    img_media = filtro_mediana(img, 3)

    for c in range(2):
        img_media = filtro_mediana(img_media, 3)
        
    res = cv2.hconcat([img, img_media])
    
    img_hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    img_detalhes_melhorados_hist = cv2.calcHist(img_media, [0], None, [256], [0, 256])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes[0].plot(img_hist) 
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Nível de Cinza')
    axes[0].set_ylabel('Pixels')
    
    axes[1].plot(img_detalhes_melhorados_hist, 'tab:red') 
    axes[1].set_title('Img Modificada - Hitograma')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Pixels')

    return res, plt

def remover_revoada_2(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 1000 or tamanho[1] >= 1000:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) 
    
    img_media = filtro_mediana(img, 3)
    img_lapla = filtro_laplaciano(img_media)     
    img_op_1 = cv2.addWeighted(img_lapla, 1.5, img_media, -0.9, 0)    
    img_op_2 = cv2.addWeighted(img_op_1, 0.9, img_media, 1, 0)
    img_op_3 = cv2.addWeighted(img_media, 1.5, img_op_2, -0.8, 0)
    img_op_4 = cv2.addWeighted(img_op_3, 1.5, img_media, 0.8, 0)
    img_op_5 = cv2.addWeighted(img_media, 1.5, img_op_4, -0.9, 0)
    img_op_6 = cv2.addWeighted(img_op_5, 1.5, img_media, 0.9, 0)
    img_op_7 = cv2.addWeighted(img_op_6, 1.5, img_media, 0.9, 0)
    img_op_8 = cv2.addWeighted(img_media, 1.5, img_op_7, -0.9, 0)
    img_sem_revoada = filtro_mediana(img_op_8, 9) 

    res = cv2.hconcat([img, img_sem_revoada])
    
    img_hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    img_detalhes_melhorados_hist = cv2.calcHist(img_sem_revoada, [0], None, [256], [0, 256])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes[0].plot(img_hist) 
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Nível de Cinza')
    axes[0].set_ylabel('Pixels')
    
    axes[1].plot(img_detalhes_melhorados_hist, 'tab:red') 
    axes[1].set_title('Img Modificada - Hitograma')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Pixels')

    return res, plt

def agucar_1(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 1000 or tamanho[1] >= 1000:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) 
    
    img_sobel = filtro_sobel(img) 
    img_sobel_media = filtro_media(img_sobel)
    img_sobel_media_sub = cv2.addWeighted(img, 1.5, img_sobel_media, -0.7, 0)
    
    res = cv2.hconcat([img, img_sobel_media_sub])
    
    img_hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    img_sobel_media_sub_hist = cv2.calcHist(img_sobel_media_sub, [0], None, [256], [0, 256])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes[0].plot(img_hist) 
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Nível de Cinza')
    axes[0].set_ylabel('Pixels')
    
    axes[1].plot(img_sobel_media_sub_hist, 'tab:red') 
    axes[1].set_title('Img Agucada - Hitograma')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Pixels')

    return res, plt

def agucar_2(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 1000 or tamanho[1] >= 1000:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) 
    
    img_nit = filtro_nitidez(img)
    img_detalhes = cv2.addWeighted(img_nit, 1.5, img, -0.8, 0)
    img_detalhes_melhorados = cv2.addWeighted(img_detalhes, 1.5, img_nit, 0.2, 0)
    
    res = cv2.hconcat([img, img_detalhes_melhorados])
    
    img_hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    img_detalhes_melhorados_hist = cv2.calcHist(img_detalhes_melhorados, [0], None, [256], [0, 256])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes[0].plot(img_hist) 
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Nível de Cinza')
    axes[0].set_ylabel('Pixels')
    
    axes[1].plot(img_detalhes_melhorados_hist, 'tab:red') 
    axes[1].set_title('Img Agucada - Hitograma')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Pixels')

    return res, plt

def melhorar_imagem(path_img):
    img = cv2.imread(path_img)
    tamanho = img.shape
    
    if tamanho[0] >= 700 and tamanho[1] >= 700:
        img = cv2.resize(img, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC) #reducao de 50% em relacao a img original
    
    img_nit = filtro_nitidez(img)
    img_homo = filtro_homomorfico(img_nit) 
    img_homo_equ = equalizar_histograma(img_homo)
            
    res = cv2.hconcat([img, img_homo_equ])
    
    img_hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    img_homo_equ_hist = cv2.calcHist(img_homo_equ, [0], None, [256], [0, 256])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes[0].plot(img_hist) 
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Nível de Cinza')
    axes[0].set_ylabel('Pixels')
    
    axes[1].plot(img_homo_equ_hist, 'tab:red') 
    axes[1].set_title('Img melhorada - Hitograma')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Pixels')

    return res, plt

if __name__ == '__main__':
    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    path_database_imgs = os.path.join(pathname, 'imgs', '')
    path_database_result = os.path.join(pathname, 'result', '')
    
    #------------- Clarear -----------------#
    # res_1, plt_1 = clarear(path_database_imgs+"Clarear_1.png")
    # show_imgs(res_1, plt_1)
    # save_imgs(path_database_result, "Clarear_1.png", res_1, plt_1)
    
    # res_2, plt_2 = clarear(path_database_imgs+"Clarear_2.png")
    # show_imgs(res_2, plt_2)
    # save_imgs(path_database_result, "Clarear_2.png", res_2, plt_2)

    # #------------- Escurecer -----------------#
    # res_3, plt_3 = escurecer(path_database_imgs+"Escurecer_1.png")
    # show_imgs(res_3, plt_3)
    # save_imgs(path_database_result, "Escurecer_1.png", res_3, plt_3)
    
    # res_4, plt_4 = escurecer(path_database_imgs+"Escurecer_2.png")
    # show_imgs(res_4, plt_4)
    # save_imgs(path_database_result, "Escurecer_2.png", res_4, plt_4)
    
    # #------------- Remover Ruido -----------------#
    res_5, plt_5 = remover_ruido(path_database_imgs+"Ruido_1.png", canal_rgb=1) #passando o canal G
    show_imgs(res_5, plt_5)
    # save_imgs(path_database_result, "Ruido_1.png", res_5, plt_5)
    
    res_6, plt_6 = remover_ruido(path_database_imgs+"Ruido_2.png", canal_rgb=0) #passando o canal R
    show_imgs(res_6, plt_6)
    # save_imgs(path_database_result, "Ruido_2.png", res_6, plt_6)
    
    # #------------- Agucar -----------------#
    # res_8, plt_8 = agucar_1(path_database_imgs+"Agucar_1.png")
    # # show_imgs(res_8, plt_8)
    # save_imgs(path_database_result, "Agucar_1.png", res_8, plt_8)

    # res_9, plt_9 = agucar_2(path_database_imgs+"Agucar_2.png")
    # # show_imgs(res_9, plt_9)
    # save_imgs(path_database_result, "Agucar_2.png", res_9, plt_9)
        
    # #------------- Melhorar Imagem -----------------#
    # res_10, plt_10 = melhorar_imagem(path_database_imgs+"Image_1.jpg")
    # # show_imgs(res_10, plt_10)
    # save_imgs(path_database_result, "Image_1.jpg", res_10, plt_10)
    
    # #------------- Remover Revoada -----------------#
    # res_11, plt_11 = remover_revoada_1(path_database_imgs+"Revoada_1.jpg")
    # # show_imgs(res_11, plt_11)
    # save_imgs(path_database_result, "Revoada_1.jpg", res_11, plt_11)
    
    # res_12, plt_12 = remover_revoada_2(path_database_imgs+"Revoada_2.jpg")
    # # show_imgs(res_12, plt_12)
    # save_imgs(path_database_result, "Revoada_2.jpg", res_12, plt_12)
    
    print("Processo concluido")
