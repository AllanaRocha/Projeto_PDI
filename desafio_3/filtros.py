import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalizar_histograma(img):    
    m, n = img.shape 
    img_final = img.copy()
    
    #inicializando o histograma -> um objeto com 256 valores, cada um dos quais é um dos níveis de intensidade que a imagem pode assumir
    hist_array= [] 
    for i in range(0,256): 
        hist_array.append(str(i)) 
        hist_array.append(0) 
    
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range (0, len(hist_array), 2)} 

    #contando quantas vezes cada intensidade aparece na imagem
    for linha in img:
        for coluna in linha:
            hist_dct[str(int(coluna))] = hist_dct[str(int(coluna))] + 1

    #calculando a probabilidade de um valor aparecer, ou seja, vamos fazer a divisão 
    # de quantas vezes o valor apareceu pelo número total de pixels na imagem
    n_pixels = m * n
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist_dct[str(i)] / n_pixels
    
    #vamos calcular a probabilidade acumulada, onde para cada iteração o valor do histograma é 
    # somado à probabilidade acumulada das iterações anteriores.
    acc_proba = {}
    sum_proba = 0
    for i in range(0, 256):
        if i == 0:
            pass
        else: 
            sum_proba += hist_proba[str(i - 1)]
        acc_proba[str(i)] = hist_proba[str(i)] + sum_proba
    
    #cálculo dos novos valores de cinza da imagem
    novos_valores_de_cinza = {}
    for i in range(0, 256):
        novos_valores_de_cinza[str(i)] = np.ceil(acc_proba[str(i)] * 255)
    
    #Vamos aplicar os novos valores na imagem original.
    for linha in range(m):
        for coluna in range(n):
            img_final[linha][coluna] = novos_valores_de_cinza[str(int(img_final[linha][coluna]))]
    
    img_final = img_final.astype(np.uint8)
    return img_final

#Filtragem de mediana: também conhecida como filtragem não linear. É usado para eliminar o ruído do sal e da pimenta. 
# Aqui, o valor do pixel é substituído pelo valor mediano do pixel vizinho.
def filtro_mediana(img, ksize):
    temp = []
    indexador = ksize // 2
    img_final = []
    m, n = img.shape 
    img_final = np.zeros([m, n]) 
    
    for i in range(1, m-1): 
        for j in range(1, n-1): 
            for z in range(1, ksize):
                if i + z - indexador < 0 or i + z - indexador > m - 1:
                    for c in range(ksize):
                        temp.append(0)
                else:
                    if j + z - indexador < 0 or j + indexador > n - 1:
                        temp.append(0)
                    else:
                        for k in range(ksize):
                            temp.append(img[i + z - indexador][j + k - indexador])
            temp.sort()
            img_final[i][j] = temp[len(temp) // 2]
            temp = []
    
    img_final = img_final.astype(np.uint8)
    return img_final

def filtro_erosao(img, k=5):
    m,n= img.shape 
    imgErode= np.zeros((m,n), dtype=np.uint8)
    
    # elemento estruturante
    SE = np.ones((k,k), dtype=np.uint8)
    constant= (k-1)//2
    
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*SE
            imgErode[i,j]= np.min(product)
            
    return imgErode

def filtro_dilatacao(img):
    p,q= img.shape
    imgDilate= np.zeros((p,q), dtype=np.uint8)
    
    #elemento estruturante
    SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
    constant1= 1
    
    for i in range(constant1, p-constant1):
        for j in range(constant1,q-constant1):
            temp= img[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
            product= temp*SED
            imgDilate[i,j]= np.max(product)
    
    return imgDilate

def binarizacao_otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = 0
        muf = 0
        
        if pcb>0:
            mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        if pcf>0:
            muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
            
    final_img = gray.copy()
    # print("final_thresh: ", final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img
