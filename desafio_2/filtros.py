import cv2
import numpy as np
from matplotlib import pyplot as plt

# a convolucao é simplesmente uma multiplicação de elementos do kernel por alguma parte 
# da imagem de origem para produzir um novo ponto de dados único representando um pixel, 
# fazendo isso em todas as partes possíveis da imagem para criar uma nova imagem.
def convolucao(img, filtro_kernel):
    # peguando as dimensões espaciais da imagem, junto com as dimensões espaciais do kernel
	(img_altura, img_largura) = img.shape[:2]
	(kernel_altura, kernel_largura) = filtro_kernel.shape[:2]
	
    # aloca memória para a imagem de saída, tomando cuidado para
	# preencher as bordas da imagem de entrada para que o tamanho não seja reduzido
	pad = (kernel_largura - 1) // 2
	img = cv2.copyMakeBorder(img, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
	img_saida = np.zeros((img_altura, img_largura), dtype="float32")
    
    # fazendo um loop sobre a imagem de entrada, "deslizando" o kernel
	# cada coordenada (x, y) da esquerda para a direita e de cima para o final
	for y in np.arange(pad, img_altura + pad):
		for x in np.arange(pad, img_largura + pad):
			# extrai o ROI da imagem extraindo a região -centro- das coordenadas atuais (x, y)
			roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# realiza a convolução pegando a multiplicação de elemento a elemento entre o ROI e o kernel, e então somando a matriz
			k = (roi * filtro_kernel).sum()
			# armazena o valor convolvido na imagem de saída 
			img_saida[y - pad, x - pad] = k
    
	img_saida = (img_saida * 255).astype("uint8")

	return img_saida

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

def filtro_sobel(img):
    filtro_kernel = np.array([[1, 0, -1], 
                              [2, 0, -2], 
                              [1, 0, -1]])

    #Calculo do gradiente (calculo das derivadas parciais nas direcoes horizontal e vertical)
    #Ressalta altas frequencias (descontinuidades)
    img_conv_x = convolucao(img, filtro_kernel)
    img_conv_y = convolucao(img, np.flip(filtro_kernel.T, axis=0)) # a mascara horizontal eh a transposta da mascara vertical
    
    #Vamos combinar as arestas verticais e horizontais (derivadas)
    #Calcular magnitude do gradiente ⇒ Calcular o módulo do vetor gradiente (é um vetor com componentes nas direções x e y)
    magnitude_grad = np.sqrt(np.square(img_conv_x) + np.square(img_conv_y))
    # redimensiona a imagem de saída para ficar no intervalo [0, 255]
    magnitude_grad *= 255.0 / magnitude_grad.max()

    magnitude_grad = magnitude_grad.astype(np.uint8)
    return magnitude_grad

def filtro_laplaciano(img):
    filtro_kernel = np.array([[1,1,1],
                            [1,-8,1],
                            [1,1,1]])
    img_conv = convolucao(img , filtro_kernel)
    img_conv = img_conv.astype(np.uint8)
    return img_conv
    
def filtro_homomorfico(img, a = 0.75, b = 1.25, filter_params=[30,2]):
    # Levando a imagem para o domínio de log e depois para o domínio de frequência 
    img_log = np.log1p(np.array(img, dtype="float"))
    img_fft = np.fft.fft2(img_log)

    #Filtro Gaussiano
    img_shape = img_fft.shape
    P = img_shape[0]/2
    Q = img_shape[1]/2
    filtro_H = np.zeros(img_shape)
    U, V = np.meshgrid(range(img_shape[0]), range(img_shape[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    filtro_H = np.exp((-Duv/(2*(filter_params[0])**2)))
    filtro_H = (1 - filtro_H)
    
    #Aplicando o filtro no domínio da frequência e levando a imagem de volta ao domínio espacial
    filtro_H = np.fft.fftshift(filtro_H)
    img_filtered = (a + b*filtro_H)*img_fft    
    img_fft_filt = img_filtered
    img_filt = np.fft.ifft2(img_fft_filt)
    img = np.exp(np.real(img_filt))-1
    
    return np.uint8(img)

def filtro_media(img):    
    m, n = img.shape
    filtro = np.ones([3, 3], dtype = int)
    filtro = filtro / 9
    img_new = np.zeros([m, n])
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*filtro[0, 0]+img[i-1, j]*filtro[0, 1]+img[i-1, j + 1]*filtro[0, 2]+img[i, j-1]*filtro[1, 0]+ img[i, j]*filtro[1, 1]+img[i, j + 1]*filtro[1, 2]+img[i + 1, j-1]*filtro[2, 0]+img[i + 1, j]*filtro[2, 1]+img[i + 1, j + 1]*filtro[2, 2]
            img_new[i, j]= temp
            
    img_new = img_new.astype(np.uint8)
    return img_new

def filtro_nitidez(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_filter2d = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    img_filter2d = img_filter2d.astype(np.uint8)
    return img_filter2d

def filtro_gamma(img, gamma=0.2):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)
    return img_gamma

def _set_kernel_filtro_gaussiano(ksize):
    if ksize == 3:
        filtro_kernel = np.array([[1,2,1],
                                 [2,4,2],
                                 [1,2,1]])
        filtro_kernel = filtro_kernel * (1/16)

    elif ksize == 5:
        filtro_kernel = np.array([[1,4,7,4,1],
                                 [4,16,26,16,4],
                                 [7,26,41,26,7],
                                 [4,16,26,16,4],
                                 [1,4,7,4,1]])
        filtro_kernel = filtro_kernel * (1/273)

    elif ksize == 7:
        filtro_kernel = np.array([[0,0,1,2,1,0,0],
                                 [0,3,13,22,13,3,0],
                                 [1,13,59,97,59,13,1],
                                 [2,22,97,159,97,22,2],
                                 [1,13,59,97,59,13,1],
                                 [0,3,13,22,13,3,0],
                                 [0,0,1,2,1,0,0]])
        filtro_kernel = filtro_kernel * (1/1003)

    return filtro_kernel

def filtro_gaussiano(img,ksize):
    filtro_kernel = _set_kernel_filtro_gaussiano(ksize)
    img = convolucao(img,filtro_kernel)
    return img