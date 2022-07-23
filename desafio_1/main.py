import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def show_imgs(res, plt):
    cv2.imshow('image',res)
    cv2.waitKey(0)
    plt.show()

def clarear(path_img):
    img = cv2.imread(path_img,0)
    equ = cv2.equalizeHist(img)
    res = cv2.hconcat([img, equ])
    
    img_hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    equ_hist = cv2.calcHist(equ, [0], None, [256], [0, 256])
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    axes[0].plot(img_hist) 
    axes[0].set_title('Img Original - Hitograma')
    axes[0].set_xlabel('Nível de Cinza')
    axes[0].set_ylabel('Pixels')
    
    axes[1].plot(equ_hist, 'tab:red') 
    axes[1].set_title('Img Clareada - Hitograma Equalizado')
    axes[1].set_xlabel('Nível de Cinza')
    axes[1].set_ylabel('Pixels')

    return res, plt
    
    
if __name__ == '__main__':
    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    path_database = os.path.join(pathname, 'imgs', '')
    
    res_1, plt_1 = clarear(path_database+"Clarear_1.jpg")
    show_imgs(res_1, plt_1)
    
    res_2, plt_2 = clarear(path_database+"Clarear_2.jpg")
    show_imgs(res_2, plt_2)