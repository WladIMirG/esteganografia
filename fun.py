# fazer upload da imagem
import sys
import os
import cv2
import numpy as np
from dados_sys import *

from typing import Any
from types import MethodType
    
def img_bin(img : Any) -> np.ndarray:
    img.r = np.array([[[bin(img.r[i,j])[2:].zfill(8)] for i in range(img.h)] for j in range(img.w)])
    img.g = np.array([[[bin(img.g[i,j])[2:].zfill(8)] for i in range(img.h)] for j in range(img.w)])
    img.b = np.array([[[bin(img.b[i,j])[2:].zfill(8)] for i in range(img.h)] for j in range(img.w)])
    return np.concatenate([img.b, img.g, img.r], axis=2)
    
def bin_img(img : Any) -> np.ndarray:
    img.b = np.array([[int(img.array[i,j,0], 2) for i in range(img.h)] for j in range(img.w)], dtype=np.uint8)
    img.g = np.array([[int(img.array[i,j,1], 2) for i in range(img.h)] for j in range(img.w)], dtype=np.uint8)
    img.r = np.array([[int(img.array[i,j,2], 2) for i in range(img.h)] for j in range(img.w)], dtype=np.uint8)
    return cv2.merge([img.b, img.g, img.r])

def arq_bin(file : str) -> list:
    return [bin(ord(file.data[i]))[2:].zfill(8) for i in range(len(file.data))]

def bin_arq(file : list) -> str:
    file.data = "".join([chr(int(file.data[i], 2)) for i in range(len(file.data))])

def conv_bgr(img):
    img.b, img.g, img.r = cv2.split(img.array)
    
def com_weight(ima, arq):
    print("Peso de archivo: {0}\nPeso de la imagen: {1}".format(arq.weight(), ima.weight()))
    return arq.weight() < (3/8)*ima.weight()

def enc_proces(binimg, binfile, ch):
    n : int = 1
    m : int = 1
    c : int = 0
    r : list = [-1, 0, 1]
    for l in binfile.data:
        l = list(l)
        #print (l)
        for i in r:
            for j in r:
                if n+i != n or m+j != m:
                    binimg.array[n+i,m+j,ch[c]] = binimg.array[n+i,m+j,ch[c]][:-1]+l.pop(0)
                    up = binimg.array[n+i,m+j,ch[c]]
        m += 3
        if m >= binimg.w - 2:
            m = 1
            n += 3
        if n >= binimg.h - 2: n = 1
        
        if c < len(ch)-1:
            c += 1
        else:
            c = 0
            
def enc_proces2(binimg, binfile, ch):
    n : int = 1
    m : int = 1
    c : int = 0
    r : list = [-1, 0, 1]
    for l in binfile.data:
        
        l = list(l)
        #print (l)
        for i in r:
            for j in r:
                if n+i != n or m+j != m:
                    binimg.array[n+i,m+j,ch[c]] = "00000111"
                    up = binimg.array[n+i,m+j,ch[c]]
        m += 3
        if m >= binimg.w - 2:
            m = 1
            n += 3
        if n >= binimg.h - 2: n = 1
        
        if c < len(ch)-1:
            c += 1
        else:
            c = 0
            
def dec_proces(img, ch):
    l : list = []
    file : list = []
    n : int = 1
    m : int = 1
    c : int = 0
    r : list = [-1, 0, 1]
    while l != "%":
        l = []
        if m >= img.w - 2:
            m = 1
            n += 3
        if n >= img.h - 2: n = 1
        for i in r:
            for j in r:
                if n+i != n or m+j != m:
                    l.append(img.array[n+i,m+j,ch[c]][-1:])
        l = chr(int("".join(l), 2))
        file.append(l)
        m += 3
        if c < len(ch)-1:
            c += 1
        else:
            c = 0
            
    return file


    
def encrypter(nimg, nfile, ch, pt = "cv2"):
    if type(nimg) == str:
        binimg = Imagem()
        binimg.imag_up(nimg)
        binfile = File()
        binfile.arq_up(nfile)
    else:
        binimg = nimg
        binfile = nfile
    
    if com_weight(binimg, binfile):
        org = binimg.array
        conv_bgr(binimg)
        
        binimg.array = img_bin(binimg)
        binfile.data = arq_bin(binfile)
        
        enc_proces(binimg, binfile, ch)
        
        binimg.array = bin_img(binimg)
        
        if pt == "cv2":
            cv2.imwrite("imagem/"+"_"+nimg, binimg.array)
            cv2.imshow("Orginal", org)
            cv2.imshow("Esteganografia", binimg.array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if pt == "plt":
            from matplotlib import pyplot as plt
            org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
            binimg.array = cv2.cvtColor(binimg.array, cv2.COLOR_BGR2RGB)
            plt.imshow(org)
            plt.title('Original')
            plt.show()
            plt.imshow(binimg.array)
            plt.title('Encript')
            plt.show()
            
            
    
    else:
        print("Pesos de archovos incompatible. 1/8 del peso del archivo de destino")
        return None

def decrypter(nimg, nfile, ch):
    img = Imagem()
    img.imag_up(nimg)
    conv_bgr(img)
    img.array = img_bin(img)
    file = dec_proces(img, ch)
    with open(nfile, "w") as f:
        f.write("".join(file))
        f.close()
    print("".join(file))

def save_img(img, nome):
    cv2.imwrite (nome, img)
    


func = {"BGR": [0,1,2],
        "B"  : [0],
        "G"  : [1],
        "R"  : [2],
        "BG"  : [0,1],
        "BR"  : [0,2],
        "DR"  : [1,2],
        "RGB" : [0,1,2],
        "encrypter" : encrypter,
        "decrypter" : decrypter}


def bin_img2(img):
    tic = time.time()
    r = []
    g = []
    b = []
    
    for i in range(img.w):
        rr = []
        gg = []
        bb = []
        for j in range(img.h):
            rr.append( bin(img.r[i,j]))
            gg.append(bin(img.g[i,j]))
            bb.append(bin(img.b[i,j]))
        r.append(rr)
        g.append(gg)
        b.append(bb)
    img.r = np.array(r)
    img.g = np.array(g)
    img.b = np.array(b)
    toc = time.time()
    tim1 = toc - tic
    print("el tiempo en que se demora haciendo esto es: {%4.2f}"%tim1)