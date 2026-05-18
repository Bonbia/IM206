'''Mise en place de la partie SCA de la Pipeline'''

import numpy as np

def DFT2D(IM):
    '''Implementation de la DFT2D sur l'image IM post traitée avec Rank ou TV '''
    spectre = np.fft.fft2(IM)
    return spectre

def IDFT2D(S):
    '''Implementation de la IDFT2D sur le spectre S'''
    image = np.fft.ifft2(S)
    return image 