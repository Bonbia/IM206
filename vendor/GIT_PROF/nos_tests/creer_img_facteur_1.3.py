import numpy as np
import imageio.v3 as imageio
from PIL import Image
import sys, os


def retailler(img, taille):
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.resize((taille, taille), Image.BICUBIC)).astype(np.uint8)

img = imageio.imread("../img/baboon.png")

facteur = 1.3
nouvelle_taille = int(img.shape[0] * facteur)
img_reechantillonnee = retailler(img, nouvelle_taille)
imageio.imwrite("baboon_1.3.png", img_reechantillonnee)