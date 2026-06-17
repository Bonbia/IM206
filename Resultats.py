"""
Comparaison des deux prétraitements :
  - Rank Transform  → importé depuis Pretraitement.py
  - Résidu TV       → importé depuis TV.py

Les deux fichiers doivent se trouver dans le même répertoire que ce script,
ou être accessibles via le PYTHONPATH.
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.color as color


from Pretraitement import ranktransform   
from TV import tv_chambolle               

IMAGE_PATH = './IMG_InPut/babbon.png'   

img_color = io.imread(IMAGE_PATH)
I = color.rgb2gray(img_color)
I = I * 255 / I.max()  

img_rank = ranktransform(I)
_, img_tv_residual = tv_chambolle(I, lambda_tv=1.0, n_iter=500)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Comparaison des prétraitements", fontsize=14, fontweight='bold')

axes[0].imshow(I, cmap='gray')
axes[0].set_title("Image originale (niveaux de gris)")
axes[0].axis('off')

axes[1].imshow(img_rank, cmap='gray')
axes[1].set_title("Rank Transform (voisinage 7×7)")
axes[1].axis('off')

axes[2].imshow(img_tv_residual, cmap='gray')
axes[2].set_title("Résidu TV Chambolle (λ=1, 500 it.)")
axes[2].axis('off')

plt.tight_layout()
plt.show()