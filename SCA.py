'''Mise en place de la partie SCA de la Pipeline'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def DFT2D(IM):
    '''Implementation de la DFT2D sur l'image IM post traitée avec Rank ou TV '''
    spectre = np.fft.fft2(IM)
    return spectre

def IDFT2D(S):
    '''Implementation de la IDFT2D sur le spectre S'''
    image = np.fft.ifft2(S)
    return image 


if __name__ == "__main__":
    
    # Test de la DFT2D et IDFT2D
    img = Image.open("babbon.png").convert("L") #Image sans pretraitement convertie en NB pour test simple
    IM = np.array(img, dtype=np.float64)
    S = DFT2D(IM)

    # IDFT et garder la partie réelle
    IM_reconstructed = np.real(IDFT2D(S))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(IM, cmap="gray")
    axes[0].set_title("Image originale (gris)")
    axes[0].axis("off")

    axes[1].imshow(np.log(np.abs(np.fft.fftshift(S)) + 1), cmap="gray")
    axes[1].set_title("Spectre (log magnitude)")
    axes[1].axis("off")

    axes[2].imshow(np.clip(IM_reconstructed, 0, 255).astype(np.uint8), cmap="gray")
    axes[2].set_title("Image reconstruite")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()