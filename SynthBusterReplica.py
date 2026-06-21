import numpy as np
from PIL import Image
from scipy.ndimage import grey_dilation
import matplotlib.pyplot as plt
from pathlib import Path

def cross_difference(img_array):
    return np.abs(
        img_array[:-1, :-1] + img_array[1:, 1:]
        - img_array[:-1, 1:] - img_array[1:, :-1]
    )

def compute_avg_fft_raw(image_paths):
    """Retourne la magnitude brute SANS log pour contrôler l'affichage."""
    accum = None
    count = 0
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB")).astype(float)
        cd = cross_difference(img)
        accum = cd if accum is None else accum + cd
        count += 1

    avg_cd = accum / count
    avg_cd_gray = avg_cd.mean(axis=2)
    fft = np.fft.fft2(avg_cd_gray)
    mag = np.abs(fft) / avg_cd_gray.size
    mag_shift = np.fft.fftshift(mag)
    mag_dilated = grey_dilation(mag_shift, size=5)
    return mag_dilated  # brut, sans log


def show_cross_difference(image_path):
    img = np.array(Image.open(image_path).convert("RGB")).astype(float)
    print(f"Image shape : {img.shape}")

    cd = cross_difference(img)                        
    cd_gray = cd.mean(axis=2)                         
    print(f"Cross-diff shape : {cd.shape}, max={cd.max():.2f}")
    CLIP_PERCENTILE = 99.5

    fft = np.fft.fft2(cd_gray)
    mag = np.abs(fft) / cd_gray.size                  
    mag_shift = np.fft.fftshift(mag)     
    vmax = np.percentile(mag_shift, CLIP_PERCENTILE)   
    mag_clipped = np.clip(mag_shift, 0, vmax)          
    mag_log = np.log1p(mag_clipped)                     
    mag_dilated = grey_dilation(mag_shift, size=5)    

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(img.astype(np.uint8))
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(cd_gray, cmap="gray")
    axes[1].set_title("Cross-difference (niveaux de gris)")
    axes[1].axis("off")

    axes[2].imshow(mag_log, cmap="inferno")
    axes[2].set_title("Spectre FFT (log) de la CD avec clipping")
    axes[2].axis("off")

    axes[3].imshow(np.log1p(mag_dilated), cmap="inferno")
    axes[3].set_title("Spectre dilaté (log)")
    axes[3].axis("off")

    plt.suptitle(Path(image_path).name, fontsize=12)
    plt.tight_layout()
    plt.savefig("cross_difference_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Sauvegardé : cross_difference_result.png")


if __name__ == "__main__":
    image_path = "./synthbuster/synthbuster/stable-diffusion-1-3/r000da54ft.png"
    show_cross_difference(image_path)
    
    