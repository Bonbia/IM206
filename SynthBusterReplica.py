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

paths = list(Path("./synthbuster/synthbuster/stable-diffusion-1-3").glob("*.png"))
mag_raw = compute_avg_fft_raw(paths)

# Percentiles à tester : de 90% à 100% avec pas variable
percentiles = [90, 92, 94, 95, 96, 97, 98, 99, 99.5, 99.9]

n = len(percentiles)
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for i, p in enumerate(percentiles):
    vmax = np.percentile(mag_raw, p)
    # On clippe les valeurs au-dessus du seuil, PUIS on applique le log
    mag_clipped = np.clip(mag_raw, 0, vmax)
    mag_log = np.log1p(mag_clipped)

    axes[i].imshow(mag_log, cmap="inferno")
    axes[i].set_title(f"Clip {p}% → log\nvmax={vmax:.2e}", fontsize=9)
    axes[i].axis("off")

plt.suptitle("Stable Diffusion 1.3 — clip puis log", fontsize=13)
plt.tight_layout()
plt.show()