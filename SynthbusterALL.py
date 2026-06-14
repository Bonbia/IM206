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
from collections import Counter

def compute_avg_fft_raw(image_paths):
    # 1. Trouver la taille la plus fréquente
    sizes = []
    for p in image_paths:
        with Image.open(p) as img:
            sizes.append(img.size)  # (W, H)
    
    most_common_size = Counter(sizes).most_common(1)[0][0]
    n_total = len(sizes)
    n_kept = sizes.count(most_common_size)
    print(f"  → Taille dominante : {most_common_size} ({n_kept}/{n_total} images conservées)")

    # 2. N'accumuler que les images de cette taille
    accum = None
    count = 0
    for p in image_paths:
        with Image.open(p) as img:
            if img.size != most_common_size:
                continue
            arr = np.array(img.convert("RGB")).astype(float)
        
        cd = cross_difference(arr)
        accum = cd if accum is None else accum + cd
        count += 1

    if accum is None:
        raise ValueError("Aucune image conservée.")

    avg_cd = accum / count
    avg_cd_gray = avg_cd.mean(axis=2)
    fft = np.fft.fft2(avg_cd_gray)
    mag = np.abs(fft) / avg_cd_gray.size
    mag_shift = np.fft.fftshift(mag)
    mag_dilated = grey_dilation(mag_shift, size=5)
    return mag_dilated

# Nom du dossier → label affiché
models = {
    "stable-diffusion-1-3": "Stable Diffusion 1.3",
    "stable-diffusion-1-4": "Stable Diffusion 1.4",
    "stable-diffusion-2":   "Stable Diffusion 2",
    "stable-diffusion-xl":  "Stable Diffusion XL",
    "glide":                "Glide",
    "midjourney-v5":        "Midjourney v5",
    "dalle2":             "DALL·E 2",
    "dalle3":             "DALL·E 3",
    "firefly":        "Adobe Firefly",
}

BASE = Path("./synthbuster/synthbuster")
CLIP_PERCENTILE = 99.5

n_models = len(models)
n_cols = 5
n_rows = (n_models + n_cols - 1) // n_cols  # 2 lignes de 5

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()

for i, (folder, label) in enumerate(models.items()):
    ax = axes[i]
    paths = list((BASE / folder).glob("*.png"))
    print(f"{label} : {len(paths)} images trouvées")

    try:
        mag_raw = compute_avg_fft_raw(paths)
        vmax = np.percentile(mag_raw, CLIP_PERCENTILE)
        mag_clipped = np.clip(mag_raw, 0, vmax)
        mag_log = np.log1p(mag_clipped)
        ax.imshow(mag_log, cmap="inferno")
    except Exception as e:
        ax.text(0.5, 0.5, f"Erreur:\n{e}", transform=ax.transAxes,
                ha="center", va="center", color="red", fontsize=8)

    ax.set_title(label, fontsize=10)
    ax.axis("off")

# Masquer les axes vides si n_models < n_rows * n_cols
for j in range(n_models, len(axes)):
    axes[j].axis("off")

plt.suptitle(f"FFT cross-difference moyennée — clip {CLIP_PERCENTILE}% + log", fontsize=13)
plt.tight_layout()
plt.savefig("synthbuster_all_models.png", dpi=150, bbox_inches="tight")
plt.show()