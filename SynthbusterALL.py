import numpy as np
from PIL import Image
from scipy.ndimage import grey_dilation
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter



def make_structuring_element(shape: str, size: int) -> np.ndarray:
    """
    Génère un élément structurant 2D.

    Parameters
    ----------
    shape : 'square' | 'cross' | 'disk'
    size  : taille (côté pour carré, diamètre pour croix/disque)
             doit être impair pour une symétrie centrée

    Returns
    -------
    np.ndarray booléen (True = appartient à l'élément)
    """
    size = size if size % 2 == 1 else size + 1   # force impair
    half = size // 2

    if shape == "square":
        return np.ones((size, size), dtype=bool)

    elif shape == "cross":
        se = np.zeros((size, size), dtype=bool)
        se[half, :] = True
        se[:, half] = True
        return se

    elif shape == "disk":
        y, x = np.ogrid[-half:half + 1, -half:half + 1]
        return (x ** 2 + y ** 2) <= half ** 2

    else:
        raise ValueError(f"Forme inconnue : '{shape}'. Choisir parmi 'square', 'cross', 'disk'.")


def morpho_dilate_spectrum(
    mag_shift: np.ndarray,
    se_shape: str = "disk",
    se_size: int = 5,
    apply_before_log: bool = True,
) -> dict:
    """
    Applique une dilatation morphologique de niveaux de gris sur un spectre FFT.

    Parameters
    ----------
    mag_shift       : spectre FFT centré (amplitudes, non-logarithmiques)
    se_shape        : forme de l'élément structurant ('square', 'cross', 'disk')
    se_size         : taille de l'élément structurant (pixels)
    apply_before_log: si True, dilate les amplitudes brutes puis applique log1p ;
                      si False, applique log1p puis dilate.

    Returns
    -------
    dict avec clés :
      'original'  – spectre clipé + log, sans dilatation
      'dilated'   – spectre clipé + log, avec dilatation
      'se'        – élément structurant utilisé
      'se_shape'  – nom de la forme
      'se_size'   – taille effective (peut différer si forcée impaire)
    """
    se = make_structuring_element(se_shape, se_size)

    vmax = np.percentile(mag_shift, 99.5)
    mag_clipped = np.clip(mag_shift, 0, vmax)

    if apply_before_log:
        # Dilate d'abord, puis log
        mag_dilated_raw = grey_dilation(mag_clipped, footprint=se)
        original = np.log1p(mag_clipped)
        dilated  = np.log1p(mag_dilated_raw)
    else:
        # Log d'abord, puis dilate
        original = np.log1p(mag_clipped)
        dilated  = grey_dilation(original, footprint=se)

    return {
        "original": original,
        "dilated":  dilated,
        "se":       se,
        "se_shape": se_shape,
        "se_size":  se.shape[0],   # taille effective (peut être se_size+1 si pair)
    }

def cross_difference(img_array: np.ndarray) -> np.ndarray:
    return np.abs(
        img_array[:-1, :-1] + img_array[1:, 1:]
        - img_array[:-1, 1:] - img_array[1:, :-1]
    )


def compute_avg_fft_raw(image_paths: list) -> np.ndarray:
    """Retourne le spectre FFT moyen centré (amplitudes brutes, non-log)."""
    sizes = [Image.open(p).size for p in image_paths]
    most_common_size = Counter(sizes).most_common(1)[0][0]
    n_kept = sizes.count(most_common_size)
    print(f"  → Taille dominante : {most_common_size} ({n_kept}/{len(sizes)} images conservées)")

    accum, count = None, 0
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

    avg_cd_gray = (accum / count).mean(axis=2)
    fft         = np.fft.fft2(avg_cd_gray)
    mag         = np.abs(fft) / avg_cd_gray.size
    return np.fft.fftshift(mag)          # spectre centré, brut


def plot_dilation_comparison(
    mag_shift: np.ndarray,
    label: str = "",
    se_shape: str = "disk",
    se_size:  int = 5,
    apply_before_log: bool = True,
    save_path: str | None = None,
):
    """
    Affiche côte à côte :
      - le spectre original (log)
      - l'élément structurant
      - le spectre dilaté (log)
      - la différence (dilaté − original)
    """
    result = morpho_dilate_spectrum(mag_shift, se_shape, se_size, apply_before_log)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    cmap = "inferno"
    vmin_orig = result["original"].min()
    vmax_orig = result["original"].max()

    axes[0].imshow(result["original"], cmap=cmap, vmin=vmin_orig, vmax=vmax_orig)
    axes[0].set_title("Spectre original (log)")

    se_display = result["se"].astype(float)
    axes[1].imshow(se_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Élément structurant\n{result['se_shape']} {result['se_size']}×{result['se_size']}"
    )

    axes[2].imshow(result["dilated"], cmap=cmap, vmin=vmin_orig, vmax=vmax_orig)
    axes[2].set_title(f"Spectre dilaté (log)\n{result['se_shape']} — taille {result['se_size']}")

    diff = result["dilated"] - result["original"]
    im   = axes[3].imshow(diff, cmap="hot")
    axes[3].set_title("Différence (dilaté − original)")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    suptitle = f"Dilatation morphologique du spectre FFT — {label}"
    if apply_before_log:
        suptitle += "\n(dilatation sur amplitudes brutes, puis log)"
    else:
        suptitle += "\n(log d'abord, puis dilatation)"
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Sauvegardé : {save_path}")
    plt.show()

models = {
    "stable-diffusion-1-3": "Stable Diffusion 1.3",
    "stable-diffusion-1-4": "Stable Diffusion 1.4",
    "stable-diffusion-2":   "Stable Diffusion 2",
    "stable-diffusion-xl":  "Stable Diffusion XL",
    "glide":                "Glide",
    "midjourney-v5":        "Midjourney v5",
    "dalle2":               "DALL·E 2",
    "dalle3":               "DALL·E 3",
    "firefly":              "Adobe Firefly",
}

BASE = Path("./synthbuster/synthbuster")

SE_SHAPE         = "disk"    # 'square' | 'cross' | 'disk'
SE_SIZE          = 7        
APPLY_BEFORE_LOG = True      
                            


n_models = len(models)
n_cols   = 5
n_rows   = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()

for i, (folder, label) in enumerate(models.items()):
    ax    = axes[i]
    paths = list((BASE / folder).glob("*.png"))
    print(f"{label} : {len(paths)} images trouvées")

    try:
        mag_shift = compute_avg_fft_raw(paths)
        result    = morpho_dilate_spectrum(
            mag_shift,
            se_shape=SE_SHAPE,
            se_size=SE_SIZE,
            apply_before_log=APPLY_BEFORE_LOG,
        )
        ax.imshow(result["dilated"], cmap="inferno")

    except Exception as e:
        ax.text(0.5, 0.5, f"Erreur:\n{e}", transform=ax.transAxes,
                ha="center", va="center", color="red", fontsize=8)

    ax.set_title(label, fontsize=10)
    ax.axis("off")

for j in range(n_models, len(axes)):
    axes[j].axis("off")

plt.suptitle(
    f"FFT cross-difference — dilatation morphologique ({SE_SHAPE} {SE_SIZE}px) + log",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("synthbuster_all_models.png", dpi=150, bbox_inches="tight")
plt.show()

