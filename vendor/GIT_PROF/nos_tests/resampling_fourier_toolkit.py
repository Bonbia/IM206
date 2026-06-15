

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.ndimage import uniform_filter
import argparse
import os


# ─── Fonctions du dépôt (src/) ────────────────────────────────────────────────

def rgb2luminance(img: np.ndarray) -> np.ndarray:
    """Identique à src/misc.py : RGB float → luminance float32"""
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def rank_transform(img: np.ndarray, sz: int = 3) -> np.ndarray:
    """
    Approximation du rank transform : soustraction de la moyenne locale.
    Identique à l'effet de src/filters/rank_transform.py (résidu haute fréquence).
    sz : taille du voisinage (3 par défaut, comme dans detect_one_image.py)
    """
    lm = uniform_filter(img.astype(np.float64), size=sz, mode='reflect')
    return (img - lm).astype(np.float32)


def resample_image(img_arr: np.ndarray, target_size: int, filt_name: str) -> np.ndarray:
    """
    Rééchantillonne img_arr : original_size → target_size → original_size.
    Cela introduit les pics de corrélation dans le spectre de Fourier.

    Parameters
    ----------
    img_arr   : image grayscale float32 (H, W)
    target_size : taille intermédiaire (ex: 350 pour 512→350→512)
    filt_name : 'nearest' | 'bilinear' | 'bicubic' | 'lanczos'

    Returns
    -------
    np.ndarray : image rééchantillonnée (H, W) float32
    """
    FILTERS = {
        "nearest":  Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic":  Image.BICUBIC,
        "lanczos":  Image.LANCZOS,
    }
    H, W = img_arr.shape
    pil   = Image.fromarray(img_arr.astype(np.uint8))
    small = pil.resize((target_size, target_size), resample=FILTERS[filt_name])
    back  = small.resize((W, H), resample=FILTERS[filt_name])
    return np.array(back, dtype=np.float32)


def get_fourier_spectrum(img: np.ndarray, preproc: bool = True, rt_sz: int = 3):
    """
    Calcule le spectre de Fourier 2D centré (fftshift).

    Returns (magnitude_log, cos_component, sin_component, angle)
    → Les 4 composantes mentionnées dans les notes du RDV 4 comme utiles pour le CNN.

    Parameters
    ----------
    img    : image grayscale float32 (H, W)
    preproc: si True, applique rank_transform avant la FFT (recommandé)
    rt_sz  : taille du rank transform

    Returns
    -------
    tuple : (mag_log, cos_c, sin_c, angle) — tous de shape (H, W)
    """
    x = rank_transform(img, sz=rt_sz) if preproc else img.copy()
    F     = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
    mag   = np.log1p(np.abs(F)).astype(np.float32)
    angle = np.angle(F).astype(np.float32)
    cos_c = np.cos(angle)
    sin_c = np.sin(angle)
    return mag, cos_c, sin_c, angle


def get_1d_profile(img: np.ndarray, direction: str = "horizontal",
                   preproc: bool = True, rt_sz: int = 3) -> np.ndarray:
    """
    Profil 1D : moyenne de la magnitude du spectre sur l'axe perpendiculaire.
    Utile pour visualiser les pics à une fréquence précise.

    direction : 'horizontal' (FFT sur les colonnes) | 'vertical' (FFT sur les lignes)
    Returns   : profil centré (fftshift) de longueur W ou H
    """
    x    = rank_transform(img, sz=rt_sz) if preproc else img.copy()
    axis = 1 if direction == "horizontal" else 0
    F    = np.fft.fft(x.astype(np.float64), axis=axis)
    prof = np.abs(F).mean(axis=1 - axis)
    return np.fft.fftshift(prof).astype(np.float32)


def expected_peaks(orig_size: int, target_size: int, max_k: int = 4):
    """
    Positions théoriques des pics de rééchantillonnage dans le spectre 1D centré.
    Pour un rééchantillonnage orig_size → target_size → orig_size,
    les pics apparaissent aux fréquences : ±k * |orig_size - target_size|
    """
    delta = abs(orig_size - target_size)
    return [k * delta for k in range(1, max_k + 1) if k * delta <= orig_size // 2]


# ─── Visualisation ────────────────────────────────────────────────────────────

FILTER_COLORS = {
    "nearest":  "#ff6b6b",
    "bilinear": "#ffd93d",
    "bicubic":  "#6bcb77",
    "lanczos":  "#4d96ff",
}
FILTER_NAMES = ["nearest", "bilinear", "bicubic", "lanczos"]


def plot_analysis(img_orig: np.ndarray, target_size: int,
                  save_path: str = "fourier_analysis.png",
                  direction: str = "horizontal"):
    """
    Figure complète :
      Ligne 0 : images (original + 4 filtres)
      Ligne 1 : spectres 2D magnitude log
      Ligne 2 : profils 1D avec pics théoriques annotés
      Ligne 3 : différence spectre (résamp − original)
    """
    H, W = img_orig.shape
    resampled = {f: resample_image(img_orig, target_size, f) for f in FILTER_NAMES}

    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor('#0f0f0f')
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.42, wspace=0.25,
                           left=0.03, right=0.99, top=0.93, bottom=0.03)

    spec_orig  = get_fourier_spectrum(img_orig)[0]
    prof_orig  = get_1d_profile(img_orig, direction)
    peaks_th   = expected_peaks(W if direction == "horizontal" else H, target_size)
    xs         = np.linspace(-(W//2), W//2, len(prof_orig))

    # --- Ligne 0 : images ----
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_orig, cmap='gray', aspect='auto'); ax.axis('off')
    ax.set_title("Original", color='white', fontsize=10, pad=3)

    for j, fname in enumerate(FILTER_NAMES):
        ax = fig.add_subplot(gs[0, j+1])
        ax.imshow(resampled[fname], cmap='gray', aspect='auto'); ax.axis('off')
        ax.set_title(f"{fname}\n→{target_size}→{W}", color=FILTER_COLORS[fname],
                     fontsize=10, fontweight='bold', pad=3)

    # --- Ligne 1 : spectres 2D ---
    vmax_g = np.percentile(spec_orig, 99.5)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(spec_orig, cmap='inferno', vmin=0, vmax=vmax_g, aspect='auto'); ax.axis('off')
    ax.set_title("Spectre 2D\nOriginal", color='white', fontsize=9, pad=3)

    for j, fname in enumerate(FILTER_NAMES):
        spec = get_fourier_spectrum(resampled[fname])[0]
        ax = fig.add_subplot(gs[1, j+1])
        ax.imshow(spec, cmap='inferno', vmin=0, vmax=vmax_g, aspect='auto'); ax.axis('off')
        ax.set_title(f"Spectre 2D\n{fname}", color=FILTER_COLORS[fname], fontsize=9, fontweight='bold', pad=3)

    # --- Ligne 2 : profils 1D + pics théoriques ---
    ax_all = fig.add_subplot(gs[2, 0])
    ax_all.plot(xs, prof_orig/prof_orig.max(), color='white', lw=1, alpha=0.6, label='original')
    for fname in FILTER_NAMES:
        p = get_1d_profile(resampled[fname], direction)
        ax_all.plot(xs, p/p.max(), color=FILTER_COLORS[fname], lw=1, alpha=0.85, label=fname)
    for pk in peaks_th:
        ax_all.axvline( pk, color='cyan', lw=0.7, ls='--', alpha=0.6)
        ax_all.axvline(-pk, color='cyan', lw=0.7, ls='--', alpha=0.6)
    ax_all.set_title(f"Profils 1D {direction}\n(tous filtres)", color='white', fontsize=9)
    _style_ax(ax_all); ax_all.legend(fontsize=7, facecolor='#222', labelcolor='white', framealpha=0.7)

    for j, fname in enumerate(FILTER_NAMES):
        p = get_1d_profile(resampled[fname], direction)
        norm = max(prof_orig.max(), p.max())
        ax = fig.add_subplot(gs[2, j+1])
        ax.fill_between(xs, prof_orig/norm, alpha=0.2, color='white')
        ax.plot(xs, prof_orig/norm, color='white', lw=0.8, alpha=0.6)
        ax.fill_between(xs, p/norm, alpha=0.2, color=FILTER_COLORS[fname])
        ax.plot(xs, p/norm, color=FILTER_COLORS[fname], lw=1.3)
        for k, pk in enumerate(peaks_th):
            lbl = f'k={k+1}(±{pk})' if k < 2 else ''
            ax.axvline( pk, color='cyan', lw=0.8, ls='--', alpha=0.7, label=lbl)
            ax.axvline(-pk, color='cyan', lw=0.8, ls='--', alpha=0.7)
        ax.set_title(f"Profil {direction}\n{fname}", color=FILTER_COLORS[fname], fontsize=9, fontweight='bold')
        _style_ax(ax); ax.legend(fontsize=6, facecolor='#222', labelcolor='white', framealpha=0.7)

    # --- Ligne 3 : diff spectres 2D ---
    ax = fig.add_subplot(gs[3, 0]); ax.axis('off'); ax.set_facecolor('#0f0f0f')
    ax.text(0.5, 0.5, "Diff spectre\n(résamp − orig)\nRouge = +, Bleu = −",
            ha='center', va='center', color='white', fontsize=9, transform=ax.transAxes)

    diffs = [get_fourier_spectrum(resampled[f])[0] - spec_orig for f in FILTER_NAMES]
    vd = np.percentile(np.abs(np.stack(diffs)), 98)
    for j, fname in enumerate(FILTER_NAMES):
        ax = fig.add_subplot(gs[3, j+1])
        ax.imshow(diffs[j], cmap='RdBu_r', vmin=-vd, vmax=vd, aspect='auto'); ax.axis('off')
        ax.set_title(f"Diff\n{fname}", color=FILTER_COLORS[fname], fontsize=9, fontweight='bold', pad=3)

    fig.suptitle(
        f"Analyse Fourier – Rééchantillonnage {W}→{target_size}→{W}  |  "
        f"Pics théoriques aux fréquences ±{peaks_th}  |  Prétraitement : Rank Transform (sz=3)",
        color='white', fontsize=11, fontweight='bold', y=0.97
    )
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"Sauvegardé : {save_path}")
    plt.close()


def _style_ax(ax):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='gray', labelsize=7)
    ax.set_xlabel("fréquence", color='gray', fontsize=8)
    for sp in ax.spines.values(): sp.set_color('#333')


def plot_4components(img_orig: np.ndarray, target_size: int,
                     save_path: str = "fourier_4components.png"):
    """
    Figure des 4 composantes spectrales utiles pour le CNN :
      Magnitude (log) | Cos(angle) | Sin(angle) | Angle
    Pour chaque filtre + l'original.
    """
    COMP_NAMES = ["Magnitude (log)", "Cos(angle)", "Sin(angle)", "Angle"]
    CMAPS      = ["inferno", "coolwarm", "coolwarm", "hsv"]
    resampled  = {f: resample_image(img_orig, target_size, f) for f in FILTER_NAMES}

    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle(
        "Composantes spectrales pour CNN  |  Magnitude · Cos · Sin · Angle  (Rank Transform)",
        color='white', fontsize=12, fontweight='bold'
    )

    for row, (comp_name, cmap) in enumerate(zip(COMP_NAMES, CMAPS)):
        comps_orig = get_fourier_spectrum(img_orig)
        c = comps_orig[row]
        axes[row, 0].imshow(c, cmap=cmap, aspect='auto',
                            vmin=np.percentile(c, 1), vmax=np.percentile(c, 99))
        axes[row, 0].set_title(f"{comp_name}\nOriginal", color='white', fontsize=8)
        axes[row, 0].axis('off')

        for col, fname in enumerate(FILTER_NAMES):
            comps = get_fourier_spectrum(resampled[fname])
            c = comps[row]
            axes[row, col+1].imshow(c, cmap=cmap, aspect='auto',
                                    vmin=np.percentile(c, 1), vmax=np.percentile(c, 99))
            axes[row, col+1].set_title(f"{comp_name}\n{fname}",
                                       color=FILTER_COLORS[fname], fontsize=8, fontweight='bold')
            axes[row, col+1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"Sauvegardé : {save_path}")
    plt.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse Fourier des filtres de rééchantillonnage")
    parser.add_argument("--image",   type=str, default="img/baboon.png", help="Image source (PNG/JPEG)")
    parser.add_argument("--target",  type=int, default=350,              help="Taille intermédiaire de rééchantillonnage")
    parser.add_argument("--out_dir", type=str, default="results",        help="Dossier de sortie")
    parser.add_argument("--direction", type=str, default="horizontal",   choices=["horizontal", "vertical"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw = np.array(Image.open(args.image))
    img = rgb2luminance(raw)

    print(f"Image : {args.image}  →  {img.shape}  |  target={args.target}")

    plot_analysis(img, args.target,
                  save_path=os.path.join(args.out_dir, "fourier_analysis.png"),
                  direction=args.direction)

    plot_4components(img, args.target,
                     save_path=os.path.join(args.out_dir, "fourier_4components.png"))

    print("Terminé.")
