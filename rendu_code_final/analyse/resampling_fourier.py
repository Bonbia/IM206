import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
import argparse, os


def rgb2luminance(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def rank_transform(img, sz=3):
    lm = uniform_filter(img.astype(np.float64), size=sz, mode='reflect')
    return (img - lm).astype(np.float32)


def retailler(img, taille, filtre):
    filtres = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
               "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}
    H, W = img.shape
    pil   = Image.fromarray(img.astype(np.uint8))
    petit = pil.resize((taille, taille), resample=filtres[filtre])
    grand = petit.resize((W, H), resample=filtres[filtre])
    return np.array(grand, dtype=np.float32)


def spectre_2d(img):
    x = rank_transform(img)
    F = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
    return np.log1p(np.abs(F)).astype(np.float32)


def profil_1d(img, direction="horizontal"):
    x    = rank_transform(img)
    axe  = 1 if direction == "horizontal" else 0
    F    = np.fft.fft(x.astype(np.float64), axis=axe)
    prof = np.abs(F).mean(axis=1 - axe)
    return np.fft.fftshift(prof).astype(np.float32)


def pics_theoriques(taille_orig, taille_cible, max_k=4):
    delta = abs(taille_orig - taille_cible)
    pics  = []
    for k in range(1, max_k + 1):
        if k * delta <= taille_orig // 2:
            pics.append(k * delta)
    return pics


def style_ax(ax):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='gray', labelsize=7)
    ax.set_xlabel("fréquence", color='gray', fontsize=8)
    for sp in ax.spines.values():
        sp.set_color('#333')


FILTRES  = ["nearest", "bilinear", "bicubic", "lanczos"]
COULEURS = {"nearest": "#ff6b6b", "bilinear": "#ffd93d",
            "bicubic": "#6bcb77", "lanczos": "#4d96ff"}


def figure_analyse(img_orig, taille_cible, chemin_sortie, direction="horizontal"):
    H, W        = img_orig.shape
    reimages    = {f: retailler(img_orig, taille_cible, f) for f in FILTRES}
    spec_orig   = spectre_2d(img_orig)
    prof_orig   = profil_1d(img_orig, direction)
    pics_th     = pics_theoriques(W if direction == "horizontal" else H, taille_cible)
    xs          = np.linspace(-(W // 2), W // 2, len(prof_orig))

    fig, axes = plt.subplots(4, 5, figsize=(24, 18))
    fig.patch.set_facecolor('#0f0f0f')

    # ligne 0 : images
    axes[0, 0].imshow(img_orig, cmap='gray', aspect='auto')
    axes[0, 0].axis('off')
    axes[0, 0].set_title("Original", color='white', fontsize=10)
    for j, f in enumerate(FILTRES):
        axes[0, j+1].imshow(reimages[f], cmap='gray', aspect='auto')
        axes[0, j+1].axis('off')
        axes[0, j+1].set_title(f"{f}\n→{taille_cible}→{W}", color=COULEURS[f], fontsize=10, fontweight='bold')

    # ligne 1 : spectres 2D
    vmax = np.percentile(spec_orig, 99.5)
    axes[1, 0].imshow(spec_orig, cmap='inferno', vmin=0, vmax=vmax, aspect='auto')
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Spectre 2D\nOriginal", color='white', fontsize=9)
    for j, f in enumerate(FILTRES):
        axes[1, j+1].imshow(spectre_2d(reimages[f]), cmap='inferno', vmin=0, vmax=vmax, aspect='auto')
        axes[1, j+1].axis('off')
        axes[1, j+1].set_title(f"Spectre 2D\n{f}", color=COULEURS[f], fontsize=9, fontweight='bold')

    # ligne 2 : profils 1D
    axes[2, 0].plot(xs, prof_orig / prof_orig.max(), color='white', lw=1, alpha=0.6, label='original')
    for f in FILTRES:
        p = profil_1d(reimages[f], direction)
        axes[2, 0].plot(xs, p / p.max(), color=COULEURS[f], lw=1, alpha=0.85, label=f)
    for pk in pics_th:
        axes[2, 0].axvline( pk, color='cyan', lw=0.7, ls='--', alpha=0.6)
        axes[2, 0].axvline(-pk, color='cyan', lw=0.7, ls='--', alpha=0.6)
    axes[2, 0].set_title(f"Profils 1D\n(tous filtres)", color='white', fontsize=9)
    style_ax(axes[2, 0])
    axes[2, 0].legend(fontsize=7, facecolor='#222', labelcolor='white', framealpha=0.7)

    for j, f in enumerate(FILTRES):
        p    = profil_1d(reimages[f], direction)
        norm = max(prof_orig.max(), p.max())
        ax   = axes[2, j+1]
        ax.fill_between(xs, prof_orig / norm, alpha=0.2, color='white')
        ax.plot(xs, prof_orig / norm, color='white', lw=0.8, alpha=0.6)
        ax.fill_between(xs, p / norm, alpha=0.2, color=COULEURS[f])
        ax.plot(xs, p / norm, color=COULEURS[f], lw=1.3)
        for k, pk in enumerate(pics_th):
            lbl = f'k={k+1}(±{pk})' if k < 2 else ''
            ax.axvline( pk, color='cyan', lw=0.8, ls='--', alpha=0.7, label=lbl)
            ax.axvline(-pk, color='cyan', lw=0.8, ls='--', alpha=0.7)
        ax.set_title(f"Profil {direction}\n{f}", color=COULEURS[f], fontsize=9, fontweight='bold')
        style_ax(ax)
        ax.legend(fontsize=6, facecolor='#222', labelcolor='white', framealpha=0.7)

    # ligne 3 : diff spectres
    axes[3, 0].axis('off')
    axes[3, 0].set_facecolor('#0f0f0f')
    axes[3, 0].text(0.5, 0.5, "Diff spectre\n(résamp − orig)\nRouge=+  Bleu=−",
                    ha='center', va='center', color='white', fontsize=9, transform=axes[3, 0].transAxes)

    diffs = [spectre_2d(reimages[f]) - spec_orig for f in FILTRES]
    vd    = np.percentile(np.abs(np.stack(diffs)), 98)
    for j, f in enumerate(FILTRES):
        axes[3, j+1].imshow(diffs[j], cmap='RdBu_r', vmin=-vd, vmax=vd, aspect='auto')
        axes[3, j+1].axis('off')
        axes[3, j+1].set_title(f"Diff\n{f}", color=COULEURS[f], fontsize=9, fontweight='bold')

    fig.suptitle(
        f"Analyse Fourier – {W}→{taille_cible}→{W}  |  pics théoriques ±{pics_th}  |  Rank Transform (sz=3)",
        color='white', fontsize=11, fontweight='bold', y=0.97
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(chemin_sortie, dpi=130, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"Sauvegardé : {chemin_sortie}")
    plt.close()


def figure_4composantes(img_orig, taille_cible, chemin_sortie):
    noms_comp = ["Magnitude (log)", "Cos(angle)", "Sin(angle)", "Angle"]
    cmaps     = ["inferno", "coolwarm", "coolwarm", "hsv"]
    reimages  = {f: retailler(img_orig, taille_cible, f) for f in FILTRES}

    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle("Composantes spectrales pour CNN  |  Magnitude · Cos · Sin · Angle  (Rank Transform)",
                 color='white', fontsize=12, fontweight='bold')

    for ligne, (nom, cmap) in enumerate(zip(noms_comp, cmaps)):
        def afficher(ax, img, titre, couleur):
            x  = rank_transform(img)
            F  = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
            mag   = np.log1p(np.abs(F)).astype(np.float32)
            angle = np.angle(F).astype(np.float32)
            comps = [mag, np.cos(angle), np.sin(angle), angle]
            c = comps[ligne]
            ax.imshow(c, cmap=cmap, aspect='auto',
                      vmin=np.percentile(c, 1), vmax=np.percentile(c, 99))
            ax.set_title(f"{nom}\n{titre}", color=couleur, fontsize=8, fontweight='bold')
            ax.axis('off')

        afficher(axes[ligne, 0], img_orig, "Original", 'white')
        for col, f in enumerate(FILTRES):
            afficher(axes[ligne, col+1], reimages[f], f, COULEURS[f])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(chemin_sortie, dpi=120, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"Sauvegardé : {chemin_sortie}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     type=str, default="img/baboon.png")
    parser.add_argument("--target",    type=int, default=350)
    parser.add_argument("--out_dir",   type=str, default="results")
    parser.add_argument("--direction", type=str, default="horizontal", choices=["horizontal", "vertical"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = rgb2luminance(np.array(Image.open(args.image)))
    print(f"Image : {args.image}  →  {img.shape}  |  target={args.target}")

    figure_analyse(img, args.target,
                   chemin_sortie=os.path.join(args.out_dir, "fourier_analysis.png"),
                   direction=args.direction)

    figure_4composantes(img, args.target,
                        chemin_sortie=os.path.join(args.out_dir, "fourier_4components.png"))

    print("Terminé.")