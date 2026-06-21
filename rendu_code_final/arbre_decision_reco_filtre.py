import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
import argparse

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(REPO_ROOT, "img", "baboon.png")
OUT_DIR    = os.path.join(REPO_ROOT, "results")
TARGET     = 350

THR_NEAREST = 0.5616
THR_LANCZOS = 0.7771
THR_BICUBIC = 0.2176

COULEURS = {"nearest": "#ff6b6b", "bilinear": "#ffd93d",
            "bicubic": "#6bcb77", "lanczos": "#4d96ff"}
FILTRES = ["nearest", "bilinear", "bicubic", "lanczos"]


def rgb2luminance(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)


def rank_transform(img, sz=3):
    lm = uniform_filter(img.astype(np.float64), size=sz, mode='reflect')
    return (img - lm).astype(np.float32)


def retailler(img, taille, filtre):
    filtres_pil = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
                   "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}
    H, W  = img.shape
    pil   = Image.fromarray(img.astype(np.uint8))
    petit = pil.resize((taille, taille), resample=filtres_pil[filtre])
    grand = petit.resize((W, H), resample=filtres_pil[filtre])
    return np.array(grand, dtype=np.float32)


def extraire_features(img_test, target):
    H, W  = img_test.shape
    delta = W - target

    x   = rank_transform(img_test)
    F   = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
    mag = np.abs(F)

    profil = mag.mean(axis=0)
    freqs  = np.fft.fftshift(np.fft.fftfreq(W, d=1.0/W))

    profil_n = profil / (profil[np.abs(freqs) > 5].max() + 1e-9)

    largeur_fenetre = 12
    mask_pic = (np.abs(freqs) >= delta - largeur_fenetre) & (np.abs(freqs) <= delta + largeur_fenetre) & (freqs != 0)
    mask_mid = (np.abs(freqs) > 30) & (np.abs(freqs) < delta - largeur_fenetre)
    mask_hf  = (np.abs(freqs) > delta + largeur_fenetre) & (np.abs(freqs) < W // 2)

    e_pic = float(profil_n[mask_pic].mean()) if mask_pic.any() else 0.0
    e_mid = float(profil_n[mask_mid].mean()) if mask_mid.any() else 1.0
    e_hf  = float(profil_n[mask_hf].mean())  if mask_hf.any()  else 0.0

    return {
        "e_pic": e_pic, "e_mid": e_mid, "e_hf": e_hf,
        "hf_mid": e_hf / (e_mid + 1e-9),
        "delta": delta, "freqs": freqs, "profil": profil_n,
    }


def predire_filtre(feats):
    hf_mid = feats["hf_mid"]
    e_mid  = feats["e_mid"]
    e_hf   = feats["e_hf"]

    if hf_mid > THR_NEAREST:
        filtre = "nearest"
        raison = f"hf_mid={hf_mid:.4f} > seuil {THR_NEAREST:.4f} → beaucoup d'énergie HF, aucun filtrage"
    elif e_mid > THR_LANCZOS:
        filtre = "lanczos"
        raison = f"e_mid={e_mid:.4f} > seuil {THR_LANCZOS:.4f} → énergie concentrée en basse fréquence"
    elif e_hf > THR_BICUBIC:
        filtre = "bicubic"
        raison = f"e_hf={e_hf:.4f} > seuil {THR_BICUBIC:.4f} → léger rebond HF du noyau cubique"
    else:
        filtre = "bilinear"
        raison = f"e_hf={e_hf:.4f} ≤ seuil {THR_BICUBIC:.4f} → atténuation régulière, pas de rebond"

    return filtre, raison


def calibrer(img_ref, target):
    feats = {}
    for f in FILTRES:
        feats[f] = extraire_features(retailler(img_ref, target, f), target)

    autres_hf_mid = []
    for f in ["bilinear", "bicubic", "lanczos"]:
        autres_hf_mid.append(feats[f]["hf_mid"])
    thr_nearest = (feats["nearest"]["hf_mid"] + max(autres_hf_mid)) / 2

    autres_e_mid = []
    for f in ["bilinear", "bicubic"]:
        autres_e_mid.append(feats[f]["e_mid"])
    thr_lanczos = (feats["lanczos"]["e_mid"] + max(autres_e_mid)) / 2

    thr_bicubic = (feats["bicubic"]["e_hf"] + feats["bilinear"]["e_hf"]) / 2

    print("\n=== Calibration des seuils ===")
    print(f"{'filtre':10s} | {'hf_mid':>8s} | {'e_mid':>8s} | {'e_hf':>8s}")
    print("-" * 45)
    for f in FILTRES:
        print(f"{f:10s} | {feats[f]['hf_mid']:8.4f} | {feats[f]['e_mid']:8.4f} | {feats[f]['e_hf']:8.4f}")
    print(f"\nSeuils calculés :")
    print(f"  THR_NEAREST = {thr_nearest:.4f}")
    print(f"  THR_LANCZOS = {thr_lanczos:.4f}")
    print(f"  THR_BICUBIC = {thr_bicubic:.4f}")
    print("\nMettez à jour ces valeurs en haut du script.")
    return thr_nearest, thr_lanczos, thr_bicubic


def visualiser(img_test, feats, filtre_predit, raison, chemin_sortie):
    delta  = feats["delta"]
    freqs  = feats["freqs"]
    profil = feats["profil"]
    W      = len(freqs)
    w      = 12

    mask_pos = freqs > 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f0f')
    couleur = COULEURS.get(filtre_predit, "#aaaaaa")

    axes[0].imshow(img_test, cmap='gray', aspect='auto')
    axes[0].set_title(f"Image analysée\nPrédiction : {filtre_predit.upper()}",
                      color=couleur, fontsize=13, fontweight='bold')
    axes[0].axis('off')

    ax = axes[1]
    ax.set_facecolor('#1a1a1a')

    ax.fill_between(freqs[mask_pos], profil[mask_pos], alpha=0.25, color=couleur)
    ax.plot(freqs[mask_pos], profil[mask_pos], color=couleur, lw=1.8, label="profil spectral")

    ax.axvspan(30, max(0, delta - w), alpha=0.08, color='white',
               label=f"bande MID  (e_mid={feats['e_mid']:.3f})")
    ax.axvspan(max(0, delta - w), delta + w, alpha=0.15, color='cyan',
               label=f"bande PIC  (e_pic={feats['e_pic']:.3f})")
    ax.axvspan(delta + w, W // 2, alpha=0.08, color='orange',
               label=f"bande HF   (e_hf={feats['e_hf']:.3f})")

    ax.axvline(delta, color='cyan', lw=1.5, ls='--', alpha=0.9,
               label=f"pic théorique k=1 ({delta} Hz)")

    ax.text(0.98, 0.95, f"hf_mid  = {feats['hf_mid']:.4f}   (seuil nearest  {THR_NEAREST:.4f})",
            transform=ax.transAxes, ha='right', va='top', color='white', fontsize=9, family='monospace')
    ax.text(0.98, 0.89, f"e_mid   = {feats['e_mid']:.4f}   (seuil lanczos  {THR_LANCZOS:.4f})",
            transform=ax.transAxes, ha='right', va='top', color='white', fontsize=9, family='monospace')
    ax.text(0.98, 0.83, f"e_hf    = {feats['e_hf']:.4f}   (seuil bicubic  {THR_BICUBIC:.4f})",
            transform=ax.transAxes, ha='right', va='top', color='white', fontsize=9, family='monospace')

    ax.set_title(f"Profil spectral 1D\n→ {raison}", color=couleur, fontsize=10, fontweight='bold')
    ax.set_xlabel("fréquence (bins)", color='gray', fontsize=10)
    ax.set_ylabel("magnitude normalisée", color='gray', fontsize=10)
    ax.tick_params(colors='gray', labelsize=8)
    ax.legend(fontsize=8, facecolor='#222', labelcolor='white', framealpha=0.8, loc='upper left')
    ax.set_xlim(0, W // 2)
    ax.set_ylim(bottom=0)

    fig.suptitle(f"Détection du filtre de rééchantillonnage  |  Prédit : {filtre_predit.upper()}  |  TARGET={W - delta}",
                 color='white', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(chemin_sortie, dpi=130, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"Figure sauvegardée : {chemin_sortie}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, default=IMAGE_PATH)
    parser.add_argument("--target",   type=int, default=TARGET)
    parser.add_argument("--calibrer", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    img = rgb2luminance(np.array(Image.open(args.image)))
    print(f"Image : {args.image}  |  shape={img.shape}  |  target={args.target}")

    if args.calibrer:
        calibrer(img, args.target)
        sys.exit(0)

    feats = extraire_features(img, args.target)
    filtre_predit, raison = predire_filtre(feats)

    print(f"\n{'─'*55}")
    print(f"  Filtre prédit  : {filtre_predit.upper()}")
    print(f"  Raison         : {raison}")
    print(f"{'─'*55}\n")

    chemin = os.path.join(OUT_DIR, "detection_filtre.png")
    visualiser(img, feats, filtre_predit, raison, chemin)