"""
detecter_filtre.py
==================
Détecte le filtre de rééchantillonnage utilisé sur une image,
uniquement à partir de l'analyse de son spectre de Fourier.
Aucun machine learning — uniquement des règles sur des features spectrales.

COMMENT ÇA MARCHE :
    Le rééchantillonnage (W → TARGET → W) laisse des traces spectrales
    caractéristiques selon le filtre utilisé :
    - nearest  : beaucoup d'énergie haute fréquence (aucun filtrage)
    - bilinear : énergie HF modérée, peu de concentration basse fréquence
    - bicubic  : similaire à bilinear mais avec un léger rebond HF
    - lanczos  : presque toute l'énergie concentrée en basse fréquence (meilleur anti-alias)

    On extrait 3 features du profil spectral 1D :
    - hf_mid  : ratio énergie HF / énergie mid  → sépare nearest des autres
    - e_mid   : énergie basse fréquence          → sépare lanczos des autres
    - e_hf    : énergie haute fréquence absolue  → sépare bicubic de bilinear

    Puis on applique un arbre de décision avec des seuils calibrés.

USAGE :
    python detecter_filtre.py                   # image par défaut (baboon.png)
    python detecter_filtre.py --image mon_img.png --target 350
    python detecter_filtre.py --calibrer        # recalibrer les seuils sur les 4 filtres
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter

# ─── Chemins ──────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(REPO_ROOT, "img", "baboon.png")
OUT_DIR    = os.path.join(REPO_ROOT, "results")
TARGET     = 350  # taille intermédiaire supposée lors du rééchantillonnage

# Seuils de décision (calibrés sur baboon.png, TARGET=350)
# Recalibrer avec --calibrer si vous changez d'image ou de TARGET
THR_NEAREST  = 0.5616  # hf_mid > THR_NEAREST  → nearest
THR_LANCZOS  = 0.7771  # e_mid  > THR_LANCZOS  → lanczos
THR_BICUBIC  = 0.2176  # e_hf   > THR_BICUBIC  → bicubic  (sinon bilinear)


# ─── Fonctions de base ────────────────────────────────────────────────────────

def rgb2luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)

def rank_transform(img: np.ndarray, sz: int = 3) -> np.ndarray:
    """Résidu haute fréquence : supprime le fond basse fréquence de l'image."""
    lm = uniform_filter(img.astype(np.float64), size=sz, mode='reflect')
    return (img - lm).astype(np.float32)

def formater(img: np.ndarray, taille: int, filtre: str) -> np.ndarray:
    FILTRES_PIL = {"nearest":  Image.NEAREST,
                   "bilinear": Image.BILINEAR,
                   "bicubic":  Image.BICUBIC,
                   "lanczos":  Image.LANCZOS}
    H, W  = img.shape
    pil   = Image.fromarray(img.astype(np.uint8))
    petit = pil.resize((taille, taille), resample=FILTRES_PIL[filtre])
    grand = petit.resize((W, H),         resample=FILTRES_PIL[filtre])
    return np.array(grand, dtype=np.float32)


# ─── Extraction des features spectrales ───────────────────────────────────────

def extraire_features(img_test: np.ndarray, target: int) -> dict:
    """
    Calcule 3 features à partir du profil 1D horizontal du spectre 2D.

    Paramètres
    ----------
    img_test : image grayscale float32 (déjà rééchantillonnée)
    target   : taille intermédiaire utilisée lors du rééchantillonnage

    Retourne
    --------
    dict avec les clés : e_pic, e_mid, e_hf, hf_mid, pic_mid
    """
    H, W  = img_test.shape
    delta = W - target   # fréquence théorique du pic k=1

    # Spectre 2D centré
    x    = rank_transform(img_test)
    F    = np.fft.fftshift(np.fft.fft2(x.astype(np.float64)))
    mag  = np.abs(F)

    # Profil 1D horizontal (moyenne sur l'axe vertical)
    prof  = mag.mean(axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(W, d=1.0/W))

    # Normalisation par le max hors DC
    prof_n = prof / (prof[np.abs(freqs) > 5].max() + 1e-9)

    # Définition des 3 bandes de fréquence
    w         = 12  # demi-largeur de la fenêtre autour du pic
    mask_pic  = (np.abs(freqs) >= delta - w) & (np.abs(freqs) <= delta + w) & (freqs != 0)
    mask_mid  = (np.abs(freqs) > 30) & (np.abs(freqs) < delta - w)
    mask_hf   = (np.abs(freqs) > delta + w) & (np.abs(freqs) < W // 2)

    e_pic = float(prof_n[mask_pic].mean()) if mask_pic.any() else 0.0
    e_mid = float(prof_n[mask_mid].mean()) if mask_mid.any() else 1.0
    e_hf  = float(prof_n[mask_hf].mean())  if mask_hf.any()  else 0.0

    return {
        "e_pic":   e_pic,
        "e_mid":   e_mid,
        "e_hf":    e_hf,
        "hf_mid":  e_hf  / (e_mid + 1e-9),
        "pic_mid": e_pic / (e_mid + 1e-9),
        "delta":   delta,
        "freqs":   freqs,
        "profil":  prof_n,
    }


# ─── Arbre de décision ────────────────────────────────────────────────────────

def predire_filtre(feats: dict,
                   thr_nearest: float = THR_NEAREST,
                   thr_lanczos: float = THR_LANCZOS,
                   thr_bicubic: float = THR_BICUBIC) -> tuple[str, str]:
    """
    Arbre de décision à 3 niveaux basé sur les features spectrales.

    Retourne (filtre_prédit, explication)
    """
    hf_mid = feats["hf_mid"]
    e_mid  = feats["e_mid"]
    e_hf   = feats["e_hf"]

    if hf_mid > thr_nearest:
        filtre = "nearest"
        raison = (f"hf_mid={hf_mid:.4f} > seuil {thr_nearest:.4f} "
                  f"→ beaucoup d'énergie HF, aucun filtrage passe-bas")
    elif e_mid > thr_lanczos:
        filtre = "lanczos"
        raison = (f"e_mid={e_mid:.4f} > seuil {thr_lanczos:.4f} "
                  f"→ énergie très concentrée en basse fréquence, meilleur anti-alias")
    elif e_hf > thr_bicubic:
        filtre = "bicubic"
        raison = (f"e_hf={e_hf:.4f} > seuil {thr_bicubic:.4f} "
                  f"→ légère remontée HF due au lobe négatif du noyau cubique")
    else:
        filtre = "bilinear"
        raison = (f"e_hf={e_hf:.4f} ≤ seuil {thr_bicubic:.4f} "
                  f"→ atténuation HF régulière, pas de rebond cubique")

    return filtre, raison


# ─── Calibration des seuils ───────────────────────────────────────────────────

def calibrer(img_ref: np.ndarray, target: int) -> tuple[float, float, float]:
    """
    Génère les 4 versions rééchantillonnées et calcule les seuils optimaux
    comme milieu entre les valeurs des filtres adjacents.
    """
    FILTRES = ["nearest", "bilinear", "bicubic", "lanczos"]
    feats   = {f: extraire_features(formater(img_ref, target, f), target) for f in FILTRES}

    thr_nearest = (feats["nearest"]["hf_mid"] +
                   max(feats[f]["hf_mid"] for f in ["bilinear","bicubic","lanczos"])) / 2

    thr_lanczos = (feats["lanczos"]["e_mid"] +
                   max(feats[f]["e_mid"] for f in ["bilinear","bicubic"])) / 2

    thr_bicubic = (feats["bicubic"]["e_hf"] + feats["bilinear"]["e_hf"]) / 2

    print("\n=== Calibration des seuils ===")
    print(f"{'filtre':10s} | {'hf_mid':>8s} | {'e_mid':>8s} | {'e_hf':>8s}")
    print("-"*45)
    for f in FILTRES:
        print(f"{f:10s} | {feats[f]['hf_mid']:8.4f} | {feats[f]['e_mid']:8.4f} | {feats[f]['e_hf']:8.4f}")
    print(f"\nSeuils calculés :")
    print(f"  THR_NEAREST = {thr_nearest:.4f}")
    print(f"  THR_LANCZOS = {thr_lanczos:.4f}")
    print(f"  THR_BICUBIC = {thr_bicubic:.4f}")
    print("\nMettez à jour ces valeurs en haut du script.")
    return thr_nearest, thr_lanczos, thr_bicubic


# ─── Visualisation ────────────────────────────────────────────────────────────

COULEURS = {"nearest": "#ff6b6b", "bilinear": "#ffd93d",
            "bicubic": "#6bcb77", "lanczos": "#4d96ff",
            "inconnu": "#aaaaaa"}

def visualiser(img_test: np.ndarray, feats: dict, filtre_predit: str,
               raison: str, chemin_sortie: str):
    """
    Figure : profil spectral annoté avec les bandes et les seuils de décision.
    """
    delta  = feats["delta"]
    freqs  = feats["freqs"]
    profil = feats["profil"]
    W      = len(freqs)

    mask_pos = freqs > 0
    w        = 12

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f0f')
    couleur = COULEURS.get(filtre_predit, "#aaaaaa")

    # ── Axe gauche : image rééchantillonnée ──
    axes[0].imshow(img_test, cmap='gray', aspect='auto')
    axes[0].set_title(f"Image analysée\nPrédiction : {filtre_predit.upper()}",
                      color=couleur, fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # ── Axe droit : profil spectral annoté ──
    ax = axes[1]
    ax.set_facecolor('#1a1a1a')

    # Profil (moitié positive)
    ax.fill_between(freqs[mask_pos], profil[mask_pos],
                    alpha=0.25, color=couleur)
    ax.plot(freqs[mask_pos], profil[mask_pos],
            color=couleur, lw=1.8, label="profil spectral")

    # Bandes colorées
    w_band = 12
    e_mid_v = feats["e_mid"]
    e_hf_v  = feats["e_hf"]
    e_pic_v = feats["e_pic"]

    # Zone MID
    ax.axvspan(30, max(0, delta - w_band), alpha=0.08, color='white',
               label=f"bande MID  (e_mid={e_mid_v:.3f})")
    # Zone PIC
    ax.axvspan(max(0, delta - w_band), delta + w_band, alpha=0.15, color='cyan',
               label=f"bande PIC  (e_pic={e_pic_v:.3f})")
    # Zone HF
    ax.axvspan(delta + w_band, W // 2, alpha=0.08, color='orange',
               label=f"bande HF   (e_hf={e_hf_v:.3f})")

    # Ligne du pic théorique
    ax.axvline(delta, color='cyan', lw=1.5, ls='--', alpha=0.9,
               label=f"pic théorique k=1 ({delta} Hz)")

    # Seuils annotés
    hf_mid = feats["hf_mid"]
    ax.text(0.98, 0.95, f"hf_mid  = {hf_mid:.4f}   (seuil nearest  {THR_NEAREST:.4f})",
            transform=ax.transAxes, ha='right', va='top',
            color='white', fontsize=9, family='monospace')
    ax.text(0.98, 0.89, f"e_mid   = {e_mid_v:.4f}   (seuil lanczos  {THR_LANCZOS:.4f})",
            transform=ax.transAxes, ha='right', va='top',
            color='white', fontsize=9, family='monospace')
    ax.text(0.98, 0.83, f"e_hf    = {e_hf_v:.4f}   (seuil bicubic  {THR_BICUBIC:.4f})",
            transform=ax.transAxes, ha='right', va='top',
            color='white', fontsize=9, family='monospace')

    ax.set_title(f"Profil spectral 1D\n→ {raison}",
                 color=couleur, fontsize=10, fontweight='bold')
    ax.set_xlabel("fréquence (bins)", color='gray', fontsize=10)
    ax.set_ylabel("magnitude normalisée", color='gray', fontsize=10)
    ax.tick_params(colors='gray', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.legend(fontsize=8, facecolor='#222', labelcolor='white',
              framealpha=0.8, loc='upper left')
    ax.set_xlim(0, W // 2)
    ax.set_ylim(bottom=0)

    fig.suptitle(f"Détection du filtre de rééchantillonnage  |  "
                 f"Prédit : {filtre_predit.upper()}  |  TARGET={W - delta}",
                 color='white', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(chemin_sortie, dpi=130, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"Figure sauvegardée : {chemin_sortie}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Détection du filtre de rééchantillonnage par analyse spectrale")
    parser.add_argument("--image",    type=str, default=IMAGE_PATH, help="Image à analyser")
    parser.add_argument("--target",   type=int, default=TARGET,     help="Taille intermédiaire de rééchantillonnage")
    parser.add_argument("--calibrer", action="store_true",          help="Recalibrer les seuils sur les 4 filtres")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    raw = np.array(Image.open(args.image))
    img = rgb2luminance(raw)
    print(f"Image : {args.image}  |  shape={img.shape}  |  target={args.target}")

    if args.calibrer:
        calibrer(img, args.target)
        sys.exit(0)

    # ── Extraction + Prédiction ──
    feats           = extraire_features(img, args.target)
    filtre_predit, raison = predire_filtre(feats)

    print(f"\n{'─'*55}")
    print(f"  Filtre prédit  : {filtre_predit.upper()}")
    print(f"  Raison         : {raison}")
    print(f"{'─'*55}\n")

    chemin = os.path.join(OUT_DIR, "detection_filtre.png")
    visualiser(img, feats, filtre_predit, raison, chemin)
