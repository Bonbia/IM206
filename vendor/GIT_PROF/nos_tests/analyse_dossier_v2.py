"""
analyse_dossier.py
──────────────────
Pour chaque image du dossier, applique la logique de estimate_factor.py :
  - version originale
  - versions rééchantillonnées x1.09, x1.20, x1.50, x2.00
Récupère les résultats de détection et génère stats + graphiques.

Usage :
    python analyse_dossier.py --dossier chemin/vers/images [--output resultats]

Dépendances :
    pip install numpy imageio Pillow matplotlib pandas tqdm
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Ajout de la racine du projet au path ──────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ird import detect_resampling
from src.misc import rgb2luminance

# ── Config ────────────────────────────────────────────────────────────────────
EXTENSIONS    = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
FACTEURS      = [1.09, 1.20, 1.50, 2.0]   # facteurs de rééchantillonnage
SEUIL_LOG_NFA = -5


# ── Fonctions extraites de estimate_factor.py ─────────────────────────────────

def charger_image(chemin):
    img = imageio.imread(str(chemin))
    return rgb2luminance(img.astype(np.float32))


def retailler(img, taille):
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.resize((taille, taille), Image.BICUBIC)).astype(np.float32)


def analyser_version(img, nom, vrai_k):
    """Reproduit exactement la logique de estimate_factor.py pour une version d'image."""
    largeur = img.shape[1]

    nfa = detect_resampling(
        img, preproc="rt", preproc_param={"rt_size": 3},
        window_ratio=0.10, nb_neighbor=20,
        direction="horizontal", is_jpeg=False, is_demosaic=False,
        max_distance=largeur - 1
    )

    log_nfa = np.log10(nfa + 1e-50)

    pics = [d for d in range(1, len(log_nfa))
            if log_nfa[d] < SEUIL_LOG_NFA and d <= largeur // 2]

    if not pics:
        return {
            "version":    nom,
            "vrai_k":     vrai_k,
            "detection":  False,
            "meilleur_d": None,
            "k_estime":   None,
            "erreur_k":   None,
            "log_nfa":    None,
        }

    meilleur_d = min(pics, key=lambda d: log_nfa[d])
    k_estime   = largeur / (largeur - meilleur_d)

    return {
        "version":    nom,
        "vrai_k":     vrai_k,
        "detection":  True,
        "meilleur_d": int(meilleur_d),
        "k_estime":   float(k_estime),
        "erreur_k":   float(abs(k_estime - vrai_k)),
        "log_nfa":    float(log_nfa[meilleur_d]),
    }


def analyser_image(chemin: Path) -> list[dict]:
    """
    Charge une image, crée les 5 versions (original + 4 facteurs),
    appelle analyser_version() sur chacune — exactement comme estimate_factor.py.
    Retourne une liste de dicts (une entrée par version).
    """
    resultats = []
    try:
        base        = charger_image(chemin)
        taille_base = base.shape[1]

        versions = [(base, "original", 1.0)] + [
            (retailler(base, int(taille_base * f)), f"x{f:.2f}", f)
            for f in FACTEURS
        ]

        for img, nom, vrai_k in versions:
            r = analyser_version(img, nom, vrai_k)
            r["fichier"] = chemin.name
            resultats.append(r)

    except Exception as e:
        # En cas d'erreur de chargement, on enregistre une ligne d'erreur
        resultats.append({
            "fichier":    chemin.name,
            "version":    "—",
            "vrai_k":     None,
            "detection":  False,
            "meilleur_d": None,
            "k_estime":   None,
            "erreur_k":   None,
            "log_nfa":    None,
            "erreur":     str(e),
        })

    return resultats


# ── Graphiques ────────────────────────────────────────────────────────────────

def generer_graphiques(df: pd.DataFrame, output_dir: Path) -> None:
    palette = {
        "bleu":   "#2563EB", "orange": "#F97316",
        "vert":   "#16A34A", "gris":   "#94A3B8",
        "fond":   "#F8FAFC", "texte":  "#1E293B",
    }

    df_det = df[df["detection"]]
    df_non = df[~df["detection"]]

    VERSIONS_ORDER = ["original"] + [f"x{f:.2f}" for f in FACTEURS]
    colors_v = [palette["bleu"], "#F97316", "#EAB308", "#16A34A", "#8B5CF6"]

    fig = plt.figure(figsize=(20, 14), facecolor=palette["fond"])
    fig.suptitle("Analyse de rééchantillonnage — toutes images × tous facteurs",
                 fontsize=15, fontweight="bold", color=palette["texte"], y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── 1. Taux de détection par version ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    taux = []
    for v in VERSIONS_ORDER:
        sub = df[df["version"] == v]
        t   = 100 * sub["detection"].sum() / max(len(sub), 1)
        taux.append(t)
    bars = ax1.bar(VERSIONS_ORDER, taux, color=colors_v, edgecolor="white", linewidth=0.6)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Taux de détection (%)", color=palette["texte"])
    ax1.set_title("Taux détection par version", fontsize=11, color=palette["texte"])
    ax1.set_facecolor(palette["fond"])
    ax1.tick_params(colors=palette["texte"])
    for bar, t in zip(bars, taux):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{t:.0f}%", ha="center", va="bottom", fontsize=8, color=palette["texte"])

    # ── 2. k estimé vs k réel par version ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for v, col in zip(VERSIONS_ORDER, colors_v):
        sub = df_det[df_det["version"] == v]
        if not sub.empty:
            ax2.scatter(sub["vrai_k"], sub["k_estime"], label=v,
                        color=col, alpha=0.7, s=30, edgecolors="none")
    lims = [df["vrai_k"].min() - 0.05, df["vrai_k"].max() + 0.05]
    ax2.plot(lims, lims, "k--", linewidth=1, label="k parfait")
    ax2.set_xlabel("k réel", color=palette["texte"])
    ax2.set_ylabel("k estimé", color=palette["texte"])
    ax2.set_title("k estimé vs k réel", fontsize=11, color=palette["texte"])
    ax2.legend(fontsize=7)
    ax2.set_facecolor(palette["fond"])
    ax2.tick_params(colors=palette["texte"])

    # ── 3. Erreur |k_estime - k_reel| par version (boxplot) ──────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    data_box = [df_det[df_det["version"] == v]["erreur_k"].dropna().values
                for v in VERSIONS_ORDER]
    bp = ax3.boxplot(data_box, labels=VERSIONS_ORDER, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 1.5})
    for patch, col in zip(bp["boxes"], colors_v):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax3.set_ylabel("|k estimé − k réel|", color=palette["texte"])
    ax3.set_title("Erreur sur k par version", fontsize=11, color=palette["texte"])
    ax3.set_facecolor(palette["fond"])
    ax3.tick_params(colors=palette["texte"])

    # ── 4. Distribution log NFA par version ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    for v, col in zip(VERSIONS_ORDER, colors_v):
        sub = df_det[df_det["version"] == v]["log_nfa"].dropna()
        if not sub.empty:
            ax4.hist(sub, bins=15, alpha=0.6, color=col, label=v, edgecolor="none")
    ax4.axvline(SEUIL_LOG_NFA, color="red", linestyle="--",
                linewidth=1.2, label=f"Seuil {SEUIL_LOG_NFA}")
    ax4.set_xlabel("log₁₀(NFA)", color=palette["texte"])
    ax4.set_ylabel("Nombre d'images", color=palette["texte"])
    ax4.set_title("Distribution log NFA par version", fontsize=11, color=palette["texte"])
    ax4.legend(fontsize=8)
    ax4.set_facecolor(palette["fond"])
    ax4.tick_params(colors=palette["texte"])

    # ── 5. Tableau récap ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    n_img = df["fichier"].nunique()
    rows  = [["Images analysées", str(n_img)]]
    for v in VERSIONS_ORDER:
        sub   = df[df["version"] == v]
        n_det = sub["detection"].sum()
        e_moy = sub[sub["detection"]]["erreur_k"].mean()
        rows.append([
            v,
            f"{n_det}/{len(sub)}  err={e_moy:.4f}" if not np.isnan(e_moy) else f"{n_det}/{len(sub)}"
        ])
    table = ax5.table(cellText=rows, colLabels=["Version", "Détectées / Erreur moy."],
                      cellLoc="left", loc="center", colWidths=[0.38, 0.62])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor(palette["bleu"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EFF6FF")
        else:
            cell.set_facecolor("white")
    ax5.set_title("Résumé par version", fontsize=11, color=palette["texte"], pad=12)

    out = output_dir / "rapport_graphiques.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=palette["fond"])
    plt.close(fig)
    print(f"  ✔ Graphiques : {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Détection de rééchantillonnage — batch dossier × 5 facteurs.")
    parser.add_argument("--dossier", required=True,
                        help="Dossier contenant les images source.")
    parser.add_argument("--output", default="resultats",
                        help="Dossier de sortie (défaut : 'resultats').")
    args = parser.parse_args()

    dossier    = Path(args.dossier)
    output_dir = Path(args.output)

    if not dossier.is_dir():
        print(f"Erreur : '{dossier}' n'est pas un dossier valide.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in dossier.rglob("*") if p.suffix.lower() in EXTENSIONS])
    if not images:
        print(f"Aucune image trouvée dans '{dossier}'.")
        sys.exit(0)

    print(f"\n🔍 {len(images)} image(s) × {1 + len(FACTEURS)} versions "
          f"= {len(images) * (1 + len(FACTEURS))} analyses\n")

    # ── Boucle principale ─────────────────────────────────────────────────────
    tous_resultats = []
    for chemin in tqdm(images, desc="Images", unit="img"):
        tous_resultats.extend(analyser_image(chemin))

    df = pd.DataFrame(tous_resultats)

    # Export CSV
    csv_path = output_dir / "resultats.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✔ CSV : {csv_path}")

    # Graphiques
    print("  📊 Génération des graphiques…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generer_graphiques(df, output_dir)

    # Résumé console
    print(f"\n{'─'*55}")
    print(f"{'Version':<12} {'Détectées':>10} {'Taux':>7} {'Erreur moy k':>14}")
    print(f"{'─'*55}")
    for v in ["original"] + [f"x{f:.2f}" for f in FACTEURS]:
        sub   = df[df["version"] == v]
        n_det = sub["detection"].sum()
        taux  = 100 * n_det / max(len(sub), 1)
        e_moy = sub[sub["detection"]]["erreur_k"].mean()
        e_str = f"{e_moy:.4f}" if not np.isnan(e_moy) else "—"
        print(f"{v:<12} {n_det:>10} {taux:>6.1f}% {e_str:>14}")
    print(f"{'─'*55}")
    print(f"\n  📁 Résultats dans : {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
