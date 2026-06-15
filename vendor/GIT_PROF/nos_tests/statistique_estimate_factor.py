import numpy as np
import imageio.v3 as imageio
from PIL import Image
import sys, os, argparse, warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.ird import detect_resampling
from src.misc import rgb2luminance


# ── Fonctions identiques au script de référence ───────────────────────────────

def charger_image(chemin):
    img = imageio.imread(chemin)
    return rgb2luminance(img.astype(np.float32))


def retailler(img, taille):
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.resize((taille, taille), Image.NEAREST)).astype(np.float32)


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dossier", required=True)
parser.add_argument("--output", default="results")
args = parser.parse_args()

extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
images_fichiers = sorted([
    os.path.join(args.dossier, f)
    for f in os.listdir(args.dossier)
    if os.path.splitext(f)[1].lower() in extensions
])

facteurs = [1.09, 1.20, 1.50, 2.0]
versions  = ["original"] + [f"x{f:.2f}" for f in facteurs]

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n🔍 {len(images_fichiers)} image(s) trouvée(s)\n")

# ── Boucle principale — même logique que batch_estimate_factor.py ─────────────
# resultats : liste de dicts, une entrée par (image × version)
resultats = []

for chemin in images_fichiers:
    nom_fichier = os.path.basename(chemin)

    try:
        base        = charger_image(chemin)
        taille_base = base.shape[1]
    except Exception as e:
        print(f"✗ {nom_fichier} : erreur chargement ({e})")
        for v, k_reel in zip(versions, [1.0] + facteurs):
            resultats.append({"fichier": nom_fichier, "version": v, "vrai_k": k_reel,
                               "detecte": False, "k_estime": None, "erreur_k": None,
                               "log_nfa": None, "meilleur_d": None, "erreur_chargement": True})
        continue

    images_versions = [(base, "original", 1.0)] + [
        (retailler(base, int(taille_base * f)), f"x{f:.2f}", f)
        for f in facteurs
    ]

    for img, nom, vrai_k in images_versions:
        largeur = img.shape[1]

        nfa     = detect_resampling(
            img, preproc="rt", preproc_param={"rt_size": 3},
            window_ratio=0.10, nb_neighbor=20,
            direction="horizontal", is_jpeg=False, is_demosaic=False,
            max_distance=largeur - 1
        )
        log_nfa = np.log10(nfa + 1e-50)

        pics = [d for d in range(1, len(log_nfa))
                if log_nfa[d] < -5 and d <= largeur // 2]

        if not pics:
            resultats.append({"fichier": nom_fichier, "version": nom, "vrai_k": vrai_k,
                               "detecte": False, "k_estime": None, "erreur_k": None,
                               "log_nfa": None, "meilleur_d": None, "erreur_chargement": False})
            continue

        meilleur_d = min(pics, key=lambda d: log_nfa[d])
        k_estime   = largeur / (largeur - meilleur_d)
        erreur_k   = abs(k_estime - vrai_k)
        lnfa       = float(log_nfa[meilleur_d])

        resultats.append({"fichier": nom_fichier, "version": nom, "vrai_k": vrai_k,
                           "detecte": True, "k_estime": float(k_estime),
                           "erreur_k": float(erreur_k), "log_nfa": lnfa,
                           "meilleur_d": int(meilleur_d), "erreur_chargement": False})

    print(f"  ✔ {nom_fichier}")

# ── CSV brut ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(resultats)
df.to_csv(output_dir / "resultats_bruts.csv", index=False)

# ── Statistiques par version ──────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"{'Version':<12} {'N img':>6} {'Détectées':>10} {'Taux':>7} {'Erreur moy':>11} {'Erreur med':>11}")
print(f"{'─'*65}")

stats_par_version = []
for v in versions:
    sub     = df[df["version"] == v]
    n_total = len(sub)
    n_det   = sub["detecte"].sum()
    taux    = 100 * n_det / max(n_total, 1)
    det     = sub[sub["detecte"]]
    e_moy   = det["erreur_k"].mean() if not det.empty else float("nan")
    e_med   = det["erreur_k"].median() if not det.empty else float("nan")
    stats_par_version.append({"version": v, "n_total": n_total, "n_detectees": int(n_det),
                               "taux_pct": taux, "erreur_moy": e_moy, "erreur_med": e_med})
    e_moy_s = f"{e_moy:.4f}" if not np.isnan(e_moy) else "—"
    e_med_s = f"{e_med:.4f}" if not np.isnan(e_med) else "—"
    print(f"{v:<12} {n_total:>6} {int(n_det):>10} {taux:>6.1f}% {e_moy_s:>11} {e_med_s:>11}")

print(f"{'─'*65}")

df_stats = pd.DataFrame(stats_par_version)
df_stats.to_csv(output_dir / "statistiques_par_version.csv", index=False)

# ── Graphiques ────────────────────────────────────────────────────────────────
palette = {"bleu": "#2563EB", "orange": "#F97316", "vert": "#16A34A",
           "gris": "#94A3B8", "fond": "#F8FAFC", "texte": "#1E293B"}
colors_v = ["#2563EB", "#F97316", "#EAB308", "#16A34A", "#8B5CF6"]

fig = plt.figure(figsize=(20, 13), facecolor=palette["fond"])
fig.suptitle("Statistiques de détection de rééchantillonnage",
             fontsize=15, fontweight="bold", color=palette["texte"], y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# 1. Taux de détection par version
ax1 = fig.add_subplot(gs[0, 0])
taux_vals = [s["taux_pct"] for s in stats_par_version]
bars = ax1.bar(versions, taux_vals, color=colors_v, edgecolor="white", linewidth=0.6)
ax1.set_ylim(0, 115)
ax1.set_ylabel("Taux de détection (%)", color=palette["texte"])
ax1.set_title("Taux de détection par version", fontsize=11, color=palette["texte"])
ax1.set_facecolor(palette["fond"])
ax1.tick_params(colors=palette["texte"])
for bar, t in zip(bars, taux_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{t:.0f}%", ha="center", fontsize=9, color=palette["texte"])

# 2. Erreur moyenne et médiane par version
ax2 = fig.add_subplot(gs[0, 1])
e_moy_vals = [s["erreur_moy"] for s in stats_par_version]
e_med_vals = [s["erreur_med"] for s in stats_par_version]
x = np.arange(len(versions))
ax2.bar(x - 0.2, e_moy_vals, 0.4, label="Moyenne", color=palette["orange"], edgecolor="white")
ax2.bar(x + 0.2, e_med_vals, 0.4, label="Médiane", color=palette["bleu"], edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(versions)
ax2.set_ylabel("|k estimé − k réel|", color=palette["texte"])
ax2.set_title("Erreur sur k par version", fontsize=11, color=palette["texte"])
ax2.legend(fontsize=8)
ax2.set_facecolor(palette["fond"])
ax2.tick_params(colors=palette["texte"])

# 3. Boxplot erreur par version
ax3 = fig.add_subplot(gs[0, 2])
data_box = [df[(df["version"] == v) & df["detecte"]]["erreur_k"].dropna().values for v in versions]
bp = ax3.boxplot(data_box, labels=versions, patch_artist=True,
                 medianprops={"color": "white", "linewidth": 1.5})
for patch, col in zip(bp["boxes"], colors_v):
    patch.set_facecolor(col); patch.set_alpha(0.8)
ax3.set_ylabel("|k estimé − k réel|", color=palette["texte"])
ax3.set_title("Distribution erreur (boxplot)", fontsize=11, color=palette["texte"])
ax3.set_facecolor(palette["fond"])
ax3.tick_params(colors=palette["texte"])

# 4. Scatter k_estime vs vrai_k
ax4 = fig.add_subplot(gs[1, 0:2])
df_det = df[df["detecte"]]
for v, col in zip(versions, colors_v):
    sub = df_det[df_det["version"] == v]
    if not sub.empty:
        ax4.scatter(sub["vrai_k"], sub["k_estime"], label=v, color=col, alpha=0.7, s=35, edgecolors="none")
lim = [df["vrai_k"].min() - 0.05, df["vrai_k"].max() + 0.05]
ax4.plot(lim, lim, "k--", linewidth=1, label="k parfait")
ax4.set_xlabel("k réel", color=palette["texte"])
ax4.set_ylabel("k estimé", color=palette["texte"])
ax4.set_title("k estimé vs k réel", fontsize=11, color=palette["texte"])
ax4.legend(fontsize=8)
ax4.set_facecolor(palette["fond"])
ax4.tick_params(colors=palette["texte"])

# 5. Tableau récap
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
rows = [["Images analysées", str(len(df["fichier"].unique()))]]
for s in stats_par_version:
    e = f"{s['erreur_moy']:.4f}" if not np.isnan(s["erreur_moy"]) else "—"
    rows.append([s["version"], f"{s['n_detectees']}/{s['n_total']}  ({s['taux_pct']:.0f}%)  err={e}"])
table = ax5.table(cellText=rows, colLabels=["Version", "Détectées / Taux / Erreur moy."],
                  cellLoc="left", loc="center", colWidths=[0.28, 0.72])
table.auto_set_font_size(False)
table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#CBD5E1")
    if r == 0:
        cell.set_facecolor(palette["bleu"]); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EFF6FF")
    else:
        cell.set_facecolor("white")
ax5.set_title("Résumé par version", fontsize=11, color=palette["texte"], pad=12)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig.savefig(output_dir / "rapport_graphiques.png", dpi=150, bbox_inches="tight", facecolor=palette["fond"])
plt.close(fig)

print(f"\n  ✔ Graphiques  : {(output_dir / 'rapport_graphiques_nearest.png').resolve()}")
print(f"  ✔ CSV brut    : {(output_dir / 'resultats_bruts_nearest.csv').resolve()}")
print(f"  ✔ CSV stats   : {(output_dir / 'statistiques_par_version_nearest.csv').resolve()}\n")