import numpy as np
import imageio.v3 as imageio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.ird import detect_resampling, detect_resampling_with_cross_val
from src.misc import rgb2luminance


def charger_image(chemin):
    img = imageio.imread(chemin)
    return rgb2luminance(img.astype(np.float32))


def retailler(img, taille):
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.resize((taille, taille), Image.BICUBIC)).astype(np.float32)


def detection(img, direction="horizontal", preproc="rt"):
    max_d = img.shape[1] - 1 if direction == "horizontal" else img.shape[0] - 1
    return detect_resampling(
        img, preproc=preproc,
        preproc_param={"rt_size": 3} if preproc == "rt" else None,
        window_ratio=0.10, nb_neighbor=20,
        direction=direction, is_jpeg=False, is_demosaic=False,
        max_distance=max_d
    )


def detection_both(img, preproc="rt"):
    return detect_resampling_with_cross_val(
        img, preproc=preproc,
        preproc_param={"rt_size": 3} if preproc == "rt" else None,
        window_ratio=0.10, nb_neighbor=20,
        is_jpeg=False, is_demosaic=False
    )


print("Chargement baboon.png (512×512)...")
base = charger_image("/tmp/vendor/GIT_PROF/img/baboon.png")

# ── test 3 : ratios de rééchantillonnage ──────────────────────────────────────
print("\nTest 3 : ratios de rééchantillonnage...")
cibles  = [560, 614, 666, 768]
couleurs3 = ["#e66101", "#d01c8b", "#4dac26", "#0571b0"]

nfas3 = {}
for t in cibles:
    print(f"  Génération {t}×{t}...")
    img_r = retailler(base, t)
    nfas3[t] = detection(img_r)

# ── test 4 : h / v / both ─────────────────────────────────────────────────────
print("\nTest 4 : directions h / v / both sur baboon_666...")
img666 = charger_image("/tmp/vendor/GIT_PROF/img/baboon_666.png")

nfa_h    = detection(img666, "horizontal")
nfa_v    = detection(img666, "vertical")
nfa_both = detection_both(img666)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.suptitle("Tests complémentaires IRD — lot 2", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)
seuil = -5

# ligne 1 : un graphe par ratio
for col, t in enumerate(cibles):
    ax = fig.add_subplot(gs[0, col])
    log_nfa  = np.log10(nfas3[t] + 1e-50)
    xs       = np.arange(len(log_nfa))
    attendu  = t - 512

    ax.plot(xs, log_nfa, color=couleurs3[col], linewidth=0.9)
    ax.axhline(seuil, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(attendu, color="black", linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"attendu d={attendu}")

    pics = []
    for d in range(len(log_nfa)):
        if log_nfa[d] < seuil:
            pics.append(d)

    if pics:
        ax.scatter(pics, log_nfa[pics], color="red", s=18, zorder=5)
        for d in pics[:4]:
            ax.annotate(f"d={d}", (d, log_nfa[d]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="red")

    statut = "✓ Détecté" if pics else "✗ Non détecté"
    ax.set_facecolor("#f8fff8" if pics else "#fff8f8")
    ax.text(0.97, 0.05, statut, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color="green" if pics else "red", fontweight="bold")

    ratio = t / 512
    ax.set_title(f"512 → {t} (×{ratio:.2f})\nd attendu = {attendu}", fontsize=9)
    ax.set_xlabel("distance d", fontsize=8)
    ax.set_ylabel("log₁₀ NFA(d)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")

# ligne 2 : h / v / both / superposition
jeux = [
    (nfa_h,    "Direction H seulement",   "#2c7bb6"),
    (nfa_v,    "Direction V seulement",   "#d7191c"),
    (nfa_both, "Both (cross-validation)", "#1a9641"),
]

for col, (nfa, titre, c) in enumerate(jeux):
    ax = fig.add_subplot(gs[1, col])
    log_nfa = np.log10(nfa + 1e-50)
    xs      = np.arange(len(log_nfa))

    ax.plot(xs, log_nfa, color=c, linewidth=0.9)
    ax.axhline(seuil, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(154, color="black", linewidth=0.8, linestyle=":", alpha=0.6, label="d=154 attendu")

    pics = []
    for d in range(len(log_nfa)):
        if log_nfa[d] < seuil:
            pics.append(d)

    if pics:
        ax.scatter(pics, log_nfa[pics], color="red", s=18, zorder=5)
        for d in pics[:4]:
            ax.annotate(f"d={d}", (d, log_nfa[d]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="red")

    statut = "✓ Détecté" if pics else "✗ Non détecté"
    ax.set_facecolor("#f8fff8" if pics else "#fff8f8")
    ax.text(0.97, 0.05, statut, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color="green" if pics else "red", fontweight="bold")

    ax.set_title(titre, fontsize=9)
    ax.set_xlabel("distance d", fontsize=8)
    ax.set_ylabel("log₁₀ NFA(d)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

# superposition des 3
ax_super = fig.add_subplot(gs[1, 3])
for nfa, titre, c in jeux:
    log_nfa = np.log10(nfa + 1e-50)
    xs      = np.arange(len(log_nfa))
    ax_super.plot(xs, log_nfa, color=c, linewidth=0.85, alpha=0.85, label=titre.split(" ")[1])
ax_super.axhline(seuil, color="gray", linewidth=0.7, linestyle="--")
ax_super.axvline(154, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax_super.set_title("Superposition H / V / Both", fontsize=9)
ax_super.set_xlabel("distance d", fontsize=8)
ax_super.set_ylabel("log₁₀ NFA(d)", fontsize=8)
ax_super.tick_params(labelsize=7)
ax_super.legend(fontsize=7)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
plt.savefig(os.path.join(ROOT, "results/test_ratio_bothdir.png"), dpi=150, bbox_inches="tight")
print("\nFigure sauvegardée : results/test_ratio_bothdir.png")
plt.close()
print("Terminé.")