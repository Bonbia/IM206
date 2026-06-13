"""
  Test 3 : Effet du ratio de rééchantillonnage sur le NFA
           On génère soi-même des images rééchantillonnées à différents facteurs
           et on regarde si le pic détecté se déplace comme prévu."""
           
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

WINDOW_RATIO = 0.10
NB_NEIGHBOR  = 20
LOG_THRESHOLDS = {"rt": -5, "tv": -7}


def load_gray(path):
    img = imageio.imread(path)
    return rgb2luminance(img.astype(np.float32))


def resample(img_gray, target_size):
    """Rééchantillonne une image carrée vers target_size x target_size (PIL BICUBIC)."""
    pil = Image.fromarray(img_gray.astype(np.uint8))
    pil_r = pil.resize((target_size, target_size), Image.BICUBIC)
    return np.array(pil_r).astype(np.float32)


def run_h(img, preproc="rt"):
    nfa = detect_resampling(
        img, preproc=preproc,
        preproc_param={"rt_size": 3} if preproc == "rt" else None,
        window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR,
        direction="horizontal", is_jpeg=False, is_demosaic=False,
        max_distance=img.shape[1] - 1
    )
    return nfa


def run_v(img, preproc="rt"):
    nfa = detect_resampling(
        img, preproc=preproc,
        preproc_param={"rt_size": 3} if preproc == "rt" else None,
        window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR,
        direction="vertical", is_jpeg=False, is_demosaic=False,
        max_distance=img.shape[0] - 1
    )
    return nfa


def run_both(img, preproc="rt"):
    nfa = detect_resampling_with_cross_val(
        img, preproc=preproc,
        preproc_param={"rt_size": 3} if preproc == "rt" else None,
        window_ratio=WINDOW_RATIO, nb_neighbor=NB_NEIGHBOR,
        is_jpeg=False, is_demosaic=False
    )
    return nfa


def lognfa(nfa):
    return np.log10(nfa + 1e-50)


# ── Chargement image de base ──────────────────────────────────────────────────
print("Chargement baboon.png (512×512)...")
base = load_gray("/tmp/vendor/GIT_PROF/img/baboon.png")   # 512×512

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 : différents ratios de rééchantillonnage
# On upsample baboon 512→ 560, 614, 666, 768
# Distance théorique attendue : target - 512
# ══════════════════════════════════════════════════════════════════════════════
print("\nTest 3 : ratios de rééchantillonnage...")
targets = [560, 614, 666, 768]
colors3 = ["#e66101", "#d01c8b", "#4dac26", "#0571b0"]

nfas3 = {}
for t in targets:
    print(f"  Génération {t}×{t}...")
    img_r = resample(base, t)
    nfas3[t] = run_h(img_r)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 : direction both vs h vs v
# Sur baboon_666.png (rééchantillonnage isotrope → doit apparaître H et V)
# ══════════════════════════════════════════════════════════════════════════════
print("\nTest 4 : directions h / v / both sur baboon_666...")
img666 = load_gray("/tmp/vendor/GIT_PROF/img/baboon_666.png")

nfa_h    = run_h(img666)
nfa_v    = run_v(img666)
nfa_both = run_both(img666)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.suptitle("Tests complémentaires IRD — lot 2", fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)

thresh = LOG_THRESHOLDS["rt"]

# ── Ligne 1 : un graphe par ratio ─────────────────────────────────────────────
for col, (t, col_color) in enumerate(zip(targets, colors3)):
    ax = fig.add_subplot(gs[0, col])
    nfa = nfas3[t]
    ln  = lognfa(nfa)
    xs  = np.arange(len(ln))
    expected = t - 512   # distance théorique

    ax.plot(xs, ln, color=col_color, linewidth=0.9)
    ax.axhline(thresh, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(expected, color="black", linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"attendu d={expected}")

    sig = xs[ln < thresh]
    if len(sig):
        ax.scatter(sig, ln[sig], color="red", s=18, zorder=5)
        for d in sig[:4]:
            ax.annotate(f"d={d}", (d, ln[d]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="red")

    detected = "✓ Détecté" if len(sig) > 0 else "✗ Non détecté"
    ax.set_facecolor("#f8fff8" if len(sig) > 0 else "#fff8f8")
    ax.text(0.97, 0.05, detected, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color="green" if len(sig) else "red", fontweight="bold")

    ratio = t / 512
    ax.set_title(f"512 → {t} (×{ratio:.2f})\nd attendu = {expected}", fontsize=9)
    ax.set_xlabel("distance d", fontsize=8)
    ax.set_ylabel("log₁₀ NFA(d)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")

# ── Ligne 2 : h / v / both / superposition ───────────────────────────────────
datasets4 = [
    (nfa_h,    "Direction H seulement",   "#2c7bb6"),
    (nfa_v,    "Direction V seulement",   "#d7191c"),
    (nfa_both, "Both (cross-validation)", "#1a9641"),
]

for col, (nfa, title, c) in enumerate(datasets4):
    ax = fig.add_subplot(gs[1, col])
    ln = lognfa(nfa)
    xs = np.arange(len(ln))
    ax.plot(xs, ln, color=c, linewidth=0.9)
    ax.axhline(thresh, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(154, color="black", linewidth=0.8, linestyle=":", alpha=0.6, label="d=154 attendu")

    sig = xs[ln < thresh]
    if len(sig):
        ax.scatter(sig, ln[sig], color="red", s=18, zorder=5)
        for d in sig[:4]:
            ax.annotate(f"d={d}", (d, ln[d]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="red")

    detected = "✓ Détecté" if len(sig) > 0 else "✗ Non détecté"
    ax.set_facecolor("#f8fff8" if len(sig) > 0 else "#fff8f8")
    ax.text(0.97, 0.05, detected, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color="green" if len(sig) else "red", fontweight="bold")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("distance d", fontsize=8)
    ax.set_ylabel("log₁₀ NFA(d)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

# Superposition des 3
ax_super = fig.add_subplot(gs[1, 3])
for nfa, label, c in datasets4:
    ln = lognfa(nfa)
    xs = np.arange(len(ln))
    ax_super.plot(xs, ln, color=c, linewidth=0.85, alpha=0.85, label=label.split(" ")[1])
ax_super.axhline(thresh, color="gray", linewidth=0.7, linestyle="--")
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
