import numpy as np, imageio.v3 as imageio
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.ird import detect_resampling
from src.misc import rgb2luminance


def charger_image(chemin):
    img = imageio.imread(chemin)
    return rgb2luminance(img.astype(np.float32))


def detection(img, preproc="rt", is_jpeg=False, direction="horizontal"):
    max_d = img.shape[1] - 1 if direction == "horizontal" else img.shape[0] - 1
    return detect_resampling(
        img, preproc=preproc,
        preproc_param={"rt_size": 3} if preproc == "rt" else None,
        window_ratio=0.10, nb_neighbor=20,
        direction=direction, is_jpeg=is_jpeg, is_demosaic=False,
        max_distance=max_d
    )


baboon  = charger_image(os.path.join(ROOT, "img/baboon_666.png"))
pashmina = charger_image(os.path.join(ROOT, "img/pashmina_720.jpg"))

# ── test 1 : crop sur baboon ──────────────────────────────────────────────────
print("TEST 1 — crop sur baboon_666.png")
print(f"{'crop':<20} {'taille':>12} {'pics detectes (seuil -5)'}")
print("-" * 60)

h, w = baboon.shape
crops = [
    ("Full 666x666",    baboon),
    ("Crop 50%",        baboon[h//4:3*h//4,   w//4:3*w//4]),
    ("Crop 25%",        baboon[3*h//8:5*h//8, 3*w//8:5*w//8]),
]

for nom, img in crops:
    nfa = detection(img, preproc="rt")
    log_nfa = np.log10(nfa + 1e-50)

    pics = []
    for d in range(1, len(log_nfa)):
        if log_nfa[d] < -5:
            pics.append(d)

    taille = f"{img.shape[0]}x{img.shape[1]}"
    pics_str = str(pics[:5]) if pics else "aucun"
    print(f"{nom:<20} {taille:>12} {pics_str}")

# ── test 2 : is_jpeg sur pashmina ────────────────────────────────────────────
print("\nTEST 2 — is_jpeg sur pashmina_720.jpg")
print(f"{'mode':<25} {'pics detectes (seuil -7)'}")
print("-" * 55)

for nom, is_jpeg in [("sans is_jpeg", False), ("avec is_jpeg", True)]:
    nfa = detection(pashmina, preproc="tv", is_jpeg=is_jpeg)
    log_nfa = np.log10(nfa + 1e-50)

    pics = []
    for d in range(1, len(log_nfa)):
        if log_nfa[d] < -7:
            pics.append(d)

    pics_str = str(pics[:5]) if pics else "aucun"
    print(f"{nom:<25} {pics_str}")
