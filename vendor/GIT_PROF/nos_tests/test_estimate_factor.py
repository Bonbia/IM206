import numpy as np
import imageio.v3 as imageio
from PIL import Image
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.ird import detect_resampling
from src.misc import rgb2luminance


def charger_image(chemin):
    img = imageio.imread(chemin)
    return rgb2luminance(img.astype(np.float32))


def retailler(img, taille):
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.resize((taille, taille), Image.BICUBIC)).astype(np.float32)


base = charger_image(os.path.join(ROOT, "img/baboon.png"))

images = [
    (base,                                                    "baboon original 512",  1.0,     None),
    (charger_image(os.path.join(ROOT,"img/baboon_666.png")), "baboon 666",           666/512, 154),
    (retailler(base, 560),                                    "512 -> 560",           560/512, 48),
    (retailler(base, 614),                                    "512 -> 614",           614/512, 102),
    (retailler(base, 768),                                    "512 -> 768",           768/512, 256),
]

print(f"{'image':<25} {'d*':>5} {'k estimé':>10} {'k réel':>10} {'erreur':>8} {'log NFA':>9}")
print("-" * 75)

for img, nom, vrai_k, vrai_d in images:
    largeur = img.shape[1]

    nfa = detect_resampling(
        img, preproc="rt", preproc_param={"rt_size": 3},
        window_ratio=0.10, nb_neighbor=20,
        direction="horizontal", is_jpeg=False, is_demosaic=False,
        max_distance=largeur - 1
    )

    log_nfa = np.log10(nfa + 1e-50)

    # cherche les pics significatifs
    pics = []
    for d in range(1, len(log_nfa)):
        if log_nfa[d] < -5 and d <= largeur // 2:
            pics.append(d)

    if len(pics) == 0:
        print(f"{nom:<25} {'—':>5} {'—':>10} {vrai_k:>10.4f} {'—':>8} {'—':>9}")
        continue

    # meilleur pic = celui avec le plus petit log_nfa
    meilleur_d = pics[0]
    for d in pics:
        if log_nfa[d] < log_nfa[meilleur_d]:
            meilleur_d = d

    k      = largeur / (largeur - meilleur_d)
    erreur = abs(k - vrai_k)
    lnfa   = log_nfa[meilleur_d]

    print(f"{nom:<25} {meilleur_d:>5} {k:>10.4f} {vrai_k:>10.4f} {erreur:>8.4f} {lnfa:>9.2f}")
