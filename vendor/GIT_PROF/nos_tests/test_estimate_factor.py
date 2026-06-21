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

def retailler(img, facteur):
    h, w = img.shape[:2]
    nouvelle_largeur = int(w * facteur)
    pil = Image.fromarray(img.astype(np.uint8))
    img_grande = np.array(pil.resize((nouvelle_largeur, h), Image.NEAREST)).astype(np.float32)
    return img_grande[:, :w]

base = charger_image(os.path.join(ROOT, "img/png_light/r5a671b67t.png"))
print(f"shape base : {base.shape}")  # hauteur x largeur
taille_base = base.shape[1]
facteurs = [1.09, 1.20, 1.50,1.6,1.8,1.9,2]  

images = [
    (base, "original", 1.0, None),
] + [
    (retailler(base, int(taille_base * f)), f"x{f:.2f}", f, None)
    for f in facteurs
    if f > 1.0  
]

print(f"{'image':<25} {'d*':>5} {'k estimé':>10} {'k réel':>10} {'erreur':>8} {'log NFA':>9}")
print("-" * 75)

for img, nom, vrai_k, vrai_d in images:
    largeur = img.shape[1]

    nfa = detect_resampling(
            img, preproc="rt", preproc_param={"rt_size": 3},
            window_ratio=0.10, nb_neighbor=20,
            direction="horizontal", is_jpeg=False, is_demosaic=False,
            max_distance=min(largeur - 1, 500)
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
