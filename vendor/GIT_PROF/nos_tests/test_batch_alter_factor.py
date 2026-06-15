import numpy as np
import imageio.v3 as imageio
from PIL import Image
import sys, os, argparse

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


parser = argparse.ArgumentParser()
parser.add_argument("--dossier", required=True)
args = parser.parse_args()

SEUIL = -5

extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
images_fichiers = sorted([
    os.path.join(args.dossier, f)
    for f in os.listdir(args.dossier)
    if os.path.splitext(f)[1].lower() in extensions
])

facteurs = [1.09, 1.20, 1.50, 2]

for chemin in images_fichiers:
    nom_fichier = os.path.basename(chemin)
    print(f"\n{'='*75}")
    print(f"Image : {nom_fichier}")
    print(f"{'='*75}")

    base = charger_image(chemin)
    taille_base = base.shape[1]

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
            max_distance=largeur - 1
        )

        log_nfa = np.log10(nfa + 1e-50)

        # premier pic significatif en partant de d=1
        meilleur_d = None
        for d in range(1, largeur // 2):
            if log_nfa[d] < SEUIL:
                meilleur_d = d
                break

        if meilleur_d is None:
            print(f"{nom:<25} {'—':>5} {'—':>10} {vrai_k:>10.4f} {'—':>8} {'—':>9}")
            continue

        k      = largeur / (largeur - meilleur_d)
        erreur = abs(k - vrai_k)
        lnfa   = log_nfa[meilleur_d]

        print(f"{nom:<25} {meilleur_d:>5} {k:>10.4f} {vrai_k:>10.4f} {erreur:>8.4f} {lnfa:>9.2f}")
