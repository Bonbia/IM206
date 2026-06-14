import time, numpy as np, imageio.v3 as imageio
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.ird import detect_resampling
from src.misc import rgb2luminance


def charger_image(chemin):
    img = imageio.imread(chemin)
    return rgb2luminance(img.astype(np.float32))


def detection(img):
    return detect_resampling(img, preproc="rt", preproc_param={"rt_size": 3},
        window_ratio=0.10, nb_neighbor=20, direction="horizontal",
        is_jpeg=False, is_demosaic=False, max_distance=img.shape[1] - 1)


img = charger_image(os.path.join(ROOT, "img/baboon_666.png"))
h, w = img.shape

crops = [
    ("Full 666x666",   img),
    ("Crop 50% 333x333", img[h//4:3*h//4,          w//4:3*w//4]),
    ("Crop 25% 167x167", img[3*h//8:5*h//8,        3*w//8:5*w//8]),
    ("Crop 10% 67x67",   img[int(h*.45):int(h*.55), int(w*.45):int(w*.55)]),
]

N_RUNS = 5

print(f"{'crop':<22} {'pixels':>8} {'temps moy':>10} {'ecart':>8} {'pics detectes'}")
print("-" * 70)

temps_ref = None
for nom, crop in crops:
    nb_pixels = crop.shape[0] * crop.shape[1]
    mesures = []
    nfa_last = None

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        nfa_last = detection(crop)
        mesures.append(time.perf_counter() - t0)

    moy = np.mean(mesures)
    ect = np.std(mesures)

    if temps_ref is None:
        temps_ref = moy

    log_nfa = np.log10(nfa_last + 1e-50)
    pics = []
    for d in range(1, len(log_nfa)):
        if log_nfa[d] < -5:
            pics.append(d)

    acceleration = f"x{temps_ref/moy:.1f}" if moy < temps_ref else "ref"
    pics_str = str(pics[:4]) if pics else "aucun"
    print(f"{nom:<22} {nb_pixels:>8} {moy:>9.3f}s {ect:>7.3f}s  {pics_str}  ({acceleration})")
