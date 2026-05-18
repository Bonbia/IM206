'''Mise en place de la partie SCA de la Pipeline'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import Pretraitement 
    

#-----------------------------------------------------------------------------------------------
### DFT2D/IFDT2D
#-----------------------------------------------------------------------------------------------

def DFT2D(IM):
    '''Implementation de la DFT2D sur l'image IM post traitée avec Rank ou TV '''
    spectre = np.fft.fft2(IM)
    return spectre

def IDFT2D(S):
    '''Implementation de la IDFT2D sur le spectre S'''
    image = np.fft.ifft2(S)
    return image 


#-----------------------------------------------------------------------------------------------
### Interpolating Filter
#-----------------------------------------------------------------------------------------------

def c_nearest(n, M):
    """Plus proche voisin """
    return (1.0 / M) * np.sinc(n / M)          
 
def c_linear(n, M):
    """Linéaire (bilinéaire) """
    return (1.0 / M) * np.sinc(n / M) ** 2
 
def c_cubic(n, M):
    """Cubique """
    s  = np.sinc(n / M)
    s2 = np.sinc(2 * n / M)
    return (1.0 / M) * (3 * s**4 - 2 * s**2 * s2)
 
def c_lanczos(n, M, a=3):
    """
    Lanczos 
    """
    # Intégration numérique sur [-a, a] (support de Lanczos)
    x = np.linspace(-a + 1e-9, a - 1e-9, 10000)
    phi_x = np.sinc(x) * np.sinc(x / a)           
 
    # c_n = (1/M) * sum_x phi(x) * exp(-2pi*i*n*x/M) * dx
    dx = x[1] - x[0]
    n = np.asarray(n)
    phase = np.exp(-2j * np.pi * np.outer(n, x) / M)  
    coeffs = (1.0 / M) * (phase @ phi_x) * dx
    return coeffs.real if n.ndim > 0 else float(coeffs.real)
 
 
FILTERS = {
    "nearest": c_nearest,
    "linear":  c_linear,
    "cubic":   c_cubic,
    "lanczos": c_lanczos,
}
 
 
def phi_nearest(x):
    return np.where(np.abs(x) <= 0.5, 1.0, 0.0)
 
def phi_linear(x):
    ax = np.abs(x)
    return np.where(ax <= 1, 1 - ax, 0.0)
 
def phi_cubic(x):
    ax = np.abs(x)
    out = np.zeros_like(ax)
    m1 = ax <= 1
    m2 = (ax > 1) & (ax <= 2)
    out[m1] = 1.5 * ax[m1]**3 - 2.5 * ax[m1]**2 + 1
    out[m2] = -0.5 * ax[m2]**3 + 2.5 * ax[m2]**2 - 4 * ax[m2] + 2
    return out
 
def phi_lanczos(x, a=3):
    ax = np.abs(x)
    out = np.sinc(x) * np.sinc(x / a)
    out[ax >= a] = 0.0
    return out
 
SPATIAL_FILTERS = {
    "nearest": phi_nearest,
    "linear":  phi_linear,
    "cubic":   phi_cubic,
    "lanczos": phi_lanczos,
}
 
# Correspondance avec les méthodes PIL pour le rééchantillonnage
PIL_RESAMPLE = {
    "nearest": Image.NEAREST,
    "linear":  Image.BILINEAR,
    "cubic":   Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}
 
COLORS = {
    "nearest": "#e74c3c",
    "linear":  "#2ecc71",
    "cubic":   "#3498db",
    "lanczos": "#9b59b6",
}
 
LABELS = {
    "nearest": "Plus proche voisin",
    "linear":  "Linéaire",
    "cubic":   "Cubique",
    "lanczos": "Lanczos",
}
 

#-----------------------------------------------------------------------------------------------
### Calcul des coefs avec le Filtre d'interpolation choisi
#-----------------------------------------------------------------------------------------------

def Calc_coefs_FI(filter_name, M1, M2, N1, N2, z_range=3):
    """Calcul des Coefficients de Fourier avec le filtre d'interpolation"""
    if filter_name not in FILTERS:
        raise ValueError(f"Filtre inconnu: {filter_name}. Choisissez parmi {list(FILTERS.keys())}")
    c_phi = FILTERS[filter_name]
    n1 = np.arange(N1)
    n2 = np.arange(N2)
 
    # Somme sur z (repliements) — le terme z=0 est dominant
    weights1 = np.zeros(N1)
    weights2 = np.zeros(N2)
    for z in range(-z_range, z_range + 1):
        weights1 += c_phi(z * N1 + n1, M1)
        weights2 += c_phi(z * N2 + n2, M2) 
    C = np.outer(weights1, weights2)   # shape (N1, N2)
    return C

def Comp_Modulated_Spectrum(S, filter_name, M1, M2):
    """Calcule le spectre modulé par les coefficients de Fourier du filtre d'interpolation"""
    N1, N2 = S.shape
    C = Calc_coefs_FI(filter_name, M1, M2, N1, N2)
    S_modulated = S * C
    return S_modulated, C

# def apply_interpolating_filter_fourier(S_resampled, filter_name, M1, M2):
#     """
#     Applique les coefficients de Fourier du filtre d'interpolation
#     directement sur le spectre DFT d'une image rééchantillonnée.
 
#     Paramètres
#     ----------
#     S_resampled : np.ndarray (complexe), shape (N1, N2)
#         DFT de l'image rééchantillonnée (taille de sortie).
#     filter_name : str
#         'nearest', 'linear', 'cubic' ou 'lanczos'.
#     M1, M2 : int
#         Taille originale (avant rééchantillonnage).
 
#     Retour
#     ------
#     S_filtered : np.ndarray (complexe), shape (N1, N2)
#         Spectre divisé par les coefficients du filtre (égalisation spectrale).
#     C : np.ndarray (réel), shape (N1, N2)
#         Grille des coefficients c_n(phi).
#     """
#     N1, N2 = S_resampled.shape
#     C = Calc_coefs_FI(filter_name, M1, M2, N1, N2)
 
#     # Évite la division par zéro
#     C_safe = np.where(np.abs(C) < 1e-10, 1e-10, C)
#     S_filtered = S_resampled / C_safe
 
#     return S_filtered, C

#-----------------------------------------------------------------------------------------------
### Test
#-----------------------------------------------------------------------------------------------

if __name__ == "__main__":   
    #-----------------------------------------------------------------------------------------------
    # Test 1 
    #-----------------------------------------------------------------------------------------------

    # Test de la DFT2D et IDFT2D
    img = Image.open("./IMG_InPut/babbon.png").convert("L") #Image sans pretraitement convertie en NB pour test simple
    IM = np.array(img, dtype=np.float32)
    rankIM = Pretraitement.ranktransform(IM)
    S = DFT2D(rankIM)

    # IDFT et garder la partie réelle
    IM_reconstructed = np.real(IDFT2D(S))

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(IM, cmap="gray")
    axes[0].set_title("Image originale (gris)")
    axes[0].axis("off")

    axes[1].imshow(rankIM, cmap="gray")
    axes[1].set_title("Image transformée par rang")
    axes[1].axis("off")

    axes[2].imshow(np.log(np.abs(np.fft.fftshift(S)) + 1), cmap="gray")
    axes[2].set_title("Spectre (log magnitude)")
    axes[2].axis("off")

    axes[3].imshow(np.clip(IM_reconstructed, 0, 255).astype(np.uint8), cmap="gray")
    axes[3].set_title("Image reconstruite")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    #-----------------------------------------------------------------------------------------------
    # Test 2
    #-----------------------------------------------------------------------------------------------
    
    # --- 1. Chargement image ---
    img = Image.open("./IMG_InPut/babbon.png").convert("L")
    IM = np.array(img, dtype=np.float64)
    M1, M2 = IM.shape  # taille originale (ex: 512×512)
 
    # --- 2. Rééchantillonnage avec chaque filtre (downscale 80%) ---
    scale = 0.8
    N1, N2 = int(M1 * scale), int(M2 * scale)
 
    filter_names = list(FILTERS.keys())   # ["nearest", "linear", "cubic", "lanczos"]
    n_filters = len(filter_names)
 
    resampled = {}
    spectra   = {}
    coeffs_2d = {}
    spectra_eq = {}
 
    for name in filter_names:
        img_r = img.resize((N2, N1), PIL_RESAMPLE[name])
        IM_r  = np.array(img_r, dtype=np.float64)
        S_r   = DFT2D(IM_r)
        S_eq, C = Comp_Modulated_Spectrum(S_r, name, M1, M2)
        resampled[name]   = IM_r
        spectra[name]     = S_r
        coeffs_2d[name]   = C
        spectra_eq[name]  = S_eq
 
    # -----------------------------------------------------------------------
    # Figure 1 : φ(x) spatial  +  c_n(φ) Fourier  (reproduction Fig. 3 papier)
    # -----------------------------------------------------------------------
    x_vals = np.linspace(-4, 4, 2000)
    n_vals = np.arange(-2 * M1, 2 * M1 + 1)
 
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Filtres d'interpolation φ — domaine spatial et fréquentiel\n(reproduction Fig. 3 du papier)", fontsize=13)
 
    # --- φ(x) ---
    ax = axes1[0]
    for name in filter_names:
        y = SPATIAL_FILTERS[name](x_vals)
        ax.plot(x_vals, y, label=LABELS[name], color=COLORS[name], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("φ(x) — forme spatiale")
    ax.set_xlabel("x")
    ax.set_ylabel("φ(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-0.3, 1.15)
 
    # --- c_n(φ) ---
    ax = axes1[1]
    n_plot = np.linspace(-2 * M1, 2 * M1, 3000)
    for name in filter_names:
        if name == "lanczos":
            c_vals = c_lanczos(n_plot.astype(int), M1)
        else:
            c_vals = FILTERS[name](n_plot, M1) * M1   # ×M pour normaliser à 1 en n=0
        ax.plot(n_plot, c_vals, label=LABELS[name], color=COLORS[name], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    # Repères à ±M et ±2M
    for k in [-2, -1, 1, 2]:
        ax.axvline(k * M1, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(k * M1, 1.05, f"{k}M", ha="center", fontsize=8, color="gray")
    ax.set_title("c_n(φ)·M — coefficients de Fourier normalisés")
    ax.set_xlabel("n")
    ax.set_ylabel("c_n(φ)·M")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 1.15)
 
    plt.tight_layout()
 
    # -----------------------------------------------------------------------
    # Figure 2 : pour chaque filtre — image rééchantillonnée + spectre + égalisé
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(3, n_filters, figsize=(5 * n_filters, 13))
    fig2.suptitle(f"Comparaison des 4 filtres d'interpolation\n"
                  f"Image originale {M1}×{M2} → {N1}×{N2} (facteur {scale})",
                  fontsize=13)
 
    row_titles = [
        "Image rééchantillonnée",
        "Spectre DFT (log magnitude)",
        "Spectre après égalisation par φ",
    ]
 
    for col, name in enumerate(filter_names):
        # Ligne 0 : image rééchantillonnée
        axes2[0, col].imshow(resampled[name], cmap="gray")
        axes2[0, col].set_title(f"{LABELS[name]}", fontsize=12, fontweight="bold",
                                 color=COLORS[name])
        axes2[0, col].axis("off")
 
        # Ligne 1 : spectre DFT
        S_shift = np.fft.fftshift(spectra[name])
        axes2[1, col].imshow(np.log(np.abs(S_shift) + 1), cmap="inferno")
        axes2[1, col].axis("off")
 
        # Ligne 2 : spectre après égalisation
        S_eq_shift = np.fft.fftshift(spectra_eq[name])
        axes2[2, col].imshow(np.log(np.abs(S_eq_shift) + 1), cmap="inferno")
        axes2[2, col].axis("off")
 
    # Titres de lignes à gauche
    for row, title in enumerate(row_titles):
        axes2[row, 0].set_ylabel(title, fontsize=10, labelpad=8)
        axes2[row, 0].axis("on")
        axes2[row, 0].tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)
        for spine in axes2[row, 0].spines.values():
            spine.set_visible(False)
 
    plt.tight_layout()
 
    # -----------------------------------------------------------------------
    # Figure 3 : coefficients 2D c_n(φ) pour chaque filtre
    # -----------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, n_filters, figsize=(5 * n_filters, 5))
    fig3.suptitle("Grille 2D des coefficients de Fourier c_n(φ) par filtre", fontsize=13)
 
    for col, name in enumerate(filter_names):
        im = axes3[col].imshow(coeffs_2d[name], cmap="viridis", aspect="auto")
        axes3[col].set_title(f"{LABELS[name]}", fontsize=11,
                              fontweight="bold", color=COLORS[name])
        axes3[col].axis("off")
        plt.colorbar(im, ax=axes3[col], fraction=0.046, pad=0.04)
 
    plt.tight_layout()
 
    plt.show()
    
    