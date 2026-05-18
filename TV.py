import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.color as color


def tv_chambolle(f, lambda_tv=10.0, n_iter=100, tau=0.25):
    
    # Initialisation de la variable duale p = (p_x, p_y)
    # p a deux composantes (horizontale et verticale) pour chaque pixel de l'image
    p = np.zeros((2, f.shape[0], f.shape[1]))
    
    for i in range(n_iter):
        # 1. Calcul de la divergence de p (différences finies arrière)
        # div(p) = d(p_x)/dx + d(p_y)/dy
        div_p = np.zeros_like(f)
        
        # Composante X (verticale dans le tableau numpy)
        div_p[:-1, :] += p[0, :-1, :]  # + p_{i,j}
        div_p[1:, :]  -= p[0, :-1, :]  # - p_{i-1,j}
        
        # Composante Y (horizontale dans le tableau numpy)
        div_p[:, :-1] += p[1, :, :-1]
        div_p[:, 1:]  -= p[1, :, :-1]
        
        # 2. Terme d'erreur : divergence(p) - f / lambda
        # C'est ce terme qu'on cherche à minimiser dans le domaine dual
        erreur = div_p - (f / lambda_tv)
        
        # 3. Calcul du gradient de cette erreur (différences finies avant)
        # Attention : le gradient doit être l'opérateur adjoint exact de la divergence
        grad = np.zeros_like(p)
        grad[0, :-1, :] = erreur[1:, :] - erreur[:-1, :]
        grad[1, :, :-1] = erreur[:, 1:] - erreur[:, :-1]
        
        # 4. Mise à jour de la variable duale p
        # On calcule l'amplitude du gradient pour la projection
        magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
        
        # Formule de mise à jour (Eq. 7 du papier de Chambolle)
        p = (p + tau * grad) / (1.0 + tau * magnitude)
        
    # Fin de la boucle : l'algorithme a convergé
    # 5. On recalcule la divergence finale
    div_p = np.zeros_like(f)
    div_p[:-1, :] += p[0, :-1, :]
    div_p[1:, :]  -= p[0, :-1, :]
    div_p[:, :-1] += p[1, :, :-1]
    div_p[:, 1:]  -= p[1, :, :-1]
    
    # 6. L'image lissée (u) se déduit de la variable duale
    u = f - lambda_tv * div_p
    
    # Le résidu (v) est extrait par soustraction
    v = f - u
    
    return u, v

if __name__ == "__main__":
    im_obs2=io.imread('babbon.png')
    I = color.rgb2gray(im_obs2)
    I = I*255/I.max()
    u, v = tv_chambolle(I, lambda_tv=1, n_iter=500)
    # plt.imsave("residu.png", v, cmap='gray')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(u, cmap="gray"), plt.title("Image lissée (u)")
    plt.subplot(1, 3, 2)
    plt.imshow(v, cmap="gray"), plt.title("Résidu (v)")
    pre = io.imread("img_preproc.png")
    plt.subplot(1, 3, 3)
    plt.imshow(pre, cmap="gray"), plt.title("Image prétraitée")
    plt.show()