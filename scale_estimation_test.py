"""Scale estimation utilities for resampling analysis.

This file combines three evidence sources:
1. NFA peaks extracted from spectral correlations.
2. Residual spectrum comparison.
3. Brute-force simulation of several interpolation filters.

The main entry point is `estimate_most_probable_resampling_factor(...)`.
Backward-compatible wrappers are kept for the notebook:
`compute_NFA_axis`, `CrossVal`, `estimate_scale_from_nfa`,
`estimate_scale_logpolar`, and `brute_force_interpolator_scale`.
"""

import math
import os
import sys

import cv2
import numpy as np
from scipy.stats import binom


def _vendor_src_root():
    """Return the path to vendor/GIT_PROF so `import src.ird` works."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "GIT_PROF",
    )


def _import_vendor_ird():
    """Import the vendor resampling detector lazily.

    The vendor code expects to be imported as `src.ird`, so we add the parent
    of `src` to `sys.path` before importing.
    """
    vendor_root = _vendor_src_root()
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    from src.ird import detect_resampling, detect_resampling_with_cross_val

    return detect_resampling, detect_resampling_with_cross_val


def vendor_nfa_from_residual(
    residual_image,
    preproc="none",
    window_ratio=0.1,
    nb_neighbor=1,
    is_jpeg=False,
    is_demosaic=False,
    preproc_param=None,
):
    """Compute NFA using the vendor implementation from `src.ird`.

    This is the preferred path for NFA estimation going forward.
    It returns the cross-validated NFA curve computed from the residual image.
    """
    _, detect_resampling_with_cross_val = _import_vendor_ird()
    residual_np = _to_numpy(residual_image)
    nfa = detect_resampling_with_cross_val(
        residual_np,
        preproc=preproc,
        window_ratio=window_ratio,
        nb_neighbor=nb_neighbor,
        is_jpeg=is_jpeg,
        is_demosaic=is_demosaic,
        preproc_param=preproc_param,
        return_preproc_img=False,
    )
    return nfa


def _to_numpy(img):
    """Convert torch tensors or array-like inputs to float32 numpy arrays."""
    try:
        import torch

        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)
    except Exception:
        arr = np.asarray(img)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            return cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
        return arr[:, :, 0].astype(np.float32)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _normalize(a):
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-8)


def _log_magnitude_spectrum(img):
    """Return a normalized log-magnitude FFT spectrum."""
    image = _to_numpy(img)
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    log_mag = np.log(mag + 1e-8)
    return _normalize(log_mag).astype(np.float32)


def _radial_profile(spectrum):
    """Compute a radial mean profile of a centered spectrum."""
    spec = _to_numpy(spectrum)
    height, width = spec.shape
    cy, cx = height // 2, width // 2
    y, x = np.indices((height, width))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max() + 1
    sums = np.bincount(r.ravel(), weights=spec.ravel(), minlength=max_r)
    counts = np.bincount(r.ravel(), minlength=max_r)
    return sums / (counts + 1e-8)


def _log_polar(img, center=None, M=None, dsize=(256, 256)):
    """Log-polar transform wrapper with a fixed output size."""
    image = _to_numpy(img)
    if center is None:
        center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    if M is None:
        max_radius = math.hypot(center[0], center[1])
        M = dsize[0] / math.log(max_radius + 1e-8)
    log_polar = cv2.logPolar(image, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    return cv2.resize(log_polar, dsize, interpolation=cv2.INTER_LINEAR).astype(np.float32), M


def _phase_corr_score(a, b):
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    (_, _), response = cv2.phaseCorrelate(a32, b32)
    return float(response)


def _candidate_interpolators(interps=None):
    if interps is None:
        return {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
    return interps


def _simulate_resampling(original, scale_factor, interpolation):
    """Down/up sample the image with one interpolation method."""
    original_np = _to_numpy(original)
    height, width = original_np.shape[:2]
    target_w = max(1, int(round(width * scale_factor)))
    target_h = max(1, int(round(height * scale_factor)))
    down = cv2.resize(original_np, (target_w, target_h), interpolation=interpolation)
    return cv2.resize(down, (width, height), interpolation=interpolation)


def summarize_nfa_peaks(NFA_valid, image_width, threshold=1.0, min_distance=1):
    """Turn NFA peaks into scale candidates and a confidence signal."""
    NFA_valid = np.asarray(NFA_valid, dtype=np.float32)
    detected_d = np.where(NFA_valid < threshold)[0]
    detected_d = [int(d) for d in detected_d if d >= min_distance]

    if len(detected_d) == 0:
        return {
            "detected_distances": [],
            "detected_scales": [],
            "best_scale": None,
            "best_scale_nfa": None,
            "confidence": 0.0,
            "message": "No NFA peak detected",
        }

    detected_scales = [float(image_width) / float(d) for d in detected_d if d > 0]
    min_nfa = float(np.min(NFA_valid[detected_d]))
    confidence = 1.0 / (min_nfa + 1e-8)

    return {
        "detected_distances": detected_d,
        "detected_scales": detected_scales,
        "best_scale_nfa": float(detected_scales[0]) if detected_scales else None,
        "confidence": float(confidence),
        "min_nfa": min_nfa,
        "num_peaks": len(detected_scales),
    }



def estimate_scale_logpolar(residual, image=None, candidate_scales=None, dsize=(512, 512)):
    """Log-polar evidence helper kept for notebook compatibility."""
    res = _to_numpy(residual)
    if candidate_scales is None:
        candidate_scales = np.linspace(0.5, 8.0, 101)

    if image is not None:
        src = _to_numpy(image)
    else:
        src = res

    base_spec = _log_magnitude_spectrum(res)
    lp_base, _ = _log_polar(base_spec, dsize=dsize)

    scores = {}
    for s in candidate_scales:
        try:
            scaled_back = _simulate_resampling(src, float(s), cv2.INTER_LINEAR)
        except Exception:
            scaled_back = src
        spec_scaled = _log_magnitude_spectrum(scaled_back)
        lp_scaled, _ = _log_polar(spec_scaled, dsize=dsize)
        scores[float(s)] = _phase_corr_score(lp_base, lp_scaled)

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, scores

def estimate_most_probable_resampling_factor(image, residual, residual_spectrum, NFA_valid, image_width, threshold, candidate_scales, dsize):
    """Estimate the most probable resampling factor using multiple evidence sources.

    This function combines NFA peaks, residual spectrum comparison, and brute-force
    simulation of several interpolation filters to estimate the most probable
    resampling factor applied to the input image.

    Args:
        image: Input image (numpy array or torch tensor).
        residual: Residual image (numpy array or torch tensor).
        residual_spectrum: Precomputed log-magnitude spectrum of the residual.
        NFA_valid: NFA curve computed from the residual.
        image_width: Width of the input image.
        threshold: Threshold for NFA peak detection.
        candidate_scales: List of candidate scales to evaluate.
        dsize: Size for log-polar transformation.

    Returns:
        best_scale: The estimated most probable resampling factor.
        info: Dictionary containing additional information about the estimation process.
    """
    # 1. Summarize NFA peaks
    nfa_summary = summarize_nfa_peaks(NFA_valid, image_width, threshold=threshold)
    detected_scales = nfa_summary.get("detected_scales", [])
    
    # 2. Log-polar evidence
    logpolar_scale, logpolar_scores = estimate_scale_logpolar(residual, image=image, candidate_scales=candidate_scales, dsize=dsize)
    
    # 3. Combine evidence
    all_scales = set(detected_scales + [logpolar_scale])
    best_scale = None
    best_score = -np.inf

    for scale in all_scales:
        if scale < 0.5 or scale > 8.0:
            continue
        
        # Simulate resampling and compute score
        simulated_image = _simulate_resampling(image, scale, cv2.INTER_LINEAR)
        simulated_residual = simulated_image - _simulate_resampling(simulated_image, 1/scale, cv2.INTER_LINEAR)
        score = _phase_corr_score(residual_spectrum, _log_magnitude_spectrum(simulated_residual))
        
        if score > best_score:
            best_score = score
            best_scale = scale

    info = {
        "nfa_summary": nfa_summary,
        "logpolar_scale": logpolar_scale,
        "logpolar_scores": logpolar_scores,
        "best_score": best_score,
    }

    return best_scale, info

def find_most_probable_distance_with_interp(nfa_curve, image, residual, interpolator_name, threshold=1.0):
    # 1. Utiliser une taille fixe pour les calculs (Stabilité mémoire)
    TARGET_SIZE = 512
    image_np = _to_numpy(image)
    residual_np = _to_numpy(residual)
    
    # 2. Ne calculer la NFA que sur les pics identifiés
    summary = summarize_nfa_peaks(nfa_curve, image_np.shape[1], threshold=threshold)
    detected_distances = summary.get("detected_distances", [])
    
    # Générer une liste de candidats très restreinte
    candidate_scales = [0.5, 1.0, 2.0, 4.0, 8.0] 
    for d in detected_distances:
        if 2 <= d < image_np.shape[1]:
            candidate_scales.append(image_np.shape[1] / d) # Upsample
            candidate_scales.append(d / (d - 1))          # Downsample
    
    # 3. Pré-calculer la base spectrale une seule fois
    base_spec = _log_magnitude_spectrum(cv2.resize(residual_np, (TARGET_SIZE, TARGET_SIZE)))
    lp_base, _ = _log_polar(base_spec, dsize=(TARGET_SIZE, TARGET_SIZE))

    best_scale = 1.0
    best_score = -1e9
    interp_flag = _candidate_interpolators().get(interpolator_name, cv2.INTER_CUBIC)

    # 4. Boucle optimisée
    for s in candidate_scales:
        if s < 0.5 or s > 4.0: continue # Plage réaliste
        
        # Simulation sur petite taille
        recon = cv2.resize(image_np, (TARGET_SIZE, TARGET_SIZE), interpolation=interp_flag)
        recon_scaled = cv2.resize(recon, (0,0), fx=1/s, fy=1/s, interpolation=interp_flag)
        recon_back = cv2.resize(recon_scaled, (TARGET_SIZE, TARGET_SIZE), interpolation=interp_flag)
        
        sim_res = recon - recon_back
        sim_lp, _ = _log_polar(_log_magnitude_spectrum(sim_res), dsize=(TARGET_SIZE, TARGET_SIZE))
        
        score = _phase_corr_score(lp_base, sim_lp)
        
        if score > best_score:
            best_score = score
            best_scale = s
            
    return best_scale if abs(best_scale - 1.0) > 0.02 else None

if __name__ == "__main__":
    print("scale_estimation_test.py loaded.")
    print("Import estimate_most_probable_resampling_factor(...) in your notebook.")
