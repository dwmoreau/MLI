import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm  import tqdm
import json
import ast
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from cctbx import crystal
from cctbx import uctbx
from cctbx.sgtbx.lattice_symmetry import metric_subgroups


def get_symmetry(unit_cell):
    unit_cell[3:] *= 180/np.pi
    input_symmetry = crystal.symmetry(
        unit_cell=uctbx.unit_cell(parameters=list(unit_cell)),
        space_group_symbol="P1"
    )
    # Do it
    groups = metric_subgroups(
        input_symmetry,
        0.1,
        enforce_max_delta_for_generated_two_folds=True
    )
    sg = groups.result_groups[0]['best_subsym'].space_group()
    unit_cell = groups.result_groups[0]['best_subsym'].unit_cell().parameters()
    crystal_system = sg.crystal_system()
    unit_cell = np.array(unit_cell)
    unit_cell[3:] *= np.pi/180
    return unit_cell, crystal_system


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548
MAD_TO_SIGMA = 1.4826


def _params_per_peak(profile):
    return 4 if profile == "pseudo_voigt" else 3


def compute_fwhm(sigma, eta=None, profile="pseudo_voigt"):
    """FWHM from sigma (and eta for pseudo-Voigt)."""
    if profile == "pseudo_voigt" and eta is not None:
        eta = np.clip(eta, 0, 1)
        return eta * 2.0 * sigma + (1 - eta) * FWHM_GAUSS_FACTOR * sigma
    return FWHM_GAUSS_FACTOR * sigma


def sigma_from_fwhm(fwhm, eta=None, profile="pseudo_voigt"):
    """Inverse of compute_fwhm."""
    if profile == "pseudo_voigt" and eta is not None:
        eta = np.clip(eta, 0, 1)
        factor = eta * 2.0 + (1 - eta) * FWHM_GAUSS_FACTOR
    else:
        factor = FWHM_GAUSS_FACTOR
    return max(fwhm / factor, 1e-10)


def estimate_noise(y, y_smooth):
    """Robust noise estimate from the residual between data and smooth."""
    residual = y - y_smooth
    baseline_mask = y_smooth <= np.median(y_smooth)

    if np.sum(baseline_mask) > 20:
        noise = np.median(np.abs(residual[baseline_mask])) * MAD_TO_SIGMA
    else:
        noise = np.median(np.abs(residual)) * MAD_TO_SIGMA

    if noise < 1e-10:
        noise = np.std(residual) if np.std(residual) > 0 else 1e-10
    return noise


def _auto_smooth_window(n):
    """Pick an odd Savitzky-Golay window from data length."""
    w = max(5, n // 100)
    if w % 2 == 0:
        w += 1
    return w


def evaluate_peaks(x, peaks, profile="pseudo_voigt"):
    """Sum of all peak profiles from a list of peak dicts."""
    y = np.zeros_like(x, dtype=float)
    for pk in peaks:
        prof = pk.get("profile", profile)
        if prof == "pseudo_voigt":
            y += pseudo_voigt(x, pk["amplitude"], pk["center"],
                              pk["sigma"], pk.get("eta", 0.5))
        else:
            y += gaussian(x, pk["amplitude"], pk["center"], pk["sigma"])
    return y


def _rss(x, y, peaks, profile="pseudo_voigt"):
    """Residual sum of squares."""
    return float(np.sum((y - evaluate_peaks(x, peaks, profile)) ** 2))


def _half_width_pts(y, idx):
    """Estimate half-width at half-max in array index units."""
    idx = np.clip(idx, 0, len(y) - 1)
    hm = y[idx] * 0.5
    l, r = idx, idx
    while l > 0 and y[l] > hm:
        l -= 1
    while r < len(y) - 1 and y[r] > hm:
        r += 1
    return max(r - l, 4)


def _safe_curve_fit(model_fn, x, y, p0, bounds, maxfev=10000):
    """Wrapper around curve_fit that returns (popt, perr, success)."""
    try:
        popt, pcov = curve_fit(model_fn, x, y, p0=p0,
                               bounds=bounds, maxfev=maxfev, method="trf")
        perr = np.sqrt(np.maximum(np.diag(pcov), 0))
        return popt, perr, True
    except (RuntimeError, ValueError):
        return np.array(p0), np.full(len(p0), np.nan), False


def _parse_fitted_peaks(popt, perr, peak_indices, profile, x_fit,
                        noise_level, min_fwhm, max_fwhm, n_original=None):
    """
    Turn flat optimizer output into a list of peak dicts,
    applying post-fit rejection (amplitude, FWHM bounds).
    """
    npp = _params_per_peak(profile)
    n_peaks = len(popt) // npp
    if n_original is None:
        n_original = n_peaks

    results = []
    for i in range(n_peaks):
        off = i * npp
        amp = popt[off]
        cen = popt[off + 1]
        sigma = abs(popt[off + 2])
        cen_err = perr[off + 1] if not np.isnan(perr[off + 1]) else None
        eta = np.clip(popt[off + 3], 0, 1) if profile == "pseudo_voigt" else None

        fwhm = compute_fwhm(sigma, eta, profile)

        if amp < noise_level * 2:
            continue
        if fwhm < min_fwhm or fwhm > max_fwhm:
            continue

        pk = {
            "center": float(cen),
            "center_err": float(cen_err) if cen_err is not None else None,
            "amplitude": float(amp),
            "sigma": float(sigma),
            "fwhm": float(fwhm),
            "profile": profile,
            "fit_window": (float(x_fit[0]), float(x_fit[-1])),
            "was_added": i >= n_original,
        }
        if profile == "pseudo_voigt":
            pk["eta"] = float(eta)
        results.append(pk)

    return results


def _find_residual_peaks(residual, noise_level, threshold_snr,
                         min_width_pts=3, min_prominence_frac=0.5):
    """find_peaks on a smoothed residual; returns indices or empty array."""
    n = len(residual)
    win = _auto_smooth_window(n)
    win = min(win, n - 1)
    if win >= 5 and n > win:
        smooth = savgol_filter(residual, win, polyorder=2)
    else:
        smooth = residual.copy()
    smooth = np.maximum(smooth, 0)

    height = threshold_snr * noise_level
    try:
        peaks, _ = find_peaks(
            smooth, height=height,
            width=min_width_pts,
            prominence=height * min_prominence_frac,
        )
    except Exception:
        peaks = np.array([], dtype=int)
    return peaks, smooth


# ═══════════════════════════════════════════════════════════════════
# Peak profile functions
# ═══════════════════════════════════════════════════════════════════

def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def pseudo_voigt(x, amplitude, center, sigma, eta):
    """Pseudo-Voigt: linear mix of Gaussian and Lorentzian."""
    eta = np.clip(eta, 0, 1)
    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    l = 1.0 / (1.0 + ((x - center) / sigma) ** 2)
    return amplitude * (eta * l + (1 - eta) * g)


def multi_peak_model(x, *params, profile="pseudo_voigt"):
    """Sum of peak profiles from flat parameter vector."""
    y = np.zeros_like(x, dtype=float)
    npp = _params_per_peak(profile)
    for i in range(len(params) // npp):
        p = params[i * npp:(i + 1) * npp]
        if profile == "pseudo_voigt":
            y += pseudo_voigt(x, *p)
        else:
            y += gaussian(x, *p)
    return y


# ═══════════════════════════════════════════════════════════════════
# Background estimation (SNIP)
# ═══════════════════════════════════════════════════════════════════

def estimate_background(x, y, width=None, n_iterations=40):
    """
    SNIP background estimation in LLS space.

    Parameters
    ----------
    x : array  —  x-axis (used only for length).
    y : array  —  Intensity values.
    width : int or None  —  Max clipping window. None = ~5% of length.
    n_iterations : int  —  Clipping iterations (clamped to width).

    Returns
    -------
    bg : ndarray  —  Estimated background.
    """
    n = len(y)
    if width is None:
        width = max(n // 20, 10)
    n_iterations = min(n_iterations, width)

    # LLS transform
    y_pos = np.maximum(y, 0.0)
    z = np.log(np.sqrt(y_pos + 1.0) + 1.0)

    # Decreasing-window SNIP clipping
    for p in range(n_iterations, 0, -1):
        z_new = z.copy()
        avg = (z[:-2 * p] + z[2 * p:]) / 2.0
        z_new[p:n - p] = np.minimum(z[p:n - p], avg)
        z = z_new

    # Inverse LLS
    bg = (np.exp(z) - 1.0) ** 2 - 1.0
    bg = np.maximum(bg, 0.0)

    # Light smoothing
    if n > 50:
        win = min(n // 25, 51)
        if win % 2 == 0:
            win += 1
        if win >= 5:
            bg = savgol_filter(bg, window_length=win, polyorder=2)
            bg = np.maximum(bg, 0.0)

    return bg


# ═══════════════════════════════════════════════════════════════════
# Candidate peak detection
# ═══════════════════════════════════════════════════════════════════

def find_candidate_peaks(x, y_bg_sub, snr_threshold=3.0,
                         smooth_window=None, min_relative_height=0.005):
    """
    Smoothed-derivative peak detection.

    Returns
    -------
    peaks : ndarray of int  —  Candidate indices.
    y_smooth : ndarray      —  Smoothed data.
    noise_level : float     —  Estimated noise σ.
    """
    n = len(y_bg_sub)
    if smooth_window is None:
        smooth_window = _auto_smooth_window(n)

    y_smooth = savgol_filter(y_bg_sub, window_length=smooth_window, polyorder=3)
    y_smooth = np.maximum(y_smooth, 0)

    noise_level = estimate_noise(y_bg_sub, y_smooth)

    abs_height = snr_threshold * noise_level
    rel_height = min_relative_height * np.max(y_smooth)
    min_height = max(abs_height, rel_height)

    peaks, _ = find_peaks(y_smooth, height=min_height, width=4,
                          prominence=min_height)
    return peaks, y_smooth, noise_level


# ═══════════════════════════════════════════════════════════════════
# Peak clustering
# ═══════════════════════════════════════════════════════════════════

def cluster_peaks(x, peak_indices, y_smooth, merge_distance_factor=2.0):
    """Group nearby peaks for simultaneous fitting."""
    if len(peak_indices) == 0:
        return []

    clusters = []
    current = [peak_indices[0]]

    for i in range(1, len(peak_indices)):
        prev_idx = peak_indices[i - 1]
        curr_idx = peak_indices[i]
        est_fwhm_pts = max(_half_width_pts(y_smooth, prev_idx), 3)

        if (curr_idx - prev_idx) < merge_distance_factor * est_fwhm_pts:
            current.append(curr_idx)
        else:
            clusters.append(current)
            current = [curr_idx]

    clusters.append(current)
    return clusters


# ═══════════════════════════════════════════════════════════════════
# Cluster fitting (with iterative residual peak addition)
# ═══════════════════════════════════════════════════════════════════

def fit_peak_cluster(x, y, peak_indices, y_smooth, profile="pseudo_voigt",
                     padding_factor=2.0, noise_level=None,
                     max_extra_peaks=3, residual_snr=5.0,
                     min_fwhm=None, max_fwhm=0.005,
                     improvement_threshold=0.02):
    """
    Fit one cluster of peaks, iteratively adding from residuals.

    Returns list of peak dicts.
    """
    if len(peak_indices) == 0:
        return []

    npp = _params_per_peak(profile)
    avg_dx = np.median(np.diff(x))
    if min_fwhm is None:
        min_fwhm = 2.0 * avg_dx

    # Physics constraints
    min_separation_fwhm = 0.75
    width_tolerance = 1.1
    min_amplitude_frac = 0.1

    # ── Sigma bounds from FWHM limits ──
    max_sigma = max_fwhm / FWHM_GAUSS_FACTOR
    min_sigma = min_fwhm / FWHM_GAUSS_FACTOR * 0.5

    # ── Reference sigma from candidates ──
    ref_sigmas = []
    for idx in peak_indices:
        hw = _half_width_pts(y_smooth, idx)
        ref_sigmas.append(max(float(hw * avg_dx * 0.5), avg_dx))
    ref_sigma = np.clip(np.median(ref_sigmas), min_sigma, max_sigma)

    # ── Fit window ──
    leftmost, rightmost = min(peak_indices), max(peak_indices)
    left_hw = _half_width_pts(y_smooth, leftmost)
    right_hw = _half_width_pts(y_smooth, rightmost)
    i_lo = max(0, int(leftmost - padding_factor * left_hw))
    i_hi = min(len(x) - 1, int(rightmost + padding_factor * right_hw))
    x_fit = x[i_lo:i_hi + 1]
    y_fit = y[i_lo:i_hi + 1]

    if len(x_fit) < 4:
        return []

    # ── Local noise ──
    if noise_level is None:
        noise_level = estimate_noise(y[i_lo:i_hi + 1], y_smooth[i_lo:i_hi + 1])

    # ── Helpers for building guess / bounds ──
    def build_p0_bounds(indices, ref_sig):
        p0, lo, hi = [], [], []
        sigma_lo = max(ref_sig / width_tolerance, min_sigma)
        sigma_hi = min(ref_sig * width_tolerance, max_sigma)

        for idx in indices:
            idx_c = np.clip(idx, 0, len(y_smooth) - 1)
            amp0 = float(max(y_smooth[idx_c], noise_level))
            cen0 = float(x[idx_c])
            sigma0 = np.clip(float(_half_width_pts(y_smooth, idx_c) * avg_dx * 0.5),
                             sigma_lo, sigma_hi)

            p0.extend([amp0, cen0, sigma0])
            lo.extend([0, cen0 - 3 * ref_sig, sigma_lo])
            hi.extend([amp0 * 5, cen0 + 3 * ref_sig, sigma_hi])

            if profile == "pseudo_voigt":
                p0.append(0.5)
                lo.append(0.0)
                hi.append(1.0)
        return p0, lo, hi

    def do_fit(indices, ref_sig):
        p0, lo, hi = build_p0_bounds(indices, ref_sig)
        model_fn = lambda xx, *params: multi_peak_model(xx, *params, profile=profile)
        return _safe_curve_fit(model_fn, x_fit, y_fit, p0, (lo, hi))

    def compute_rss(popt):
        return float(np.sum((y_fit - multi_peak_model(x_fit, *popt, profile=profile)) ** 2))

    def fitted_centers_fwhms(popt):
        centers, fwhms = [], []
        for i in range(len(popt) // npp):
            off = i * npp
            eta = np.clip(popt[off + 3], 0, 1) if profile == "pseudo_voigt" else None
            centers.append(popt[off + 1])
            fwhms.append(compute_fwhm(abs(popt[off + 2]), eta, profile))
        return np.array(centers), np.array(fwhms)

    # ── Iterative fit loop ──
    active_indices = list(peak_indices)
    n_original = len(peak_indices)
    popt, perr, fit_ok = None, None, False
    prev_rss = None

    for iteration in range(max_extra_peaks + 1):
        popt, perr, fit_ok = do_fit(active_indices, ref_sigma)
        current_rss = compute_rss(popt)

        # Check improvement from last addition
        if iteration > 0 and prev_rss is not None:
            improvement = (prev_rss - current_rss) / prev_rss
            if improvement < improvement_threshold:
                active_indices.pop()
                popt, perr, fit_ok = do_fit(active_indices, ref_sigma)
                break

        prev_rss = current_rss

        # Update ref_sigma from fit
        n_current = len(active_indices)
        fitted_sigmas = [abs(popt[i * npp + 2]) for i in range(n_current)]
        ref_sigma = np.clip(np.median(fitted_sigmas), min_sigma, max_sigma)

        if iteration >= max_extra_peaks:
            break

        # Search residual for missing peaks
        y_model = multi_peak_model(x_fit, *popt, profile=profile)
        residual = y_fit - y_model
        resid_peaks, resid_smooth = _find_residual_peaks(
            residual, noise_level, residual_snr)

        if len(resid_peaks) == 0:
            break

        centers, fwhms = fitted_centers_fwhms(popt)
        max_amplitude = max(popt[i * npp] for i in range(n_current))

        # Try strongest residual peak
        order = np.argsort(resid_smooth[resid_peaks])[::-1]
        added = False
        for ri in order:
            rp = resid_peaks[ri]
            new_x = x_fit[rp]
            new_amp = resid_smooth[rp]

            if new_amp < min_amplitude_frac * max_amplitude:
                continue
            if np.any(np.abs(centers - new_x) < min_separation_fwhm * fwhms):
                continue

            # Expand window if near edge
            edge_frac = 0.05
            x_range = x_fit[-1] - x_fit[0]
            if new_x < x_fit[0] + edge_frac * x_range or \
               new_x > x_fit[-1] - edge_frac * x_range:
                new_idx_full = i_lo + rp
                expand = int(3 * ref_sigma / avg_dx)
                i_lo_new = max(0, min(i_lo, new_idx_full - expand))
                i_hi_new = min(len(x) - 1, max(i_hi, new_idx_full + expand))
                if i_lo_new != i_lo or i_hi_new != i_hi:
                    i_lo, i_hi = i_lo_new, i_hi_new
                    x_fit = x[i_lo:i_hi + 1]
                    y_fit = y[i_lo:i_hi + 1]
                    prev_rss = compute_rss(popt)

            new_idx_full = np.argmin(np.abs(x - new_x))
            active_indices.append(new_idx_full)
            active_indices.sort()
            added = True
            break

        if not added:
            break

    if popt is None:
        return []

    return _parse_fitted_peaks(popt, perr, active_indices, profile, x_fit,
                               noise_level, min_fwhm, max_fwhm, n_original)


# ═══════════════════════════════════════════════════════════════════
# FWHM trend fitting
# ═══════════════════════════════════════════════════════════════════

def fit_fwhm_trend(peaks, outlier_sigma=3.0):
    """
    Robust linear fit of FWHM vs center position.

    Returns dict with slope, intercept, residual_sigma, inlier_mask.
    """
    if len(peaks) < 2:
        fwhm0 = peaks[0]["fwhm"] if peaks else 0.001
        return {
            "slope": 0.0,
            "intercept": fwhm0,
            "residual_sigma": fwhm0 * 0.1 if peaks else 0.001,
            "inlier_mask": np.array([True] * len(peaks), dtype=bool),
        }

    centers = np.array([pk["center"] for pk in peaks])
    fwhms = np.array([pk["fwhm"] for pk in peaks])
    weights = np.ones(len(centers))

    for _ in range(3):
        W = np.diag(weights)
        A = np.column_stack([centers, np.ones(len(centers))])
        try:
            coeffs = np.linalg.lstsq(W @ A, W @ fwhms, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.array([0.0, np.median(fwhms)])

        slope, intercept = coeffs
        residuals = fwhms - (slope * centers + intercept)
        sigma = max(np.median(np.abs(residuals)) * MAD_TO_SIGMA, 1e-10)
        weights = np.where(np.abs(residuals) < outlier_sigma * sigma, 1.0, 0.1)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_sigma": float(sigma),
        "inlier_mask": np.abs(residuals) < outlier_sigma * sigma,
    }


def predicted_fwhm(x_val, trend):
    return trend["slope"] * x_val + trend["intercept"]


def predicted_sigma(x_val, trend, profile="pseudo_voigt", eta=0.5):
    return sigma_from_fwhm(predicted_fwhm(x_val, trend), eta, profile)


# ═══════════════════════════════════════════════════════════════════
# Global pruning
# ═══════════════════════════════════════════════════════════════════

def _prune_hard(peaks, noise_level, min_fwhm=None, max_fwhm=None):
    """Remove peaks violating absolute physical constraints."""
    kept, pruned = [], []
    for pk in peaks:
        reason = None
        if pk["amplitude"] < noise_level * 2:
            reason = "amplitude too low"
        elif min_fwhm is not None and pk["fwhm"] < min_fwhm:
            reason = "FWHM too narrow"
        elif max_fwhm is not None and pk["fwhm"] > max_fwhm:
            reason = "FWHM too broad"

        (pruned if reason else kept).append(pk)
    return kept, pruned


def _prune_trend(peaks, trend, fwhm_outlier_sigma=3.0):
    """Remove peaks whose FWHM deviates too far from the linear trend."""
    kept, pruned = [], []
    for pk in peaks:
        deviation = abs(pk["fwhm"] - predicted_fwhm(pk["center"], trend))
        if deviation > fwhm_outlier_sigma * trend["residual_sigma"]:
            pruned.append(pk)
        else:
            kept.append(pk)
    return kept, pruned


def _prune_leave_one_out(x, y_bg_sub, peaks, profile, min_contribution=0.01):
    """
    Iteratively remove the least-contributing peak until all remaining
    peaks each contribute at least min_contribution fractional RSS.
    """
    if len(peaks) <= 1:
        return peaks, []

    kept = list(peaks)
    pruned = []

    while len(kept) > 1:
        current_rss = _rss(x, y_bg_sub, kept, profile)
        worst_idx, worst_contrib = None, float("inf")

        for i in range(len(kept)):
            reduced = kept[:i] + kept[i + 1:]
            frac = (_rss(x, y_bg_sub, reduced, profile) - current_rss) / current_rss \
                if current_rss > 0 else 0
            if frac < worst_contrib:
                worst_contrib = frac
                worst_idx = i

        if worst_contrib < min_contribution:
            pruned.append(kept.pop(worst_idx))
        else:
            break

    return kept, pruned


# ═══════════════════════════════════════════════════════════════════
# Global refit
# ═══════════════════════════════════════════════════════════════════

def _refit_all_peaks(x, y_bg_sub, peaks, trend, noise_level,
                     profile="pseudo_voigt", min_fwhm=None, max_fwhm=None):
    """Refit all peaks in local groups with trend-constrained widths."""
    if not peaks:
        return peaks

    avg_dx = np.median(np.diff(x))
    npp = _params_per_peak(profile)
    min_sigma_abs = (min_fwhm or avg_dx * 2) / FWHM_GAUSS_FACTOR
    max_sigma_abs = (max_fwhm or 1.0) / 2.0

    # Group by proximity
    sorted_peaks = sorted(peaks, key=lambda p: p["center"])
    groups = [[sorted_peaks[0]]]
    for pk in sorted_peaks[1:]:
        prev = groups[-1][-1]
        if pk["center"] - prev["center"] < 2.0 * max(prev["fwhm"], pk["fwhm"]):
            groups[-1].append(pk)
        else:
            groups.append([pk])

    refined = []
    model_fn = lambda xx, *params: multi_peak_model(xx, *params, profile=profile)

    for group in groups:
        # Fit window
        x_lo = group[0]["center"] - 5 * group[0]["fwhm"]
        x_hi = group[-1]["center"] + 5 * group[-1]["fwhm"]
        mask = (x >= x_lo) & (x <= x_hi)
        x_fit = x[mask]
        y_fit = y_bg_sub[mask]

        if len(x_fit) < max(6, npp * len(group)):
            refined.extend(group)
            continue

        # Build p0 / bounds per peak using trend
        p0, lo, hi = [], [], []
        for pk in group:
            local_sigma = predicted_sigma(pk["center"], trend, profile,
                                          pk.get("eta", 0.5))
            s_lo = max(local_sigma / 2.0, min_sigma_abs)
            s_hi = min(local_sigma * 2.0, max_sigma_abs)

            p0.extend([pk["amplitude"], pk["center"],
                       np.clip(pk["sigma"], s_lo, s_hi)])
            lo.extend([0, pk["center"] - 3 * pk["fwhm"], s_lo])
            hi.extend([pk["amplitude"] * 5, pk["center"] + 3 * pk["fwhm"], s_hi])

            if profile == "pseudo_voigt":
                p0.append(pk.get("eta", 0.5))
                lo.append(0.0)
                hi.append(1.0)

        popt, perr, ok = _safe_curve_fit(model_fn, x_fit, y_fit, p0, (lo, hi))
        if not ok:
            refined.extend(group)
            continue

        parsed = _parse_fitted_peaks(popt, perr, list(range(len(group))),
                                     profile, x_fit, noise_level,
                                     min_fwhm or 0, max_fwhm or 1e10)
        # Carry forward metadata from originals
        for new_pk, old_pk in zip(parsed, group):
            new_pk["was_added"] = old_pk.get("was_added", False)
            new_pk["source"] = old_pk.get("source")
        refined.extend(parsed)

    return refined


# ═══════════════════════════════════════════════════════════════════
# Finding new peaks in global residual
# ═══════════════════════════════════════════════════════════════════

def _fit_single_peak_to_residual(x_local, y_local, cen_x, amp0,
                                 local_sigma, sigma_lo, sigma_hi,
                                 noise_level, profile):
    """Fit one peak to a local residual patch. Returns peak dict or None."""
    model_fn = lambda xx, *params: multi_peak_model(xx, *params, profile=profile)

    if profile == "pseudo_voigt":
        p0 = [amp0, cen_x, local_sigma, 0.5]
        bounds = ([0, cen_x - 3 * local_sigma, sigma_lo, 0],
[amp0 * 5, cen_x + 3 * local_sigma, sigma_hi, 1])
    else:
        p0 = [amp0, cen_x, local_sigma]
        bounds = ([0, cen_x - 3 * local_sigma, sigma_lo],
                  [amp0 * 5, cen_x + 3 * local_sigma, sigma_hi])

    popt, perr, ok = _safe_curve_fit(model_fn, x_local, y_local, p0, bounds,
                                     maxfev=5000)
    if not ok:
        return None

    amp = popt[0]
    cen = popt[1]
    sigma = abs(popt[2])
    cen_err = perr[1] if not np.isnan(perr[1]) else None
    eta = np.clip(popt[3], 0, 1) if profile == "pseudo_voigt" else None
    fwhm = compute_fwhm(sigma, eta, profile)

    if amp < noise_level * 2:
        return None

    pk = {
        "center": float(cen),
        "center_err": float(cen_err) if cen_err is not None else None,
        "amplitude": float(amp),
        "sigma": float(sigma),
        "fwhm": float(fwhm),
        "profile": profile,
        "fit_window": (float(x_local[0]), float(x_local[-1])),
        "was_added": True,
        "source": "global_residual",
    }
    if profile == "pseudo_voigt":
        pk["eta"] = float(eta)
    return pk


def _find_and_validate_new_peaks(x, y_bg_sub, current_peaks, trend,
                                 noise_level, profile="pseudo_voigt",
                                 residual_snr=10.0, min_relative_height=0.02,
                                 min_fwhm=None, max_fwhm=None, max_new=10,
                                 improvement_threshold=0.05):
    """
    Find candidates in the global residual, fit each, keep only those
    that reduce RSS by at least improvement_threshold. Strongest first.
    """
    avg_dx = np.median(np.diff(x))
    min_sigma_abs = (min_fwhm or avg_dx) / FWHM_GAUSS_FACTOR
    max_sigma_abs = (max_fwhm or 1.0) / 2.0

    working_peaks = list(current_peaks)
    added = []

    for _ in range(max_new):
        residual = y_bg_sub - evaluate_peaks(x, working_peaks, profile)
        rss_before = float(np.sum(residual ** 2))

        # Detection threshold
        min_height = residual_snr * noise_level
        if working_peaks:
            max_amp = max(pk["amplitude"] for pk in working_peaks)
            min_height = max(min_height, min_relative_height * max_amp)

        mid_x = 0.5 * (x[0] + x[-1])
        expected_fwhm = max(predicted_fwhm(mid_x, trend), min_fwhm or avg_dx * 2)
        expected_width_pts = max(3, int(expected_fwhm / avg_dx * 0.5))

        resid_peaks, resid_smooth = _find_residual_peaks(
            residual, noise_level, residual_snr,
            min_width_pts=expected_width_pts)

        if len(resid_peaks) == 0:
            break

        existing_centers = np.array([pk["center"] for pk in working_peaks])
        order = np.argsort(resid_smooth[resid_peaks])[::-1]

        added_one = False
        for ri in order:
            idx = resid_peaks[ri]
            cen_x = float(x[idx])

            # Separation check
            local_fwhm = np.clip(predicted_fwhm(cen_x, trend),
                                 min_fwhm or avg_dx, max_fwhm or 1.0)
            if len(existing_centers) > 0 and \
               np.any(np.abs(existing_centers - cen_x) < local_fwhm):
                continue

            # Local fit window
            local_sigma = predicted_sigma(cen_x, trend, profile, eta=0.5)
            sigma_lo = max(local_sigma / 2.0, min_sigma_abs)
            sigma_hi = min(local_sigma * 2.0, max_sigma_abs)

            hw = max(int(local_sigma * 5 / avg_dx), 10)
            i_lo = max(0, idx - hw)
            i_hi = min(len(x) - 1, idx + hw)
            x_local = x[i_lo:i_hi + 1]
            y_local = residual[i_lo:i_hi + 1]

            if len(x_local) < 6:
                continue

            amp0 = float(max(resid_smooth[idx], noise_level))

            candidate = _fit_single_peak_to_residual(
                x_local, y_local, cen_x, amp0,
                local_sigma, sigma_lo, sigma_hi,
                noise_level, profile)

            if candidate is None:
                continue

            # FWHM bounds check
            if candidate["fwhm"] < (min_fwhm or 0) or \
               candidate["fwhm"] > (max_fwhm or candidate["fwhm"] + 1):
                continue

            # RSS improvement test
            y_model_new = evaluate_peaks(x, working_peaks + [candidate], profile)
            rss_after = float(np.sum((y_bg_sub - y_model_new) ** 2))
            improvement = (rss_before - rss_after) / rss_before if rss_before > 0 else 0

            if improvement < improvement_threshold:
                continue

            working_peaks.append(candidate)
            added.append(candidate)
            added_one = True
            break

        if not added_one:
            break

    return added


# ═══════════════════════════════════════════════════════════════════
# Global optimization (prune-only → refit)
# ═══════════════════════════════════════════════════════════════════

def global_optimize_peaks(x, y_bg_sub, all_peaks, noise_level,
                          profile="pseudo_voigt", max_total_peaks=30,
                          min_fwhm=None, max_fwhm=0.005,
                          fwhm_outlier_sigma=3.0,
                          refit_iterations=2, show_diagnostics=False):
    """
    Global post-processing: prune bad peaks and refit survivors.

    No new peaks are added — this stage only removes and refines.

    Returns (refined_peaks, trend, diagnostics).
    """
    avg_dx = np.median(np.diff(x))
    if min_fwhm is None:
        min_fwhm = 2.0 * avg_dx

    refined = [pk.copy() for pk in all_peaks]
    diag_history = []

    for refit_iter in range(refit_iterations):
        # 1. Hard prune (amplitude, FWHM bounds)
        trend = fit_fwhm_trend(refined, fwhm_outlier_sigma)
        refined, pruned_hard = _prune_hard(refined, noise_level,
                                           min_fwhm, max_fwhm)

        # 2. Trend prune (FWHM outliers vs linear trend)
        trend = fit_fwhm_trend(refined, fwhm_outlier_sigma)
        refined, pruned_trend = _prune_trend(refined, trend, fwhm_outlier_sigma)

        # 3. Leave-one-out prune (remove least-contributing peaks)
        refined, pruned_loo = _prune_leave_one_out(
            x, y_bg_sub, refined, profile)

        all_pruned = pruned_hard + pruned_trend + pruned_loo

        # 4. Enforce total peak budget
        if len(refined) > max_total_peaks:
            refined.sort(key=lambda p: p["amplitude"], reverse=True)
            all_pruned.extend(refined[max_total_peaks:])
            refined = refined[:max_total_peaks]

        # 5. Refit survivors with trend-constrained widths
        trend = fit_fwhm_trend(refined, fwhm_outlier_sigma)
        refined = _refit_all_peaks(x, y_bg_sub, refined, trend, noise_level,
                                   profile, min_fwhm, max_fwhm)

        refined.sort(key=lambda p: p["center"])

        diag_history.append({
            "iteration": refit_iter,
            "n_peaks": len(refined),
            "n_pruned_hard": len(pruned_hard),
            "n_pruned_trend": len(pruned_trend),
            "n_pruned_loo": len(pruned_loo),
            "trend": trend.copy(),
        })

        if show_diagnostics:
            _plot_global_diagnostics(x, y_bg_sub, refined, all_pruned,
                                    [], trend, noise_level,
                                    refit_iter, profile)

    trend = fit_fwhm_trend(refined, fwhm_outlier_sigma)
    return refined, trend, {"history": diag_history, "trend": trend}

# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def pick_peaks(x, y, snr_threshold=3.0, min_relative_height=0.005,
               bg_width=None, bg_iterations=40, smooth_window=None,
               profile="pseudo_voigt", merge_distance_factor=2.0,
               padding_factor=2.0, show_diagnostics=True,
               max_total_peaks=30):
    """
    Full peak-picking pipeline for powder diffraction data.

    Returns (peaks, background, diagnostics).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Stage 1: Background
    background = estimate_background(x, y, width=bg_width,
                                     n_iterations=bg_iterations)
    y_bg_sub = np.maximum(y - background, 0)

    # Stage 2: Candidate detection
    cand_indices, y_smooth, noise_level = find_candidate_peaks(
        x, y_bg_sub, snr_threshold=snr_threshold,
        smooth_window=smooth_window,
        min_relative_height=min_relative_height)

    # Stage 3: Clustering
    sorted_idx = cand_indices[np.argsort(x[cand_indices])] \
        if len(cand_indices) > 0 else cand_indices
    clusters = cluster_peaks(x, sorted_idx, y_bg_sub, merge_distance_factor)

    # Stage 4: Fit each cluster
    all_peaks = []
    cluster_fit_results = []

    for cluster in clusters:
        remaining = max_total_peaks - len(all_peaks)
        if remaining <= 0:
            break

        max_extra = min(3, max(0, remaining - len(cluster)))
        fitted = fit_peak_cluster(
            x, y_bg_sub, cluster, y_bg_sub, profile=profile,
            padding_factor=padding_factor, noise_level=noise_level,
            max_extra_peaks=max_extra, residual_snr=5.0,
            min_fwhm=None, max_fwhm=0.005,
            improvement_threshold=0.05)

        fitted = fitted[:max_total_peaks - len(all_peaks)]
        all_peaks.extend(fitted)
        cluster_fit_results.append({
            "cluster_indices": cluster,
            "fitted_peaks": fitted,
        })

    all_peaks.sort(key=lambda p: p["center"])

    # Stage 5: Global optimization
    #refined_peaks, trend, global_diag = global_optimize_peaks(
    #    x, y_bg_sub, all_peaks, noise_level, profile=profile,
    #    max_total_peaks=max_total_peaks, min_fwhm=None, max_fwhm=0.005,
    #    refit_iterations=2, show_diagnostics=show_diagnostics)

    # Diagnostics plotting
    if show_diagnostics:
        #_plot_all_stages(x, y, background, y_bg_sub, y_smooth, noise_level,
        #                 cand_indices, clusters, cluster_fit_results,
        #                 refined_peaks, snr_threshold, profile)
        _plot_all_stages(x, y, background, y_bg_sub, y_smooth, noise_level,
                         cand_indices, clusters, cluster_fit_results,
                         all_peaks, snr_threshold, profile)

    diagnostics = {
        "y_bg_subtracted": y_bg_sub,
        "y_smoothed": y_smooth,
        "noise_level": noise_level,
        "candidate_indices": cand_indices,
        "clusters": clusters,
        "cluster_fit_results": cluster_fit_results,
    }
    #return refined_peaks, background, diagnostics
    return all_peaks, background, diagnostics


# ═══════════════════════════════════════════════════════════════════
# Visualization (all plotting code isolated here)
# ═══════════════════════════════════════════════════════════════════

# Color palette
_C = {
    "raw": "#333333", "bg": "#1f77b4", "sub": "#2ca02c",
    "smooth": "#d62728", "cand": "#ff7f0e", "fit": "#e41a1c",
    "fill": "#9467bd", "noise": "#bcbd22", "resid": "#7f7f7f",
}


def _plot_all_stages(x, y, background, y_bg_sub, y_smooth, noise_level,
                     cand_indices, clusters, cluster_fit_results,
                     final_peaks, snr_threshold, profile):
    """All diagnostic plots for the pipeline stages."""
    import matplotlib.pyplot as plt

    cmap = plt.cm.tab10

    # ── Stage 1: Background ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.set_title("Stage 1: Background Estimation", fontsize=12,
                 fontweight="bold", loc="left")
    ax.plot(x, y, color=_C["raw"], lw=0.7, alpha=0.7, label="Raw data")
    ax.plot(x, background, color=_C["bg"], lw=2, ls="--",
            label="Estimated background")
    ax.fill_between(x, 0, background, color=_C["bg"], alpha=0.08)
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=9)
    ax.set_xlim(x[0], x[-1])

    ax = axes[1]
    ax.plot(x, y_bg_sub, color=_C["sub"], lw=0.7, label="Background-subtracted")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Intensity (bg sub)")
    ax.set_xlabel("2θ")
    ax.legend(fontsize=9)
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout()
    plt.show()

    # ── Stage 2: Candidates ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.set_title("Stage 2: Smoothing & Candidate Detection", fontsize=12,
                 fontweight="bold", loc="left")
    ax.plot(x, y_bg_sub, color=_C["sub"], lw=0.5, alpha=0.5,
            label="Bg-subtracted")
    ax.plot(x, y_smooth, color=_C["smooth"], lw=1.5, label="Smoothed")
    ax.axhline(snr_threshold * noise_level, color=_C["noise"], ls=":",
               lw=1.2, label=f"SNR threshold ({snr_threshold}σ)")

    if len(cand_indices) > 0:
        ax.plot(x[cand_indices], y_smooth[cand_indices], "v",
                color=_C["cand"], ms=10, zorder=5,
                label=f"Candidates ({len(cand_indices)})")
        for ci in cand_indices:
            ax.annotate(f"{x[ci]:.2f}", (x[ci], y_smooth[ci]),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, ha="center", color=_C["cand"],
                        fontweight="bold")
    ax.set_ylabel("Intensity (bg sub)")
    ax.legend(fontsize=9)
    ax.set_xlim(x[0], x[-1])

    ax = axes[1]
    ax.plot(x, y_bg_sub - y_smooth, color=_C["resid"], lw=0.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.fill_between(x, -noise_level, noise_level, color=_C["noise"], alpha=0.1)
    ax.set_ylabel("Smoothing residual")
    ax.set_xlabel("2θ")
    ax.text(0.98, 0.92, f"Noise σ ≈ {noise_level:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                      alpha=0.8))
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout()
    plt.show()

    # ── Stage 3: Clustering ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_title("Stage 3: Peak Clustering", fontsize=12,
                 fontweight="bold", loc="left")
    ax.plot(x, y_bg_sub, color=_C["sub"], lw=0.5, alpha=0.4)
    ax.plot(x, y_smooth, color=_C["smooth"], lw=1, alpha=0.6)
    avg_dx = np.median(np.diff(x))

    from matplotlib.lines import Line2D
    legend_els = []
    for ci, cluster in enumerate(clusters):
        color = cmap(ci % 10)
        if cluster:
            cl_lo = x[cluster[0]] - 15 * avg_dx
            cl_hi = x[cluster[-1]] + 15 * avg_dx
            m = (x >= cl_lo) & (x <= cl_hi)
            ax.fill_between(x[m], 0, y_smooth[m], alpha=0.15, color=color)
        for pidx in cluster:
            ax.plot(x[pidx], y_smooth[pidx], "v", color=color, ms=10, zorder=5)
        centers_str = ", ".join(f"{x[p]:.2f}" for p in cluster)
        legend_els.append(Line2D([0], [0], marker="v", color=color, lw=0,
                                 ms=8, label=f"C{ci+1}: {centers_str}"))
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)
    ax.set_ylabel("Intensity (bg sub)")
    ax.set_xlabel("2θ")
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout()
    plt.show()

    # ── Stage 4: Cluster fits ──
    if cluster_fit_results:
        n_clusters = len(cluster_fit_results)
        n_cols = min(n_clusters, 4)
        n_rows = int(np.ceil(n_clusters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 3.5 * n_rows),
                                 squeeze=False)
        fig.suptitle("Stage 4: Individual Cluster Fits", fontsize=13,
                     fontweight="bold", x=0.02, ha="left")

        for ci, cinfo in enumerate(cluster_fit_results):
            row, col = divmod(ci, n_cols)
            ax = axes[row][col]
            fitted = cinfo["fitted_peaks"]
            color = cmap(ci % 10)

            if not fitted:
                ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes,
                        ha="center", va="center", color="red")
                ax.set_title(f"Cluster {ci+1}", fontsize=10)
                continue

            x_lo = min(pk["fit_window"][0] for pk in fitted)
            x_hi = max(pk["fit_window"][1] for pk in fitted)
            pad = (x_hi - x_lo) * 0.15
            m = (x >= x_lo - pad) & (x <= x_hi + pad)
            xw, yw = x[m], y_bg_sub[m]

            ax.plot(xw, yw, "o", color=_C["sub"], ms=2, alpha=0.5)

            y_total = np.zeros_like(xw)
            for pi, pk in enumerate(fitted):
                if pk["profile"] == "pseudo_voigt":
                    yc = pseudo_voigt(xw, pk["amplitude"], pk["center"],
                                      pk["sigma"], pk["eta"])
                else:
                    yc = gaussian(xw, pk["amplitude"], pk["center"],
                                  pk["sigma"])
                y_total += yc
                ax.fill_between(xw, 0, yc, alpha=0.2,
                                color=cmap((ci * 3 + pi) % 10))
                ax.axvline(pk["center"], color="green", ls=":", lw=0.8)
                ax.annotate(f'{pk["center"]:.3f}\nFWHM={pk["fwhm"]:.3f}',
                            (pk["center"], pk["amplitude"]),
                            textcoords="offset points", xytext=(5, 8),
                            fontsize=6.5, color="green",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="green", alpha=0.7))

            ax.plot(xw, y_total, color=_C["fit"], lw=1.8)
            resid = yw - y_total
            offset = -0.1 * np.max(yw)
            ax.plot(xw, resid + offset, color=_C["resid"], lw=0.7)
            ax.axhline(offset, color="gray", ls="--", lw=0.4)

            n_pk = len(fitted)
            ax.set_title(f"Cluster {ci+1} ({n_pk} peak{'s' if n_pk > 1 else ''})",
                         fontsize=10, fontweight="bold", color=color)
            ax.set_xlabel("2θ", fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.tick_params(labelsize=7)

        for ci in range(n_clusters, n_rows * n_cols):
            row, col = divmod(ci, n_cols)
            axes[row][col].set_visible(False)
        fig.tight_layout()
        plt.show()

    # ── Stage 5: Final reconstruction ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.set_title("Stage 5: Final Result", fontsize=12,
                 fontweight="bold", loc="left")
    ax.plot(x, y, color=_C["raw"], lw=0.7, alpha=0.6, label="Raw data")
    ax.plot(x, background, color=_C["bg"], lw=1.5, ls="--", label="Background")

    y_fit_total = np.zeros_like(x)
    for i, pk in enumerate(final_peaks):
        if pk["profile"] == "pseudo_voigt":
            ys = pseudo_voigt(x, pk["amplitude"], pk["center"],
                              pk["sigma"], pk["eta"])
        else:
            ys = gaussian(x, pk["amplitude"], pk["center"], pk["sigma"])
        y_fit_total += ys
        ax.fill_between(x, background, background + ys, alpha=0.12,
                        color=cmap(i % 10))

    ax.plot(x, background + y_fit_total, color=_C["fit"], lw=1.8,
            label="Total fit")
    for pk in final_peaks:
        ax.axvline(pk["center"], color="green", ls=":", lw=0.6, alpha=0.6)

    ax.set_ylabel("Intensity")
    ax.legend(fontsize=9)
    ax.set_xlim(x[0], x[-1])
    ax.text(0.02, 0.95, f"{len(final_peaks)} peaks found",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                      ec="gray", alpha=0.9))

    ax = axes[1]
    residual_final = y_bg_sub - y_fit_total
    ax.plot(x, residual_final, color=_C["resid"], lw=0.6)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.fill_between(x, -noise_level, noise_level, color=_C["noise"],
                    alpha=0.07)
    rms = np.sqrt(np.mean(residual_final ** 2))
    ax.set_ylabel("Residual")
    ax.set_xlabel("2θ")
    ax.text(0.98, 0.92,
            f"RMS residual = {rms:.2f}  |  Noise σ = {noise_level:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                      alpha=0.8))
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout()
    plt.show()


def _plot_global_diagnostics(x, y_bg_sub, refined, pruned, new_peaks,
                             trend, noise_level, iteration, profile):
    """Diagnostic plots for one global optimization iteration."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Global Optimization — Iteration {iteration + 1}",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")

    # Panel 1: FWHM trend
    ax = axes[0, 0]
    if refined:
        c = np.array([pk["center"] for pk in refined])
        f = np.array([pk["fwhm"] for pk in refined])
        ax.scatter(c, f, c="#2ca02c", s=40, zorder=5, label="Kept")
    if pruned:
        c = np.array([pk["center"] for pk in pruned])
        f = np.array([pk["fwhm"] for pk in pruned])
        ax.scatter(c, f, c="#d62728", s=40, marker="x", zorder=5,
                   linewidths=2, label="Pruned")
    if new_peaks:
        c = np.array([pk["center"] for pk in new_peaks])
        f = np.array([pk["fwhm"] for pk in new_peaks])
        ax.scatter(c, f, c="#ff7f0e", s=40, marker="^", zorder=5,
                   label="Added")

    xt = np.linspace(x[0], x[-1], 200)
    yt = trend["slope"] * xt + trend["intercept"]
    ax.plot(xt, yt, "k--", lw=1.5, label="Trend")
    ax.fill_between(xt, yt - 3 * trend["residual_sigma"],
                    yt + 3 * trend["residual_sigma"],
                    alpha=0.1, color="gray")
    ax.set_xlabel("2θ")
    ax.set_ylabel("FWHM")
    ax.set_title("FWHM vs Position", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Reconstruction
    ax = axes[0, 1]
    y_model = evaluate_peaks(x, refined, profile)
    ax.plot(x, y_bg_sub, color="#333", lw=0.5, alpha=0.5, label="Data")
    ax.plot(x, y_model, color="#e41a1c", lw=1.5, label="Model")
    ax.set_xlabel("2θ")
    ax.set_ylabel("Intensity")
    ax.set_title("Pattern Reconstruction", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: Residual
    ax = axes[1, 0]
    residual = y_bg_sub - y_model
    ax.plot(x, residual, color="#7f7f7f", lw=0.5)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.fill_between(x, -3 * noise_level, 3 * noise_level,
                    color="#bcbd22", alpha=0.07)
    rms = np.sqrt(np.mean(residual ** 2))
    ax.set_xlabel("2θ")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual (RMS={rms:.2f}, σ={noise_level:.2f})",
                 fontsize=10, fontweight="bold")

    # Panel 4: Stats
    ax = axes[1, 1]
    ax.axis("off")
    stats = (
        f"Peaks kept:    {len(refined)}\n"
        f"Peaks pruned:  {len(pruned)}\n"
        f"Peaks added:   {len(new_peaks)}\n"
        f"Noise σ:       {noise_level:.4f}\n"
        f"RMS residual:  {rms:.4f}\n"
        f"Trend slope:   {trend['slope']:.6f}\n"
        f"Trend int:     {trend['intercept']:.6f}\n"
        f"Trend σ:       {trend['residual_sigma']:.6f}"
    )
    ax.text(0.1, 0.9, stats, transform=ax.transAxes, fontsize=11,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="gray", alpha=0.9))
    ax.set_title("Summary", fontsize=10, fontweight="bold")

    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# Convenience: print peak table
# ═══════════════════════════════════════════════════════════════════

def print_peak_table(peaks):
    """Print a simple summary table of fitted peaks."""
    if not peaks:
        print("No peaks found.")
        return

    header = f"{'#':>3}  {'Center':>10}  {'±':>8}  {'Amplitude':>10}  {'FWHM':>8}  {'σ':>8}  {'Profile':>10}"
    if any(pk.get("eta") is not None for pk in peaks):
        header += f"  {'η':>6}"
    print(header)
    print("-" * len(header))

    for i, pk in enumerate(peaks):
        err_str = f"{pk['center_err']:.5f}" if pk.get("center_err") else "   —"
        line = (f"{i+1:>3}  {pk['center']:>10.5f}  {err_str:>8}  "
                f"{pk['amplitude']:>10.2f}  {pk['fwhm']:>8.5f}  "
                f"{pk['sigma']:>8.5f}  {pk['profile']:>10}")
        if pk.get("eta") is not None:
            line += f"  {pk['eta']:>6.3f}"
        print(line)
