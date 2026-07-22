"""
fit_engine.py — OPHIRA fit and analysis logic
Depends on physics.py and app_state.py. No UI dependency (matplotlib/PyQt).

The methods of this class:
- Read the values from the passed parameters or from AppState
- Return results (do not modify UI widgets)
- Use optional callbacks for progress notifications to the UI

The fit methods run_fit, run_fit_multi and run_fit_auto are defined below in this class.
"""
import numpy as np
import os
import csv
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.optimize import minimize

import physics as ph
from app_state import AppState


def _compute_sigma_nodes(state, wl_exp, r_exp, en, ek, extn, ang, extra=None, eps=1e-4):
    """Estimate sigma_n, sigma_k for each ext-node via numerical perturbation
    (diagonal Hessian). For each node j, perturb en[j] or ek[j] by ±eps, compute
    the resulting R and derive sigma = 1/sqrt(d²f/dx²) with f = SSE(R-r_exp).

    The forward model here MUST be the one that produced the fit, otherwise the
    curvature is that of a different function: the parameters come from
    state_to_refl_kwargs (the same source run_fit builds its own from), and at an
    HOBL angle the angular spread is applied as in the fit — `extra` carries the
    spread the fit converged to ({'da','wc','wl','wr'}).
    Returns (sigma_n, sigma_k) of shape (Ne,) each, or (None, None) on error.

    Note: this is an indicative diagonal estimate; a full covariance or a
    Monte-Carlo over the noise would be more rigorous."""
    Ne = len(extn)
    sigma_n = np.zeros(Ne)
    sigma_k = np.zeros(Ne)
    kw = ph.state_to_refl_kwargs(state)
    is_hobl = (ang in ph._HOBL_ANGLES) and bool(extra)

    def _R(n_arr, k_arr):
        if is_hobl:
            return ph.calc_refl_3angle(wl_exp, state.d, extn, n_arr, k_arr, float(ang),
                                       extra.get('da', 0.0), extra.get('wc', 0.5),
                                       extra.get('wl', 0.25), extra.get('wr', 0.25),
                                       **kw)[0]
        return ph.calculate_refl_core(wl_exp, state.d, extn, n_arr, k_arr,
                                      theta_deg=ang, **kw)[0]

    try:
        f0 = float(np.sum((_R(en, ek) - r_exp)**2))
        for j in range(Ne):
            for arr, store in [('n', sigma_n), ('k', sigma_k)]:
                base = en if arr == 'n' else ek
                bp = base.copy(); bp[j] += eps
                bm = base.copy(); bm[j] -= eps
                enp, ekp = (bp, ek) if arr == 'n' else (en, bp)
                enm, ekm = (bm, ek) if arr == 'n' else (en, bm)
                d2f = (float(np.sum((_R(enp, ekp) - r_exp)**2)) - 2*f0
                       + float(np.sum((_R(enm, ekm) - r_exp)**2))) / eps**2
                store[j] = 1.0 / np.sqrt(max(d2f, 1e-30))
        return sigma_n, sigma_k
    except Exception:
        return None, None


class FitEngine:
    """Fit and analysis engine. A single instance shared across all windows."""

    def __init__(self, state: AppState):
        self.state = state
        # Optional callbacks for UI updates during long operations
        # The UI registers its own functions here after creation
        self.on_progress = None    # callable(msg: str)
        # callable(chi2, n_iter, R, nf, kf, ext, en, ek, ang) — `ang` is the angle
        # being fitted: the UI cannot read it off state.current_ang, which is the
        # angle the user is LOOKING at and is free to change mid-fit.
        self.on_fit_step = None
        # d_from_peaks cache for run_fit_auto: computed on the first cycle, fixed
        # for the following ones. Invalidated at the start of every run_fit_auto.
        self._d_peaks_cache = None
        # Peak-centering penalty: adds to the chi² a term that forces R_fit to have
        # (a) zero first derivative and (b) negative concavity at the exp peak
        # positions (extracted at the start of the fit with _extract_peaks
        # 'exp_only'). The peaks are the "cleanest" observable (main dependence
        # 2·n·d·cos θ = mλ), and on thin samples (m=1/m=2) the plain least-squares
        # misfit on R does not center them correctly. Applied to all angles
        # (8/20/40/60), both in run_fit (single) and in run_fit_multi (obj_s1, obj_s2).
        self.peak_pen_on = True
        # Weights: they produce a penalty ~1-10%
        # of the data term in normal cases, dominant when the simulated peaks are
        # very far from the exp ones.
        self.peak_pen_w1 = 5.0e4    # weight (dR/dλ)²
        self.peak_pen_w2 = 1.0e8    # weight max(0, d²R/dλ²)² (concavity constraint)
        # Minimum λ for the centering penalty: BELOW this λ the penalty does NOT
        # act. The deep blue (<500 nm) has low contrast and noisy peak extraction →
        # an exp peak on a model trough makes the concavity term (w2) explode.
        # 500 nm excludes the deep blue while keeping the whole visible from
        # ~514 nm up. See _peak_penalty.
        self.peak_pen_lam_min = 500.0
        # Exclusion band for known spurious peaks:
        # the Lambda 950 detector swap Si→InGaAs around 860-910 nm can produce
        # micro-peaks that _extract_peaks 'exp_only' does not distinguish from the
        # real interference fringes. Discarded for the penalty.
        self.peak_pen_exclude = [(860.0, 910.0)]
        # IR anchoring at the HOBL angles: km/ks/kb model the detector jump at
        # 890nm, which is a CONSTANT of the spectrometer (it does not vary with
        # angle/measurement). When True, at the HOBL angles (40/60) km/ks/kb are NOT
        # fitted but locked to the state values (typically from the low angles 8/20,
        # where the fit is robust). It removes ks(~120nm)+kb(~108nm) from the
        # position levers → breaks part of the degeneracy that prevents isolating Δn.
        self.anchor_ir_hobl = True
        # Per-angle MEAS JUMP in the multi fit: refinement of the model above. The
        # detector jump at 890nm is NOT a single constant — it varies per sample and
        # per angle (responsivity × slit broadening). When True, run_fit_multi
        # MEASURES km from each angle's data (ph.measure_ir_jump) and KEEPS it FIXED
        # during the fit, instead of leaving it free (which trades it off against
        # n/k/scatt) or anchoring it to a single value. It takes precedence over
        # anchor_ir_hobl for km only (ks/kb stay handled by anchor_ir_hobl).
        self.measure_jump_multi = True
        # POSITION penalty for the IR peaks at the HOBL angles.
        # The derivative/concavity _peak_penalty has a position stiffness ∝ the
        # peak's R'' curvature: on the WIDE IR fringes at 40° R'' is small →
        # "wobbly" centering (slow IR residual ±4%, a peak signature with the right
        # amplitude but a position shifted by ~tens of nm). This term anchors the
        # position DIRECTLY (smooth weighted centroid) with stiffness ∝ w·δ,
        # independent of R''. Applied only to the peaks λ≥peak_pos_lam_min.
        self.peak_pos_on     = True
        # Maxima-position anchoring weight — ANGLE-DEPENDENT:
        # at 40° the fringes are already aligned → LOW weight (a high weight would
        # deform them). At 60° they are misaligned → a HIGH weight is needed to
        # compete with the regularization (1e9). A single weight cannot serve both angles.
        self.peak_pos_w      = 0.5      # 40°
        self.peak_pos_w_hi   = 500.0    # 60°
        self.peak_pos_lam_min = 1000.0

    def _filter_peaks_excluded(self, peaks_pos):
        """Remove peaks that fall inside the exclusion bands
        (`peak_pen_exclude` = list of tuples (λ_lo, λ_hi) in nm).
        Returns np.ndarray, or None if the input is None / empty list.
        """
        if peaks_pos is None:
            return None
        bands = getattr(self, 'peak_pen_exclude', None) or []
        if not bands:
            return np.asarray(peaks_pos)
        arr = np.asarray(peaks_pos, dtype=float)
        keep = np.ones(len(arr), dtype=bool)
        for lo, hi in bands:
            keep &= ~((arr >= lo) & (arr <= hi))
        return arr[keep]

    def _peak_penalty(self, R, wl, peaks_pos):
        """Centering penalty of the simulated maxima on the exp positions.

        For each position λ_i in `peaks_pos`, compute via centered finite
        differences the first and second derivative of R(λ) at the point:
        - penalty1 = w1 · (dR/dλ)²        (we want ≈ 0 at the maximum)
        - penalty2 = w2 · max(0, d²R/dλ²)² (we want concavity ≤ 0)

        Returns 0 if peak_pen_on=False, peaks_pos empty, or R None.
        Differentiable function in the fit parameters (linear in R), compatible
        with L-BFGS-B finite-difference.

        OPT-IN: acts ONLY if `state.low_contrast_aid` is True (FRINGE AID toggle,
        default OFF). On well-contrasted fringes it is useless and makes the fit
        "go crazy" (see AppState.low_contrast_aid). When active it
        skips peaks with λ < `peak_pen_lam_min` (deep blue): that is where the w2
        concavity term explodes on an exp peak landing on a trough.
        """
        if not self.peak_pen_on or peaks_pos is None or len(peaks_pos) == 0:
            return 0.0
        if not getattr(self.state, 'low_contrast_aid', False):
            return 0.0
        if R is None or wl is None or len(R) < 3:
            return 0.0
        w1, w2 = float(self.peak_pen_w1), float(self.peak_pen_w2)
        lam_min = float(getattr(self, 'peak_pen_lam_min', 500.0))
        wl = np.asarray(wl); R = np.asarray(R)
        pen = 0.0
        n = len(wl)
        for lam in peaks_pos:
            if lam < lam_min:            # skip the deep blue (concavity blow-up)
                continue
            j = int(np.argmin(np.abs(wl - lam)))
            if j <= 0 or j >= n - 1:
                continue
            wl_m, wl_0, wl_p = float(wl[j-1]), float(wl[j]), float(wl[j+1])
            R_m, R_0, R_p = float(R[j-1]), float(R[j]), float(R[j+1])
            h_total = wl_p - wl_m
            if h_total <= 0:
                continue
            # Centered first derivative (handles non-uniform steps)
            d1 = (R_p - R_m) / h_total
            pen += w1 * d1 * d1
            # Second derivative (assumes ~uniform step, true in the Lambda data)
            h_avg = 0.5 * h_total
            d2 = (R_p - 2.0 * R_0 + R_m) / (h_avg * h_avg)
            if d2 > 0.0:
                pen += w2 * d2 * d2
        return pen

    def _peak_pos_penalty(self, R, wl, peaks_pos, lam_min=1000.0, w=None):
        """POSITION penalty for the IR peaks (robust centering at 40°).

        Complementary to `_peak_penalty` (derivative/concavity), whose position
        stiffness ∝ the peak's R'' curvature → weak on the WIDE IR fringes at 40°.
        This term anchors the model's peak POSITION to the exp one with stiffness
        ∝ w·δ, independent of R''.

        For each exp peak λ_i ≥ lam_min:
          - symmetric window W within the data, width < half the distance to the
            adjacent IR peak (so it doesn't catch two fringes);
          - local linear detrend of R in the window (the centroid must follow the
            PEAK, not the IR background slope, strong between 2000-2500);
          - model peak position = smooth weighted centroid
            λ_cm = Σλ·g/Σg with g = max(0, R_detrend)²  (C¹ → smooth for
            L-BFGS-B finite-difference; FIXED window → no jump from argmax);
          - pen += w·(λ_cm − λ_i)².

        Returns 0 if off / no IR peak / R None.
        """
        if not getattr(self, 'peak_pos_on', False) or peaks_pos is None or len(peaks_pos) == 0:
            return 0.0
        if R is None or wl is None or len(R) < 5:
            return 0.0
        wl = np.asarray(wl, dtype=float); R = np.asarray(R, dtype=float)
        pk = np.sort(np.asarray(peaks_pos, dtype=float))
        pk = pk[pk >= float(lam_min)]
        if len(pk) == 0:
            return 0.0
        w = float(self.peak_pos_w if w is None else w)
        wl_lo, wl_hi = float(wl[0]), float(wl[-1])
        pen = 0.0
        for idx, lam in enumerate(pk):
            # half-window: < half the distance to the adjacent IR peak
            if len(pk) > 1:
                if idx == 0:
                    gap = pk[1] - pk[0]
                elif idx == len(pk) - 1:
                    gap = pk[-1] - pk[-2]
                else:
                    gap = min(pk[idx+1] - pk[idx], pk[idx] - pk[idx-1])
                half = 0.45 * gap
            else:
                half = 250.0
            half = float(np.clip(half, 120.0, 350.0))
            # symmetrize within the data: a centroid over an asymmetric window
            # would be biased even without a real shift of the peak.
            half = min(half, lam - wl_lo, wl_hi - lam)
            if half < 80.0:
                continue
            m = (wl >= lam - half) & (wl <= lam + half)
            if np.count_nonzero(m) < 3:
                continue
            wlw = wl[m]; Rw = R[m]
            # local linear detrend (linear operation in Rw → smooth)
            try:
                A = np.vstack([wlw, np.ones_like(wlw)]).T
                coef, *_ = np.linalg.lstsq(A, Rw, rcond=None)
                Rd = Rw - A @ coef
            except Exception:
                Rd = Rw - float(np.mean(Rw))
            g = np.maximum(0.0, Rd) ** 2
            s = float(np.sum(g))
            if s <= 1e-12:
                continue
            lam_cm = float(np.sum(wlw * g) / s)
            d = lam_cm - float(lam)
            pen += w * d * d
        return pen

    def compute_n_from_maxima(self, ang, d):
        """n(λ) from the POSITION of the maxima for an angle (n_eff method).

        For each peak (peaks_wl[i], m_values[i]) solve for n the fringe equation
        2·n·d·cos(θ_refr) = m·λ , with θ_refr = asin(sinθ/n), via brentq in the
        physical range [1.1, 3.5]. Shared by the peak window and the main window
        (Fit/Maxima view).

        Requires `state.data_angles[ang]` to have 'peaks_wl' and 'm_values'
        (populated by calcola_d_da_picchi / _calcola_nd) and a valid d. Writes the
        result into `state.data_angles[ang]['n_maxima']` (shared store) and returns
        it as np.array [[λ, n], ...], or None if the inputs are missing.
        """
        from scipy.optimize import brentq
        s = self.state
        dd = s.data_angles.get(ang)
        if dd is None or d is None:
            return None
        wl_pks = dd.get("peaks_wl")
        m_vals = dd.get("m_values")
        if wl_pks is None or m_vals is None:
            return None

        def f_n2(n2, m, lam):
            sin_ratio = np.sin(np.radians(float(ang))) / n2
            if sin_ratio > 1:
                return 1.0
            return 2.0 * n2 * d * np.sqrt(1.0 - sin_ratio ** 2) - (m * lam)

        n_list = []
        for i, wl in enumerate(wl_pks):
            if i >= len(m_vals):
                break
            m = m_vals[i]
            try:
                n_sol = brentq(f_n2, 1.1, 3.5, args=(m, wl))
                n_list.append([float(wl), float(n_sol)])
            except (ValueError, RuntimeError):
                continue
        arr = np.array(n_list) if n_list else None
        dd['n_maxima'] = arr
        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # KRAMERS-KRONIG
    # ─────────────────────────────────────────────────────────────────────────

    def calcola_kk(self, nodes_wl: np.ndarray,
                   curr_n: np.ndarray,
                   curr_k: np.ndarray):
        """Compute n_KK and k_KK via MSKK-1 (k → n).

        Receives the node values (nodes_wl, curr_n, curr_k) as parameters rather
        than reading from the sliders.

        Args:
            nodes_wl: array of node wavelengths (nm)
            curr_n:   n values at the nodes
            curr_k:   k values at the nodes

        Returns:
            (wl_out, n_kk, k_kk) or None on error
        """
        from scipy.interpolate import PchipInterpolator as _Pchip
        from scipy.ndimage import gaussian_filter1d as _gf

        c_nd = nodes_wl.copy().astype(float)
        wl_min = float(c_nd[0])
        wl_max = float(c_nd[-1])

        E_grid = np.linspace(0.05, 30.0, 35000)
        dE = float(E_grid[1] - E_grid[0])
        wl_of_E = 1240.0 / E_grid

        pn = _Pchip(c_nd, curr_n, extrapolate=False)
        pk = _Pchip(c_nd, curr_k, extrapolate=False)

        n_E = np.where(wl_of_E > wl_max, float(curr_n[-1]),
              np.where(wl_of_E < wl_min, float(curr_n[0]),
                       np.nan_to_num(pn(wl_of_E), nan=float(curr_n[0]))))
        k_E = np.where(wl_of_E > wl_max, 0.0,
              np.where(wl_of_E < wl_min, float(curr_k[0]),
                       np.maximum(np.nan_to_num(pk(wl_of_E), nan=0.0), 0.0)))

        eps1 = n_E**2 - k_E**2
        eps2 = 2.0 * n_E * k_E

        wl_out = np.arange(wl_min, wl_max + 1.0, 2.0)
        E_out  = 1240.0 / wl_out
        n_kk   = np.zeros(len(wl_out))
        k_kk   = np.zeros(len(wl_out))

        # MSKK-1: IR anchor at 2500nm
        E_IR = 1240.0 / 2500.0
        idx_IR_data = np.argmin(np.abs(E_grid - E_IR))
        eps1_target_IR = float(eps1[idx_IR_data])

        with np.errstate(divide='ignore', invalid='ignore'):
            ig_n_IR = E_grid * eps2 / (E_grid**2 - E_IR**2)
        mask_sing_ir = (np.abs(E_grid - E_IR) < 2.5 * dE)
        ig_n_IR = np.where(mask_sing_ir, 0.0, np.nan_to_num(ig_n_IR))
        int_n_IR = (2.0 / np.pi) * np.sum(ig_n_IR) * dE

        for i, E0 in enumerate(E_out):
            d0 = E_grid**2 - E0**2
            with np.errstate(divide='ignore', invalid='ignore'):
                ig_n = E_grid * eps2 / d0
            mask_sing_n = (np.abs(E_grid - E0) < 2.5 * dE)
            ig_n = np.where(mask_sing_n, 0.0, np.nan_to_num(ig_n))
            int_n = (2.0 / np.pi) * np.sum(ig_n) * dE

            eps1_kk  = eps1_target_IR + (int_n - int_n_IR)
            eps2_fit = float(np.interp(E0, E_grid, eps2))
            mod      = np.sqrt(max(eps1_kk**2 + eps2_fit**2, 0.0))
            n_kk[i]  = np.sqrt(max((eps1_kk + mod) / 2.0, 0.0))
            k_kk[i]  = float(np.interp(wl_out[i], c_nd, curr_k))

        return wl_out, n_kk, k_kk

    def calcola_kk_uncertainty(self, nodes_wl, curr_n, curr_k, sigma_k_nodes, kk=None):
        """Propagate σ_k → σ of n_KK through the MSKK-1 transform.

        Two steps, both dictated by what calcola_kk actually computes:

        1. The Hilbert integral returns ε₁, NOT n — calcola_kk then obtains
           n = √((ε₁+|ε|)/2). So σ_ε₂ propagates to σ_ε₁, and reaching n takes the
           Jacobian dn/dε₁ = (1 + ε₁/|ε|)/(4n), which is 1/(2n) wherever ε₂→0 (the
           transparency window) and departs from it where the film absorbs.
        2. The kernel is the SUBTRACTED one, K(Eⱼ,E₀) − K(Eⱼ,E_IR), because the
           transform is anchored: ε₁(E₀) = ε₁_fit(E_IR) + [∫(E₀) − ∫(E_IR)].

        With ε₂ = 2·n·k  →  σ_ε₂ = 2·n·σ_k  (neglecting σ_n, much smaller)
        σ²_ε₁(λ₀) = (2/π·ΔE)² Σⱼ [(K(Eⱼ,E₀) − K(Eⱼ,E_IR))·σ_ε₂(Eⱼ)]²

        What comes out is the uncertainty the TRANSFORM adds given σ_k. It goes to
        zero at the IR anchor, where ε₁ is imposed by the fit instead of predicted:
        that pinch is the reminder that the anchor is not a KK result. The anchor's
        own uncertainty is deliberately left out — it is built from n_fit, so it is
        common to n_fit and n_KK, and adding it in quadrature would inflate the very
        bar used to judge |n_fit − n_KK|. Sizing that term properly needs the
        covariance between n_fit and n_KK, which this indicative estimate does not
        carry.

        Args:
            nodes_wl:       array of nodes (nm)
            curr_n, curr_k: n,k values at the nodes
            sigma_k_nodes:  σ_k at the nodes (from the fit's Hessian)
            kk:             (wl, n_KK, k_KK) from calcola_kk on the same nodes — the
                            RAW n_KK, not a smoothed one. Recomputed here if absent
                            or if its grid does not match.

        Returns:
            (wl_out, sigma_n_kk) — nm grid and σ of n_KK
        """
        from scipy.interpolate import PchipInterpolator as _Pchip

        c_nd   = nodes_wl.copy().astype(float)
        wl_min = float(c_nd[0]); wl_max = float(c_nd[-1])

        E_grid = np.linspace(0.05, 30.0, 35000)
        dE     = float(E_grid[1] - E_grid[0])
        wl_of_E = 1240.0 / E_grid

        # σ_ε₂ = 2·n·σ_k on the dense grid (PCHIP from the nodes); ε₂ itself is
        # needed for the Jacobian, and is built exactly as calcola_kk builds it.
        pn  = _Pchip(c_nd, curr_n,       extrapolate=False)
        pk  = _Pchip(c_nd, curr_k,       extrapolate=False)
        psk = _Pchip(c_nd, np.maximum(sigma_k_nodes, 0.0), extrapolate=False)
        n_E   = np.where(wl_of_E > wl_max, float(curr_n[-1]),
                np.where(wl_of_E < wl_min, float(curr_n[0]),
                         np.nan_to_num(pn(wl_of_E),  nan=float(curr_n[0]))))
        k_E   = np.where(wl_of_E > wl_max, 0.0,
                np.where(wl_of_E < wl_min, float(curr_k[0]),
                         np.maximum(np.nan_to_num(pk(wl_of_E), nan=0.0), 0.0)))
        sk_E  = np.where(wl_of_E > wl_max, 0.0,
                np.where(wl_of_E < wl_min, 0.0,
                         np.maximum(np.nan_to_num(psk(wl_of_E), nan=0.0), 0.0)))
        eps2       = 2.0 * n_E * k_E
        sigma_eps2 = 2.0 * n_E * sk_E   # σ_ε₂(E)

        wl_out = np.arange(wl_min, wl_max + 1.0, 2.0)
        E_out  = 1240.0 / wl_out
        sigma_n_kk = np.zeros(len(wl_out))

        # n_KK is needed for the Jacobian: take the transform's own output.
        n_kk = None
        if kk is not None and len(kk) == 3:
            w_kk, n_cand = np.asarray(kk[0], float), np.asarray(kk[1], float)
            if len(w_kk) == len(wl_out) and np.allclose(w_kk, wl_out):
                n_kk = n_cand
        if n_kk is None:
            res = self.calcola_kk(nodes_wl, curr_n, curr_k)
            if res is None:
                return wl_out, sigma_n_kk
            n_kk = np.asarray(res[1], float)

        def _kern(E0):
            """Same integrand and same singularity masking as calcola_kk."""
            d0 = E_grid**2 - E0**2
            m  = np.abs(E_grid - E0) < 2.5 * dE
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(m, 0.0, np.nan_to_num(E_grid / d0))

        E_IR  = 1240.0 / 2500.0      # the anchor of calcola_kk
        K_IR  = _kern(E_IR)
        prefactor = (2.0 / np.pi * dE) ** 2
        for i, E0 in enumerate(E_out):
            kernel = _kern(E0) - K_IR                       # subtracted: MSKK-1
            sigma_eps1 = np.sqrt(prefactor * np.sum((kernel * sigma_eps2)**2))
            # ε₁ back out of the transform's own n_KK: inverting
            # n² = (ε₁+√(ε₁²+ε₂²))/2  gives  ε₁ = n² − ε₂²/(4n²)
            n0 = max(float(n_kk[i]), 1e-9)
            e2 = float(np.interp(E0, E_grid, eps2))
            e1 = n0**2 - e2**2 / (4.0 * n0**2)
            mod = np.sqrt(max(e1**2 + e2**2, 1e-30))
            dn_deps1 = (1.0 + e1 / mod) / (4.0 * n0)
            sigma_n_kk[i] = sigma_eps1 * dn_deps1

        return wl_out, sigma_n_kk

    def kk_uv_values(self, kk_cache, nodes_wl: np.ndarray):
        """Compute the new n values for the λ < 450nm nodes from the KK cache.

        Returns:
            dict {node_index: n_val} with the values suggested by KK.
            The UI applies these values to the sliders.
        """
        if kk_cache is None:
            return {}
        wl_kk, n_kk, _ = kk_cache
        result = {}
        for i, wl_node in enumerate(nodes_wl):
            if wl_node < 450.0:
                n_val = float(np.interp(wl_node, wl_kk, n_kk))
                result[i] = np.clip(n_val, 1.0, 6.5)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL THICKNESS COMPUTATION (from the 8° and 20° peaks)
    # ─────────────────────────────────────────────────────────────────────────

    def _n_at_lambda(self, lam):
        """Interpolation of n(λ) from the current fit (state.ext_override).
        Returns None if the fit is not available yet."""
        ov = self.state.ext_override
        if ov is None or len(ov) < 3:
            return None
        ext_nodes, n_arr = ov[0], ov[1]
        if len(ext_nodes) < 2 or len(n_arr) != len(ext_nodes):
            return None
        return float(np.interp(lam, ext_nodes, n_arr))

    def _ir_tail_floor_penalty(self, n_arr, ext_nodes, mw, tol=0.02):
        """Floor on the extreme IR node (~2550 nm): penalizes n(2550) < n(2067) - tol.
        The 2550 is ~50 nm BEYOND the last experimental datum (extrapolation) and
        coincides with the truncation edge of the Hilbert (KK) integral → without a
        constraint it collapses (a dip in the n(λ) plot + a false KK
        inconsistency). Constrains ONLY the last node: the sub-nodes 2228/2389
        are already held by the neighbors. Soft quadratic penalty, active only
        below the floor → it is 0 in good fits.
        """
        n_arr = np.asarray(n_arr)
        if n_arr.size < 2:
            return 0.0
        i2550 = n_arr.size - 1                                    # last ext-node ≈ 2550
        i2067 = int(np.argmin(np.abs(np.asarray(ext_nodes) - 2067.0)))
        if i2067 >= i2550:
            return 0.0
        deficit = (n_arr[i2067] - tol) - n_arr[i2550]
        return mw * deficit * deficit if deficit > 0.0 else 0.0

    def _spacing_filter(self, peaks_idx, props, wf, arr, theta_deg, frac=0.5):
        """ADAPTIVE minimum distance between maxima = frac·Δλ(λ), with the expected
        fringe spacing Δλ(λ)=λ²/(2·n(λ)·d·cosθ_r) (∝λ²: small in the visible, large
        in the IR). Greedy by prominence/height: keeps the strongest ones, discards
        the neighbors within frac·Δλ. The IR peaks (separated ≫ frac·Δλ) stay INTACT
        → d unchanged; recovers the dense visible fringes that a plain fixed
        scalar distance merges.
        """
        idx = np.asarray(peaks_idx)
        if idx.size <= 1:
            return idx, props
        d = float(self.state.d)
        if d <= 0:
            return idx, props
        sin_th = np.sin(np.radians(float(theta_deg)))
        # CAP: the adaptive minimum separation is capped at 25 grid steps. In the
        # visible (small Δλ) the filter recovers the dense fringes (minsep<cap);
        # in the IR (large Δλ, wide fringes → thin samples) minsep is limited to
        # the cap, so the IR peaks that determine d are preserved by construction.
        # Without this cap, on a thin sample (Δλ≈1700 nm) the single IR peak
        # would be removed by a stronger neighbor → d lost.
        dwl_grid = float(np.median(np.diff(wf))) if wf.size > 1 else 3.0
        cap = 25.0 * dwl_grid
        strength = props.get('prominences')
        if strength is None or len(strength) != idx.size:
            strength = np.asarray(arr)[idx]
        order = np.argsort(strength)[::-1]   # greedy: strongest first
        keep = np.zeros(idx.size, dtype=bool)
        acc_lam = []
        for k in order:
            lam = float(wf[idx[k]])
            n = self._n_at_lambda(lam)
            if n is None or n <= 1.0:
                n = 1.7   # PSi fallback (n unavailable without a fit)
            sr = sin_th / n
            cos_r = np.sqrt(max(1.0 - sr * sr, 1e-6))
            dlam = lam * lam / (2.0 * n * d * cos_r)
            minsep = min(frac * dlam, cap)
            if all(abs(lam - la) >= minsep for la in acc_lam):
                keep[k] = True
                acc_lam.append(lam)
        kept = np.where(keep)[0]   # kept positions, in increasing-λ order
        idx_keep = idx[kept]
        new_props = {}
        for key, val in props.items():
            try:
                new_props[key] = np.asarray(val)[kept]
            except Exception:
                new_props[key] = val
        return idx_keep, new_props

    def _extract_peaks(self, ang, source='hybrid', search_half_nm=30.0):
        """Extract refined peaks with a parabolic 3-point fit for one angle.

        Method shared between calcola_d_interno and peak_window._calcola_nd,
        ensures consistency between the value shown in the analysis window and the
        one used by fit_auto. The logic is modulated by `source`:

        - 'hybrid' (default): finds peaks on the fit as a positional guide, for
          each peak searches for the exp-data max in a ±search_half_nm window,
          then parabolic 3-point on the exp data. Requires both exp and fit.
          Robust to spurious peaks (e.g. 889 nm detector swap) and to the fit's
          systematic shifts.
        - 'exp_only': find_peaks directly on the exp data + parabolic 3pt.
          Pro: independent of the fit. Con: vulnerable to spurious peaks.
        - 'fit_only': find_peaks on the fit + parabolic 3pt on the fit. Used when
          the fit is considered the "clean model" of the signal.

        Returns an np.ndarray of refined peak positions (decimal nm), or None if
        not computable.
        """
        state = self.state
        da = state.data_angles[ang]
        if da["exp"] is None:
            return None

        wl_a = da["exp"][0]
        r_exp_raw = da["exp"][1]
        mask = wl_a >= 450
        wf = wl_a[mask]
        rf_exp = np.asarray(r_exp_raw, dtype=float)[mask].copy()
        rf_fit = None
        if da["fit"] is not None:
            rf_fit = np.asarray(da["fit"], dtype=float)[mask].copy()

        # km/ks/kb de-correction (instrumental effect) applied uniformly
        sl = da.get("sliders") or {}
        km = sl.get("km") if sl.get("km") is not None else state.km
        ks = sl.get("ks") if sl.get("ks") is not None else state.ks
        kb = sl.get("kb") if sl.get("kb") is not None else state.kb
        m_ir = wf > 890
        if np.any(m_ir) and km > 0:
            # De-correction = inverse of the forward model. Single formula in
            # ph.ir_instr_factor.
            corr = ph.ir_instr_factor(wf[m_ir], km, ks, kb)
            rf_exp[m_ir] /= corr
            if rf_fit is not None:
                rf_fit[m_ir] /= corr

        def parabolic_3pt(arr, idx_g):
            """3-point sub-sample refinement on `arr` around idx_g.
            Used as a fallback when the parabolic fit in E is not applicable
            (window too small, peak at the edge, etc.).
            """
            if 0 < idx_g < len(wf) - 1:
                y_m, y_0, y_p = float(arr[idx_g-1]), float(arr[idx_g]), float(arr[idx_g+1])
                denom = y_m - 2.0 * y_0 + y_p
                if denom < -1e-12:
                    delta = 0.5 * (y_m - y_p) / denom
                    if -1.0 < delta < 1.0:
                        dwl = 0.5 * (float(wf[idx_g+1]) - float(wf[idx_g-1]))
                        return float(wf[idx_g]) + delta * dwl
            return float(wf[idx_g])

        # Parabolic refinement in ENERGY over a window adaptive to the FWHM:
        # - The Fabry-Perot peaks are symmetric in E=hc/λ but asymmetric in λ
        #   for low m (thin samples: the m=1 peak can have FWHM>100 nm with a
        #   more stretched red flank).
        # - Strategy: for each peak found, estimate the FWHM in λ via
        #   `peak_widths`, convert the ±FWHM/2 window to E, isolate the points in
        #   the upper half of the peak (R > R_half), fit y(E)=aE²+bE+c,
        #   vertex = -b/(2a) → back-convert λ.
        # - Limiting to the upper-half points: the parabolic symmetry is exact
        #   only near the vertex, on the flanks the real shape diverges from the
        #   parabola.
        HC_NM_EV = 1239.84198  # hc in nm·eV (E[eV]=HC/λ[nm])

        def parabolic_fit_E(arr, peaks_idx, properties):
            """Parabolic fit in E on the upper half of each peak.

            arr: R curve (de-corrected) of length len(wf)
            peaks_idx: indices of the peaks on `arr`
            properties: dict returned by find_peaks(..., width=...) — must
                contain 'widths' (in sample units)
            Returns a list of refined positions in nm (same cardinality as
            peaks_idx).

            Strategy:
            - base window = HWHM (FWHM/2): the region where the shape is well
              approximated by a parabola in E.
            - clamp with half the distance to the neighboring peaks to avoid
              spilling into the adjacent peak (the m=1 case where the peaks are
              100+ nm wide and sometimes adjacent through a shallow minimum).
            - "upper half" filter in height within the window.
            """
            widths_samples = properties.get('widths')
            peaks_arr = np.asarray(peaks_idx, dtype=int)
            results = []
            for k, idx_g in enumerate(peaks_arr):
                idx_g = int(idx_g)
                w_samp = float(widths_samples[k]) if widths_samples is not None else 10.0
                lo_idx = max(0, idx_g - 5)
                hi_idx = min(len(wf) - 1, idx_g + 5)
                dwl_local = float((wf[hi_idx] - wf[lo_idx]) / max(1, hi_idx - lo_idx))
                fwhm_nm = w_samp * dwl_local
                # 0.4×FWHM window: narrow enough to stay near the vertex where
                # the shape is truly parabolic in E; asymmetric flanks excluded.
                hwhm_nm = 0.4 * fwhm_nm
                # Clamp with the distance to adjacent peaks (half distance)
                neighbor_clamp = np.inf
                if k > 0:
                    d_prev = float(wf[idx_g] - wf[peaks_arr[k-1]])
                    neighbor_clamp = min(neighbor_clamp, 0.45 * d_prev)
                if k < len(peaks_arr) - 1:
                    d_next = float(wf[peaks_arr[k+1]] - wf[idx_g])
                    neighbor_clamp = min(neighbor_clamp, 0.45 * d_next)
                half_nm = min(hwhm_nm, neighbor_clamp)
                # Minimum to have enough points for a degree-2 polyfit
                half_nm = max(half_nm, 3.0 * dwl_local)
                lo_w = float(wf[idx_g]) - half_nm
                hi_w = float(wf[idx_g]) + half_nm
                in_win = (wf >= lo_w) & (wf <= hi_w)
                if int(in_win.sum()) < 5:
                    results.append(parabolic_3pt(arr, idx_g))
                    continue
                lam_w = wf[in_win].astype(float)
                R_w   = arr[in_win].astype(float)
                # "Upper half" filter: keep only the points with
                # R >= R_min + 0.5*(R_max - R_min). If <5 remain, relax to 35%.
                R_min, R_max = float(R_w.min()), float(R_w.max())
                amp = R_max - R_min
                if amp <= 0:
                    results.append(parabolic_3pt(arr, idx_g))
                    continue
                # "Upper part" threshold at 65%: concentrates the fit near the
                # vertex — on the flanks the shape diverges from the parabola and
                # biases the vertex for asymmetric peaks.
                thr = R_min + 0.65 * amp
                sel = R_w >= thr
                if int(sel.sum()) < 5:
                    thr = R_min + 0.5 * amp
                    sel = R_w >= thr
                if int(sel.sum()) < 5:
                    thr = R_min + 0.35 * amp
                    sel = R_w >= thr
                if int(sel.sum()) < 5:
                    results.append(parabolic_3pt(arr, idx_g))
                    continue
                lam_sel = lam_w[sel]
                R_sel   = R_w[sel]
                E_sel = HC_NM_EV / lam_sel        # eV
                try:
                    coeff = np.polyfit(E_sel, R_sel, 2)  # [a, b, c]
                except Exception:
                    results.append(parabolic_3pt(arr, idx_g))
                    continue
                a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
                if a >= -1e-9:  # wrong concavity
                    results.append(parabolic_3pt(arr, idx_g))
                    continue
                E_v = -b / (2.0 * a)
                lam_v = HC_NM_EV / E_v
                # Tight check: the vertex must lie within 0.3×half_nm of the
                # original argmax. If it deviates more, the parabolic fit in E is
                # unreliable (asymmetry + small R range → the vertex slips).
                # Fallback to parabolic_3pt, which is conservative (max 1 sample
                # of shift).
                lam_argmax = float(wf[idx_g])
                if abs(lam_v - lam_argmax) > 0.3 * half_nm:
                    results.append(parabolic_3pt(arr, idx_g))
                    continue
                results.append(float(lam_v))
            return results

        min_peaks = 1 if state.thin_film else 2

        if source == 'exp_only':
            # prominence=2.0 discards spurious noise peaks; ADAPTIVE distance
            # (∝λ²) via _spacing_filter. IR (for d) unchanged.
            peaks_idx, props = find_peaks(rf_exp, distance=8, height=1.0,
                                          prominence=2.0, width=1)
            peaks_idx, props = self._spacing_filter(peaks_idx, props, wf, rf_exp, ang)
            if len(peaks_idx) < min_peaks:
                return None
            return np.array(parabolic_fit_E(rf_exp, peaks_idx, props))

        if source == 'fit_only':
            if rf_fit is None:
                return None
            peaks_idx, props = find_peaks(rf_fit, distance=8, height=1.0,
                                          prominence=0.5, width=1)
            peaks_idx, props = self._spacing_filter(peaks_idx, props, wf, rf_fit, ang)
            if len(peaks_idx) < min_peaks:
                return None
            return np.array(parabolic_fit_E(rf_fit, peaks_idx, props))

        # source == 'hybrid' (default)
        if rf_fit is None:
            peaks_idx, props = find_peaks(rf_exp, distance=8, height=1.0,
                                          prominence=2.0, width=1)
            peaks_idx, props = self._spacing_filter(peaks_idx, props, wf, rf_exp, ang)
            if len(peaks_idx) < min_peaks:
                return None
            return np.array(parabolic_fit_E(rf_exp, peaks_idx, props))

        peaks_fit_idx, props_fit = find_peaks(rf_fit, distance=8, height=1.0,
                                              prominence=0.5, width=1)
        peaks_fit_idx, props_fit = self._spacing_filter(
            peaks_fit_idx, props_fit, wf, rf_fit, ang)
        if len(peaks_fit_idx) < min_peaks:
            return None
        # For hybrid: use the fit peaks as a guide, then search for the
        # corresponding max on the exp data in a ±search_half_nm window and refine
        # with the fit-in-E on the exp data.
        lam_fit_refined_arr = parabolic_fit_E(rf_fit, peaks_fit_idx, props_fit)
        refined = []
        for lam_fit_refined in lam_fit_refined_arr:
            lo, hi = lam_fit_refined - search_half_nm, lam_fit_refined + search_half_nm
            in_window = (wf >= lo) & (wf <= hi)
            if not np.any(in_window):
                continue
            idx_globals = np.where(in_window)[0]
            k_local = int(np.argmax(rf_exp[in_window]))
            idx_g = int(idx_globals[k_local])
            # Local mini find_peaks to get the exp peak width
            try:
                p_loc, props_loc = find_peaks(rf_exp[in_window], width=1)
                if len(p_loc) > 0:
                    k_best = int(np.argmin(np.abs(p_loc - k_local)))
                    props_one = {'widths': np.array([props_loc['widths'][k_best]])}
                else:
                    props_one = {'widths': np.array([10.0])}
            except Exception:
                props_one = {'widths': np.array([10.0])}
            lam_refined = parabolic_fit_E(rf_exp, np.array([idx_g]), props_one)[0]
            refined.append(lam_refined)
        return np.array(refined) if len(refined) >= min_peaks else None

    _HC_NM_EV = 1239.84198  # hc in nm·eV (E[eV] = HC/λ[nm])

    def extract_maxima_bragg(self, ang, source, d, wl_min=500.0,
                             frac=0.40, prom=1.0, smooth_tol=0.06):
        """Physics-guided maxima for n(λ)-from-maxima and the peak-window display.

        Plain find_peaks searches blindly: it both invents peaks from noise (e.g.
        the detector-edge ramp near 2500 nm) and misses obvious ones. Here the
        POSITION stays data-driven (find_peaks + parabolic-in-E refine), but the
        Bragg relation g(λ)=2·d·√(n(λ)²−sin²θ)/λ says WHERE the orders are: each
        found maximum is labelled with the nearest order m and KEPT only if it sits
        within a fraction of the local fringe spacing of that order. Maxima matching
        no order (spurious) are dropped; below `wl_min` the fringes are not resolved
        and are ignored. A final n(λ)-smoothness guard removes the odd survivor whose
        inclusion makes n jump (a mis-associated order) — the physical prior the fit
        already encodes (monotone/convex dispersion).

        m and d are NOT recomputed here: the order labels agree with _robust_m_start
        above wl_min (verified on 8/20/40/60), and d keeps its own path
        (calcola_d_da_picchi, 8°/20°). This method feeds ONLY the display and
        compute_n_from_maxima.

        Args:
            ang: angle (8/20/40/60)
            source: 'exp' or 'fit' — which curve carries the maxima
            d: thickness (nm), Bragg scaffold (state.d); positions do NOT depend on it
            wl_min: ignore maxima below this (unresolved fringes)
            frac: association tolerance = frac·(distance to the neighbouring order)
            prom: find_peaks prominence for the candidates (generous — Bragg filters)
            smooth_tol: |Δ²n| above which a survivor is auto-dropped by the guard

        Returns (peaks_wl, m_values) sorted by DESCENDING λ (as data_angles stores),
        or (None, None) if not computable.
        """
        state = self.state
        da = state.data_angles.get(ang)
        if da is None or da.get("exp") is None or d is None or d <= 0:
            return None, None
        wl_a = np.asarray(da["exp"][0], float)
        mask = wl_a >= 450
        wf = wl_a[mask]
        if source == 'fit':
            if da.get("fit") is None:
                return None, None
            rf = np.asarray(da["fit"], float)[mask].copy()
        else:
            rf = np.asarray(da["exp"][1], float)[mask].copy()
        # instrumental IR de-correction (same single source of truth as _extract_peaks)
        sl = da.get("sliders") or {}
        km = sl.get("km") if sl.get("km") is not None else state.km
        ks = sl.get("ks") if sl.get("ks") is not None else state.ks
        kb = sl.get("kb") if sl.get("kb") is not None else state.kb
        m_ir = wf > 890
        if np.any(m_ir) and km > 0:
            rf[m_ir] /= ph.ir_instr_factor(wf[m_ir], km, ks, kb)

        # scaffold n(λ): the per-angle fit result if present, else the global one,
        # else the current node model (always available, so no "no-fit" gap)
        ov = da.get("ext_override") or state.ext_override
        if ov is not None and len(ov) >= 2 and len(ov[0]) >= 2:
            _nodes, _narr = np.asarray(ov[0], float), np.asarray(ov[1], float)
            n_of = lambda lam: np.interp(lam, _nodes, _narr)
        else:
            _cn = ph.build_c_nodes(state)
            _nu = np.array([ph.n_map(v) for v in state.n_slider_vals])
            _fn = PchipInterpolator(_cn, _nu)
            n_of = lambda lam: _fn(lam)

        s2 = np.sin(np.radians(float(ang))) ** 2
        HC = self._HC_NM_EV

        def _refine_idx(i0):
            """Parabola in E anchored on the data max i0 (data-driven half-width)."""
            n = len(wf)
            lo = max(0, i0 - 12); hi = min(n, i0 + 13)
            r_local_min = float(rf[lo:hi].min())
            thr = r_local_min + 0.5 * (float(rf[i0]) - r_local_min)
            l = i0
            while l - 1 >= 0 and rf[l-1] >= thr:
                l -= 1
            r = i0
            while r + 1 < n and rf[r+1] >= thr:
                r += 1
            l = max(0, min(l, i0 - 2)); r = min(n - 1, max(r, i0 + 2))
            lam = wf[l:r+1]; R = rf[l:r+1]
            if len(lam) >= 5 and R.max() > R.min():
                a, b, _c = np.polyfit(HC / lam, R, 2)
                if a < -1e-9:
                    lam_v = HC / (-b / (2 * a))
                    if abs(lam_v - wf[i0]) <= (wf[r] - wf[l]):
                        return float(lam_v)
            if 0 < i0 < n - 1:
                ym, y0, yp = rf[i0-1], rf[i0], rf[i0+1]
                den = ym - 2 * y0 + yp
                if den < -1e-12:
                    dl = 0.5 * (ym - yp) / den
                    if -1.0 < dl < 1.0:
                        return float(wf[i0]) + dl * 0.5 * (wf[i0+1] - wf[i0-1])
            return float(wf[i0])

        # 1. candidate maxima from the DATA (position independent of the scaffold)
        idx, _props = find_peaks(rf, distance=8, height=1.0, prominence=prom, width=1)
        cand = np.array([_refine_idx(int(i)) for i in idx])
        cand = cand[cand > wl_min]
        if cand.size == 0:
            return None, None

        # 2. Bragg orders (integer crossings of g(λ), monotone decreasing) in range
        lam_lo = max(float(wf.min()), wl_min)
        lam_hi = float(wf.max())
        lam_g = np.linspace(lam_lo, lam_hi, 4000)
        g = 2.0 * d * np.sqrt(np.clip(n_of(lam_g) ** 2 - s2, 1e-6, None)) / lam_g
        orders = []
        for m in range(int(np.floor(g.min())) + 1, int(np.ceil(g.max())) + 1):
            dif = g - m
            for i in np.where(np.diff(np.sign(dif)) != 0)[0]:
                lam_m = lam_g[i] + (lam_g[i+1] - lam_g[i]) * (0 - dif[i]) / (dif[i+1] - dif[i])
                orders.append((int(m), float(lam_m)))
        if not orders:
            return None, None
        orders.sort(key=lambda t: t[1])
        lam_ord = np.array([l for _, l in orders])

        # 3. associate each order to the nearest candidate within frac·(neighbour gap)
        assoc = []
        for j, (m, lm) in enumerate(orders):
            neigh = np.inf
            if j > 0:
                neigh = min(neigh, lm - lam_ord[j-1])
            if j < len(lam_ord) - 1:
                neigh = min(neigh, lam_ord[j+1] - lm)
            tol = frac * neigh
            k = int(np.argmin(np.abs(cand - lm)))
            if abs(cand[k] - lm) <= tol:
                assoc.append((m, float(cand[k])))
        if len(assoc) < 2:
            return None, None

        # 4a. physical guard: the implied n(λ_m)=√((m·λ/2d)²+sin²θ) must be in a
        # physical range. A phantom order (from a poor scaffold n → a non-monotone
        # g(λ) with a spurious crossing) can grab edge noise and yield an absurd n
        # (e.g. n≈10 at 2437 nm on a bad 60° fit); drop it before it corrupts n(λ).
        assoc = [(m, l) for (m, l) in assoc
                 if 1.05 <= np.sqrt((m * l / (2.0 * d)) ** 2 + s2) <= 3.2]
        if len(assoc) < 2:
            return None, None

        # 4b. n(λ)-smoothness guard: drop the survivor whose inclusion breaks the
        # regularity of n(λ) (a mis-associated order), at most a couple of times.
        assoc.sort(key=lambda t: t[1])   # ascending λ
        for _ in range(2):
            if len(assoc) < 3:
                break
            lam = np.array([p for _, p in assoc])
            mm = np.array([m for m, _ in assoc], float)
            n_here = np.sqrt((mm * lam / (2.0 * d)) ** 2 + s2)
            d2 = np.abs(n_here[:-2] - 2.0 * n_here[1:-1] + n_here[2:])
            j = int(np.argmax(d2))
            if d2[j] <= smooth_tol:
                break
            drop = j + 1   # middle point of the worst triple
            if self.on_progress is not None:
                self.on_progress(f"[maxima {ang}°] scartato λ={lam[drop]:.0f} nm "
                                 f"(rompe la regolarità di n)")
            assoc.pop(drop)
        if len(assoc) < 2:
            return None, None

        assoc.sort(key=lambda t: t[1], reverse=True)   # descending λ (storage convention)
        peaks_wl = np.array([p for _, p in assoc])
        m_values = np.array([m for m, _ in assoc])
        return peaks_wl, m_values

    def _robust_m_start(self, peaks, lo=1000.0):
        """Robust m_start for the longest-λ peak.

        The simple method uses ONLY the two longest-λ peaks:
        m0 = round(λ2/(λ1−λ2)). Robust for low m, but for high m or noisy
        positions a single imprecise peak is enough for m0 to jump by 1, falsifying
        the whole n(λ) and d.

        m·λ-regular criterion: on the IR peaks (λ>lo) n≈const, so
        m·λ = 2·n·d·cosθ_r ≈ CONSTANT. A wrong m by 1 adds a λ ramp
        (m·λ → m·λ + λ), blowing up the spread of m·λ. We try m0±1 and keep the
        one that minimizes the spread of m·λ on the IR peaks.

        Returns m_start (≥1) or None if <2 valid peaks (the caller handles the
        fallback, e.g. thin_film m=1). Does NOT touch the outlier (MAD) rejection
        of the d computation, which stays downstream.
        """
        p = np.sort(np.asarray(peaks, dtype=float))[::-1]   # decreasing λ
        if len(p) < 2 or p[0] <= p[1]:
            return None
        m0 = max(int(np.round(p[1] / (p[0] - p[1]))), 1)
        ir = p > lo
        if int(ir.sum()) < 3:        # too few IR peaks for the robust check
            return m0
        best = (m0, np.inf)
        for ms in (m0 - 1, m0, m0 + 1):
            if ms < 1:
                continue
            mlam_ir = ((ms + np.arange(len(p))) * p)[ir]   # m·λ on the IR peaks
            spread = float(np.std(mlam_ir))                # must be ~constant
            if spread < best[1]:
                best = (ms, spread)
        return best[0]

    def calcola_d_da_picchi(self, source=None, aggregation='weighted_median'):
        """Unified method to compute d from the 8°/20° peaks.

        Called both by `calcola_d_interno` (fit_auto, with cache) and by
        `peak_window._calcola_nd` (user, on-demand). Guarantees consistency:
        same source → same value of d.

        Args:
            source: None (reads state.d_source) or explicit 'exp'/'fit'/'hybrid'.
            aggregation: 'weighted_median' (default, weights 1/(m-2)) or 'mean'.

        Internal strategy:
        - Peak extraction via _extract_peaks(ang, source)
        - NIR filter with an adaptive threshold (thin_film → 600 nm, else 1000)
        - m from the fit using n(λ) from ext_override, fallback to the round formula
        - THIN-FILM branch: if state.thin_film AND a valid ext_override, uses the
          SINGLE-ANGLE formula `d = m·λ/(2·n(λ)·cos(θ_refr))` instead of the 8°/20°
          triangulation. On thin films Δλ(8°,20°) is small and amplifies errors.
        - MAD outlier rejection (≥3 pairs/peaks)
        - Final aggregation weighted_median or mean

        Returns: float (nm) or None if not computable.
        """
        state = self.state
        if source is None:
            source = getattr(state, 'd_source', 'exp')
        if source not in ('exp', 'fit', 'hybrid'):
            source = 'exp'
        # Mapping to the name accepted by _extract_peaks
        source_map = {'exp': 'exp_only', 'fit': 'fit_only', 'hybrid': 'hybrid'}
        src_extract = source_map[source]

        peaks_8  = self._extract_peaks(8,  source=src_extract)
        peaks_20 = self._extract_peaks(20, source=src_extract)
        if peaks_8 is None or peaks_20 is None:
            return None

        sin8_sq  = np.sin(np.radians(8))**2
        sin20_sq = np.sin(np.radians(20))**2

        # Adaptive λ threshold
        wl_thresh = 600.0 if state.thin_film else 1000.0
        peaks_8_NIR  = peaks_8[peaks_8 > wl_thresh]
        peaks_20_NIR = peaks_20[peaks_20 > wl_thresh]

        # m from the fit (with fallback to the round formula)
        ov = state.ext_override
        has_fit_n = ov is not None and len(ov) >= 3 and len(ov[0]) >= 2
        d_fit_current = float(state.d)

        def m_for_lambda(lam):
            if not has_fit_n:
                return None
            n_lam = self._n_at_lambda(lam)
            if n_lam is None or n_lam**2 <= sin8_sq:
                return None
            m_real = 2.0 * d_fit_current * np.sqrt(n_lam**2 - sin8_sq) / lam
            m = int(round(m_real))
            return m if m >= 1 else None

        # ── THIN-FILM branch: single-angle formula
        # Stable on thin films where Δλ(8°,20°) is small. Requires a valid
        # ext_override (for n(λ)).
        if state.thin_film and has_fit_n and len(peaks_8_NIR) >= 1:
            d_values = []
            for wl8 in np.sort(peaks_8_NIR)[::-1]:
                m = m_for_lambda(wl8)
                if m is None:
                    m = 1  # m=1 fallback for thin_film without m_for_lambda
                n_lam = self._n_at_lambda(wl8)
                cos_refr = np.sqrt(1.0 - sin8_sq / (n_lam ** 2))
                d_calc = m * wl8 / (2.0 * n_lam * cos_refr)
                d_values.append((d_calc, float(m)))
            if d_values:
                d_result = self._aggregate_d_values(d_values, aggregation)
                if self.on_progress is not None:
                    pairs_log = "; ".join(
                        f"m={v[1]:.0f} d={v[0]:.0f}" for v in d_values)
                    self.on_progress(
                        f"d_calc thin_film [single-angle, source={source}]: "
                        f"[{pairs_log}] → {aggregation} = {d_result:.0f} nm")
                return d_result

        # ── STANDARD BRANCH: 8°/20° triangulation ──
        # m_start fallback if ext_override is not available
        m_start_fallback = None
        if not has_fit_n:
            # Robust m_start (IR m·λ-regular criterion) instead of just the round
            # on the 2 longest-λ peaks. See _robust_m_start.
            m_start_fallback = self._robust_m_start(peaks_8, lo=1000.0)
            if m_start_fallback is None:
                if state.thin_film and len(np.sort(peaks_8)[::-1]) >= 1:
                    m_start_fallback = 1
                else:
                    return None

        d_values = []
        peaks_8_sorted = np.sort(peaks_8)[::-1]
        for i, wl8 in enumerate(peaks_8_sorted):
            if wl8 <= wl_thresh:
                continue
            diffs = np.abs(peaks_20 - wl8)
            idx20 = np.argmin(diffs)
            if diffs[idx20] >= 60:
                continue
            wl20 = peaks_20[idx20]
            if wl20 >= wl8:
                continue
            if has_fit_n:
                m = m_for_lambda(wl8)
                if m is None:
                    continue
            else:
                m = m_start_fallback + i
            d_calc = (m / 2.0) * np.sqrt(
                (wl8**2 - wl20**2) / (sin20_sq - sin8_sq))
            d_values.append((d_calc, float(m)))

        # thin_film fallback without fit_n: 1 pair with m=1 forced
        if not d_values and state.thin_film and not has_fit_n:
            for i, wl8 in enumerate(peaks_8_sorted):
                if wl8 <= wl_thresh:
                    continue
                diffs = np.abs(peaks_20 - wl8)
                idx20 = np.argmin(diffs)
                if diffs[idx20] >= 60:
                    continue
                wl20 = peaks_20[idx20]
                if wl20 >= wl8:
                    continue
                d_calc = 0.5 * np.sqrt(
                    (wl8**2 - wl20**2) / (sin20_sq - sin8_sq))
                d_values.append((d_calc, 1.0))
                break

        if not d_values:
            return None

        d_result = self._aggregate_d_values(d_values, aggregation)
        if self.on_progress is not None:
            src_label = f"m from {'fit n(λ)' if has_fit_n else 'round formula'}"
            thin_tag = " [thin_film]" if state.thin_film else ""
            pairs_log = "; ".join(f"m={v[1]:.0f} d={v[0]:.0f}" for v in d_values)
            self.on_progress(
                f"d_calc triangulation [{src_label}, source={source}]: "
                f"[{pairs_log}] → {aggregation} = {d_result:.0f} nm{thin_tag}")
        return d_result

    def _aggregate_d_values(self, d_values, aggregation):
        """Aggregation of a list of (d_calc, m) tuples.

        - 'weighted_median': weights 1/(m-2), floor 0.5. Low m more precise
          (peaks more separated in λ → smaller relative error).
        - 'mean': simple mean.
        Both preceded by MAD outlier rejection if ≥3 values.
        """
        d_arr = np.array([v[0] for v in d_values])
        m_arr = np.array([v[1] for v in d_values])
        # MAD outlier rejection
        if len(d_arr) >= 3:
            med = np.median(d_arr)
            mad = np.median(np.abs(d_arr - med))
            if mad > 0:
                keep = np.abs(d_arr - med) <= 2.0 * 1.4826 * mad
                if keep.sum() >= 1:
                    d_arr = d_arr[keep]
                    m_arr = m_arr[keep]
        if aggregation == 'mean':
            return float(np.mean(d_arr))
        # weighted_median (default)
        weights = 1.0 / np.maximum(m_arr - 2.0, 0.5)
        idx_sort = np.argsort(d_arr)
        v_sort = d_arr[idx_sort]
        w_sort = weights[idx_sort]
        cum_w = np.cumsum(w_sort)
        half = cum_w[-1] / 2.0
        j = int(np.searchsorted(cum_w, half))
        j = min(j, len(v_sort) - 1)
        return float(v_sort[j])

    def calcola_d_interno(self):
        """Compute d from the 8°/20° peaks (thin wrapper around calcola_d_da_picchi).

        - Adds the run_fit_auto cache (d_target stable for the whole run)
        - Reads `state.d_source` for the source/aggregation choice
        - Returns the result (with caching)

        The computation logic (m from fit, MAD, weighted median, thin_film branch
        with the single-angle formula) is centralized in `calcola_d_da_picchi` for
        consistency with peak_window.

        Returns: float (nm) or None.
        """
        if self._d_peaks_cache is not None:
            return self._d_peaks_cache
        d_result = self.calcola_d_da_picchi(
            source=getattr(self.state, 'd_source', 'exp'),
            aggregation='weighted_median')
        if d_result is not None:
            self._d_peaks_cache = d_result
        return d_result

    # ─────────────────────────────────────────────────────────────────────────
    # PARAMETER SAVING / LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def save_params(self, folder: str, sample_name: str,
                    wl_exp: np.ndarray, r_exp: np.ndarray,
                    R_fit: np.ndarray, n_wl: np.ndarray, k_wl: np.ndarray,
                    nodes_wl: np.ndarray, curr_n: np.ndarray, curr_k: np.ndarray):
        """Save the fit parameters and the complete data to disk.

        Args:
            folder:      output folder
            sample_name: sample name (used for the file names)
            wl_exp:      experimental wavelengths
            r_exp:       experimental reflectance
            R_fit:       computed reflectance
            n_wl, k_wl:  n and k interpolated on the experimental grid
            nodes_wl:    node positions
            curr_n, curr_k: n, k values at the nodes
        Returns:
            (par_path, dati_path) paths of the saved files
        """
        state = self.state
        os.makedirs(folder, exist_ok=True)
        nome = sample_name.replace(' ', '_')

        # .par file
        par_path = os.path.join(folder, f"{nome}.par")
        with open(par_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['# Parametri fit OPHIRA'])
            w.writerow(['d_nm', f'{state.d:.2f}'])
            w.writerow(['scatt', f'{state.scatt:.4f}'])
            w.writerow(['inhom', f'{state.inhom:.4f}'])
            w.writerow(['offs',  f'{state.offs:.4f}'])
            w.writerow(['km', f'{state.km:.4f}'])
            w.writerow(['ks', f'{state.ks:.4f}'])
            w.writerow(['kb', f'{state.kb:.4f}'])
            w.writerow(['pol', f'{state.pol:.4f}'])
            w.writerow(['dn',  f'{state.dn:.4f}'])
            w.writerow(['w300', f'{state.w300:.2f}'])
            w.writerow(['w335', f'{state.w335:.2f}'])
            w.writerow(['w375', f'{state.w375:.2f}'])
            w.writerow(['w400', f'{state.w400:.2f}'])
            w.writerow(['substrate', state.substrate_name])
            w.writerow(['film', state.film_name])
            w.writerow(['# nodi_wl', 'n', 'k'])
            for wl, n, k in zip(nodes_wl, curr_n, curr_k):
                w.writerow([f'{wl:.1f}', f'{n:.6f}', f'{k:.6f}'])

        # Complete-data file
        dati_path = os.path.join(folder, f"{nome}_dati_completi.txt")
        kk = state.kk_cache
        if kk is not None:
            wl_kk, n_kk_arr, k_kk_arr = kk
            n_kk_i = np.interp(wl_exp, wl_kk, n_kk_arr)
            k_kk_i = np.interp(wl_exp, wl_kk, k_kk_arr)
            header = ("lambda(nm)\tR_exp(%)\tR_fit(%)\tn\tk\tn_KK\tk_KK")
            mat = np.column_stack([wl_exp, r_exp, R_fit, n_wl, k_wl,
                                   n_kk_i, k_kk_i])
            fmt = ['%.1f','%.4f','%.4f','%.6f','%.6f','%.6f','%.6f']
        else:
            header = ("lambda(nm)\tR_exp(%)\tR_fit(%)\tn\tk")
            mat = np.column_stack([wl_exp, r_exp, R_fit, n_wl, k_wl])
            fmt = ['%.1f','%.4f','%.4f','%.6f','%.6f']

        np.savetxt(dati_path, mat, header=header, delimiter='\t',
                   fmt=fmt, comments='# ')

        return par_path, dati_path

    def save_session(self, path: str):
        """Serialize the complete state to a pickle file."""
        import pickle
        import copy
        session = self.state.to_dict()
        with open(path, 'wb') as f:
            pickle.dump(session, f)

    def _become(self, new_state):
        """Make the live AppState hold EXACTLY new_state's contents.

        Every window keeps a reference to the one AppState object, so it cannot be
        swapped — its contents are replaced instead. Replacing the whole __dict__
        (rather than writing the attributes one by one) is what makes anything
        attached to the state from outside __init__ disappear as it should: those
        attributes are absent from the new state, so an attribute-wise copy would
        leave the previous sample's values in place.
        """
        d = dict(vars(new_state))
        self.state.__dict__.clear()
        self.state.__dict__.update(d)

    def load_session(self, path: str):
        """Load a session from a pickle file. Returns True if OK."""
        import pickle
        try:
            with open(path, 'rb') as f:
                session = pickle.load(f)
            new_state = AppState.from_dict(session)
            # _become replaces the whole state, so attributes attached outside
            # AppState.__init__ — kk_metric (the KK window's figure of merit) and
            # _last_sigma_k/_last_sigma_ext (which feed the KK propagation) — do
            # not carry over from the previous session; an attribute-wise copy
            # would leave them in place.
            self._become(new_state)
            return True
        except Exception as e:
            print(f"[load_session] Error: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Back to the startup state — the reset button of any instrument.

        Nothing is carried over, materials included. Selecting a known film from
        the menu loads its tabulated n/k and locks the sliders; reset clears both
        together so the film combo, slider values and locks stay consistent.
        Returns: the same AppState (the UI must refresh sliders, plots, toggles
        and the material combos).
        """
        self._become(AppState())
        return self.state

    # ─────────────────────────────────────────────────────────────────────────
    # SPECTRAL FILE PARSING (delegated to physics)
    # ─────────────────────────────────────────────────────────────────────────

    def load_spectral_file(self, path: str, ang: int):
        """Load a spectral file and update state.data_angles[ang]['exp'].

        Returns: (wl, R) or None on error.
        """
        result = ph._parse_spectral_file(path)
        if result is None:
            return None
        wl, R = result
        # COMPLETE reset of the angle's state.
        # Loading a new spectrum invalidates everything about the previous
        # sample's fit for this angle. Without the reset, the stale fields
        # contaminate:
        #   - sliders (km/ks/kb) → _extract_peaks de-corrected with the old
        #     sample → shifted peaks → wrong/oscillating d_from_peaks
        #   - fit → the previous sample's simulated curve stays on screen
        #   - ext_override, peaks_wl, m_values, sigma_nk → stale orders/curves
        # I rebuild the dict from scratch so that no key — not even the dynamic
        # ones (peaks_wl/m_values/sigma_nk) — survives the sample change. Through
        # AppState.new_angle_dict, which is the ONE definition of the record.
        self.state.data_angles[ang] = AppState.new_angle_dict(np.array([wl, R]))
        self._invalidate_kk(ang)
        self.state.ang_filenames[ang] = os.path.basename(path)
        return wl, R

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE FIT (run_fit)
    # ─────────────────────────────────────────────────────────────────────────

    def run_fit(self, ang: int, wl_exp: np.ndarray, r_exp: np.ndarray) -> dict:
        """Fit angle `ang` on (wl_exp, r_exp).
        Returns a dict with R, nf, kf, en, ek, ext, chi2, p, extra, sigma_n, sigma_k.
        """
        state = self.state
        S = AppState.HOBL_ANGLES
        state.stop = False; state.last_p = None; state.iter_count = 0

        # Peak-centering penalty: extract the exp peaks just once at the start of
        # the fit + filter the exclusion bands (detector swap).
        try:
            _peaks_exp_pos = self._extract_peaks(ang, source='exp_only')
            _peaks_exp_pos = self._filter_peaks_excluded(_peaks_exp_pos)
        except Exception:
            _peaks_exp_pos = None

        c_nodes_0 = ph.build_c_nodes(state)
        ext_nodes_0 = ph._build_ext_nodes(c_nodes_0)
        N = len(ext_nodes_0)

        n_ext0, k_ext0 = ph._init_ext_nk(state, c_nodes_0, ext_nodes_0)

        IDX_N=slice(1,1+N); IDX_K=slice(1+N,1+2*N)
        IDX_SC=1+2*N; IDX_OF=2+2*N; IDX_IH=3+2*N

        # For known films: overwrite n_ext0/k_ext0 with tabulated values
        # and widen the d bounds (the initial thickness can be far from the true one)
        film_idx = ph._film_mat[0]
        if film_idx > 0:
            film_name = ph._FILM_MATS[film_idx]
            nk_tab = ph._nk_film_known(ext_nodes_0, film_name)
            if nk_tab is not None:
                n_ext0 = nk_tab.real.copy()
                k_ext0 = np.maximum(nk_tab.imag, 0.0).copy()

        if ang in S:
            d_val=state.d
            # For known films widen the d bounds: ±50% instead of ±0.5%
            if film_idx > 0:
                d_bds=(max(1.0, d_val*0.5), d_val*1.5)
            else:
                d_bds=(d_val*0.995, d_val*1.005)
            n_bds=[ph._node_bounds(w)[0] for w in ext_nodes_0]
            k_bds=[ph._k_bounds(w, state.k_bound) for w in ext_nodes_0]
            scatt_bds=(max(0,state.scatt*0.98), min(300,state.scatt*1.02))
            opt_thick_ref=np.mean(n_ext0)*d_val
            p0=np.array([state.d]+list(n_ext0)+list(k_ext0)+
                [state.scatt,state.offs,state.inhom,state.km,state.ks,state.kb,state.pol,
                 state.dn,state.pm,0.5,0.25,0.25,1.0,
                 state.w300,state.w335,state.w375,state.w400,state.w1100])
            bounds=([d_bds]+n_bds+k_bds+
                [scatt_bds,(0,1.5),(0,15),(0.2,2.0),(-4.0,4.0),(0.0,3.0),(0.0,1.0),
                 (-0.35,0.35),(-3.1416,3.1416),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.2,2.5),
                 (280,320),(315,355),(350,400),(390,450),(950,1200)])
            IDX_KM=4+2*N; IDX_KS=5+2*N; IDX_KB=6+2*N; IDX_POL=7+2*N
            IDX_DN=8+2*N; IDX_PM=9+2*N
            IDX_WC=10+2*N; IDX_WL=11+2*N; IDX_WR=12+2*N; IDX_DA=13+2*N
        else:
            d_bds=(10,6000); n_bds=[ph._node_bounds(w)[0] for w in ext_nodes_0]
            k_bds=[ph._k_bounds(w, state.k_bound) for w in ext_nodes_0]
            scatt_bds=(0,300); opt_thick_ref=None
            p0=np.array([state.d]+list(n_ext0)+list(k_ext0)+
                [state.scatt,state.offs,state.inhom,state.km,state.ks,state.kb,state.pol,
                 state.dn,state.pm,
                 state.w300,state.w335,state.w375,state.w400,state.w1100])
            bounds=([d_bds]+n_bds+k_bds+
                [scatt_bds,(0,1.5),(0,15),(0.2,2.0),(-4.0,4.0),(0.0,3.0),(0.0,1.0),
                 (-0.35,0.35),(-3.1416,3.1416),
                 (280,320),(315,355),(350,400),(390,450),(950,1200)])
            IDX_KM=4+2*N; IDX_KS=5+2*N; IDX_KB=6+2*N; IDX_POL=7+2*N
            IDX_DN=8+2*N; IDX_PM=9+2*N

        bounds=list(bounds)
        # The 3 slots pol/dn/pm = coeff. of the FRINGE-CONTRAST (see physics._COLLAPSE_POL):
        # fc0(level)/fc1(slope)/fc2(curv). At 8/20 the contrast is good → neutral
        # (1,0,0). At the HOBL angles free (fc range), lockable via the UI (these lines
        # come BEFORE the locks below, so a lock can re-pin them). ps1 inactive.
        state.ps1 = 0.0
        if ang in S:
            bounds[IDX_POL]=(0.0,3.0); bounds[IDX_DN]=(-2.0,2.0); bounds[IDX_PM]=(-2.0,2.0)
        else:
            state.pol, state.dn, state.pm = 1.0, 0.0, 0.0
            p0[IDX_POL]=1.0; p0[IDX_DN]=0.0; p0[IDX_PM]=0.0
            bounds[IDX_POL]=(1.0,1.0); bounds[IDX_DN]=(0.0,0.0); bounds[IDX_PM]=(0.0,0.0)
        if state.lock_left[0]: bounds[0]=(state.d,state.d)
        ph._apply_lock_nk_bounds(bounds, ext_nodes_0, c_nodes_0, n_ext0, k_ext0, state, N)
        # polariz is NOT a free parameter at 8°/20° (S = HOBL_ANGLES = {40,60}).
        # At those angles p and s are practically equivalent for unpolarized reflectance.
        if ang not in S:
            bounds[IDX_POL] = (state.pol, state.pol)
        lmap={1:(IDX_SC,state.scatt),2:(IDX_IH,state.inhom),3:(IDX_OF,state.offs),
              4:(IDX_KM,state.km),5:(IDX_KS,state.ks),6:(IDX_KB,state.kb),12:(IDX_POL,state.pol)}
        for li,(idx,v) in lmap.items():
            if state.lock_left[li]: bounds[idx]=(v,v)
        if ang in S:
            if state.lock_extra[0]: bounds[IDX_DN]=(state.dn,state.dn)
            if state.lock_extra[1]: bounds[IDX_PM]=(state.pm,state.pm)
            # IR anchoring: km/ks/kb = spectrometer constant →
            # at the HOBL angles locked to the state values (from the robust 8/20
            # fits), not refitted. Breaks the position degeneracy with Δn.
            if getattr(self, 'anchor_ir_hobl', False):
                bounds[IDX_KM]=(state.km,state.km)
                bounds[IDX_KS]=(state.ks,state.ks)
                bounds[IDX_KB]=(state.kb,state.kb)
        # Indices for W300/W335/W375/W400/W1100
        if state.lock_left[7]:  bounds[-5]=(state.w300,state.w300)
        if state.lock_left[8]:  bounds[-4]=(state.w335,state.w335)
        if state.lock_left[9]:  bounds[-3]=(state.w375,state.w375)
        if state.lock_left[10]: bounds[-2]=(state.w400,state.w400)
        if state.lock_left[11]: bounds[-1]=(state.w1100,state.w1100)
        # If known film (not PSi), pin n,k to the tabulated values + nominal W grid
        ph._lock_film_known_bounds(bounds, ext_nodes_0, N, state=state, fix_w_at_end=True)
        # (fringe-contrast collapse already set above, after bounds=list)
        bnd=np.array(bounds); p0=np.clip(p0,bnd[:,0],bnd[:,1])

        _best={'R':None,'nf':None,'kf':None,'en':None,'ek':None,
               'ext':None,'extra':{},'chi2':np.inf,'p':None}
        _eval_count=[0]
        _last_ui_time=[0.0]
        _UI_INTERVAL=0.5   # minimum seconds between UI updates

        def obj(p):
            if state.stop: state.last_p=p.copy(); raise StopIteration
            _eval_count[0]+=1
            # Update the UI only if enough time has passed since the last update
            if self.on_fit_step and _best['R'] is not None:
                import time as _time
                now=_time.time()
                if now-_last_ui_time[0]>=_UI_INTERVAL:
                    _last_ui_time[0]=now
                    self.on_fit_step(_best['chi2'],state.iter_count,
                                     _best['R'],_best['nf'],_best['kf'],
                                     _best['ext'],_best['en'],_best['ek'],ang)
            w300,w335,w375,w400,w1100=p[-5],p[-4],p[-3],p[-2],p[-1]
            c_nd=ph._apply_w_nodes(c_nodes_0,w300,w335,w375,w400,w1100)
            # Known film: use a fixed nominal grid to avoid a mismatch between n_p
            # (computed on ext_nodes_0) and ext_nd (computed from the mobile c_nd)
            if ph._film_mat[0] > 0:
                ext_nd = ext_nodes_0
            else:
                ext_nd=ph._build_ext_nodes(c_nd)
            n_p=np.clip(p[IDX_N],1.0,12.0); k_p=np.clip(p[IDX_K],0.0,15.0)
            chi2=0.0
            if ang in S:
                km_p,ks_p,kb_p,pol_p=p[IDX_KM],p[IDX_KS],p[IDX_KB],p[IDX_POL]
                dn_p,pm_p=p[IDX_DN],p[IDX_PM]
                wc,wlw,wr,da=p[IDX_WC],p[IDX_WL],p[IDX_WR],p[IDX_DA]
                reg_n_w,reg_k_w,mw=50,30,10**9
                _kw=dict(scatt=p[IDX_SC],offset=p[IDX_OF],inhom=p[IDX_IH],
                         kjump=km_p,alpha=ks_p,beta=kb_p,pol=pol_p,
                         delta_n=dn_p,phi_mix=pm_p,phi_s1=state.ps1)
                R,nf,kf=ph.calc_refl_3angle(wl_exp,p[0],ext_nd,n_p,k_p,
                                            float(ang),da,wc,wlw,wr,**_kw)
            else:
                km_p,ks_p,kb_p,pol_p=p[IDX_KM],p[IDX_KS],p[IDX_KB],p[IDX_POL]
                dn_p,pm_p=state.dn,state.pm; reg_n_w,reg_k_w=100,50
                mw=10**6
                R,nf,kf=ph.calculate_refl_core(wl_exp,p[0],ext_nd,n_p,k_p,p[IDX_SC],p[IDX_OF],p[IDX_IH],kjump=km_p,alpha=ks_p,beta=kb_p,pol=pol_p,theta_deg=ang,delta_n=dn_p,phi_mix=pm_p,phi_s1=state.ps1)
            w=np.ones_like(wl_exp)
            if ang in S:
                lmin=state.lmin; w[wl_exp<lmin]=0.0
                w[(wl_exp>=400)&(wl_exp<600)]=15.0
                w[(wl_exp>=600)&(wl_exp<=1300)]=40.0; w[wl_exp>1500]=0.5
                muv=wl_exp<min(500.0,lmin)
                if np.any(muv): chi2+=40.0*np.sum((R[muv]-r_exp[muv])**2)
                dif=(np.mean(n_p)*p[0]-opt_thick_ref)/opt_thick_ref
                chi2+=np.sum(np.where(np.abs(dif)>0.01,10**9*dif**2,0))
            chi2+=np.sum(w*(R-r_exp)**2)
            # Peak-centering penalty: forces R to have a first derivative ≈0 and
            # concavity ≤0 at the exp peak positions.
            chi2 += self._peak_penalty(R, wl_exp, _peaks_exp_pos)
            # IR-peak position anchoring at 40° AND 60°: with the collapsed model
            # the 60° maxima are shifted, not deformed → the position penalty
            # applies there too.
            if ang in (40, 60):
                wpp = self.peak_pos_w_hi if ang == 60 else self.peak_pos_w
                chi2 += self._peak_pos_penalty(R, wl_exp, _peaks_exp_pos,
                                               lam_min=self.peak_pos_lam_min, w=wpp)
            reg=np.sum(np.diff(n_p,2)**2)*reg_n_w+np.sum(np.diff(k_p,2)**2)*reg_k_w
            if mw>0 and state.mono_n:
                mk=(ext_nd>=420)&(ext_nd<=2600); dn2=np.diff(n_p[mk])
                chi2+=np.sum(np.where(dn2>0,mw*dn2**2,0))
                # Convexity of n (penalize d²n<0), symmetric with k's below and at
                # ALL angles: n's convexity is normal dispersion — a property of the
                # material, not of the angle. On a nearly-flat, already-increasing
                # n(λ) this term only tidies the smoothness, so its effect at the
                # base angles is within measurement error.
                mcn=(ext_nd>=420)&(ext_nd<=2500); nc=n_p[mcn]
                if len(nc)>2:
                    d2n=np.diff(nc,n=2); chi2+=np.sum(np.where(d2n<0,mw*d2n**2,0))
                # ANTI-ZIGZAG — deliberately NOT masked, unlike the three terms
                # around it. The extra nodes are free to chase the data on their
                # own, and n and k can compensate each other, so without this
                # term they oscillate from node to node. This brakes that, and only
                # that: pn is the product of two consecutive slopes, so it is
                # negative exactly at a reversal, and the cost goes as slope⁴.
                # At a SMOOTH extremum the slopes across the apex pass through
                # zero → pn≈0 → it costs nothing: describing a real experimental
                # maximum is allowed. A zig-zag reverses at EVERY node with large
                # slopes → it pays at every one of them. That is the whole design:
                # a maximum yes, the zig-zag no.
                # Unmasked ON PURPOSE: below 420 nm the monotonicity above is off
                # (porous silicon has anomalous dispersion there), so this is the only
                # guard left against the nodes going wild exactly where the data
                # are weakest. Above 420 it is redundant (a reversal needs a
                # positive slope, already penalized at the same weight). Masking
                # it would re-open the door.
                d1n=np.diff(n_p); pn=d1n[:-1]*d1n[1:]
                chi2+=np.sum(np.where(pn<0,mw*pn**2,0))
            if mw>0 and state.mono_k:
                mk=(ext_nd>=420)&(ext_nd<=2600); dk2=np.diff(k_p[mk])
                chi2+=np.sum(np.where(dk2>0,mw*dk2**2,0))
                # Convexity of k, same sense as n (penalize d²k<0): decreasing and
                # CONVEX, i.e. flattening asymptotically towards k≈0 in the IR — the
                # shape an absorption tail actually has. Over 420-2500 k is already
                # ~0 except a short tail near 420.
                mck=(ext_nd>=420)&(ext_nd<=2500); kc=k_p[mck]
                if len(kc)>2:
                    d2k=np.diff(kc,2); chi2+=np.sum(np.where(d2k<0,mw*d2k**2,0))
                # ANTI-ZIGZAG on k — same term, same reason, unmasked on purpose:
                # see the n block above. It has to be on BOTH: n and k compensate
                # each other, so braking only one leaves the oscillation free to
                # move into the other.
                d1k=np.diff(k_p); pk=d1k[:-1]*d1k[1:]
                chi2+=np.sum(np.where(pk<0,mw*pk**2,0))
            # Extreme IR node floor (2550): PSi only (free n). See _ir_tail_floor_penalty.
            if ph._film_mat[0] == 0:
                chi2 += self._ir_tail_floor_penalty(n_p, ext_nd, mw)
            tot2=chi2+reg
            if tot2<_best['chi2']:
                _best['extra']=({'wc':p[IDX_WC],'wl':p[IDX_WL],'wr':p[IDX_WR],'da':p[IDX_DA]}
                                if ang in S else {})
                _best.update({'R':R.copy(),'nf':nf.copy(),'kf':kf.copy(),
                              'en':n_p.copy(),'ek':k_p.copy(),
                              'ext':ext_nd.copy(),'chi2':tot2,'p':p.copy()})
            state.last_p=p.copy(); return tot2

        def cb_fit(p):
            state.iter_count+=1
            if state.iter_limit is not None and state.iter_count>=state.iter_limit:
                state.stop=True
            if self.on_fit_step and _best['R'] is not None:
                self.on_fit_step(_best['chi2'],state.iter_count,
                                 _best['R'],_best['nf'],_best['kf'],
                                 _best['ext'],_best['en'],_best['ek'],ang)
            return bool(state.stop)

        try:
            mit=5000 if ang in S else 2000; ft=1e-18 if ang in S else 1e-12
            gt=1e-12 if ang in S else 1e-7; ep=1e-8 if ang in S else 1e-6
            res=minimize(obj,p0,method='L-BFGS-B',bounds=bounds,
                         callback=cb_fit,options={'maxiter':mit,'ftol':ft,'gtol':gt,'eps':ep})
            final_p=_best['p'] if _best['p'] is not None else res.x
        except (StopIteration,Exception):
            final_p=(_best['p'] if _best['p'] is not None else
                     (state.last_p if state.last_p is not None else p0))

        if not state.stop and ang in S and final_p is not None:
            try:
                state.iter_count=0
                # In the 2nd stage temporarily disable iter_limit
                # (the quick limit applies only to the 1st stage)
                saved_limit = state.iter_limit
                state.iter_limit = None
                state.stop = False
                if self.on_progress: self.on_progress('2nd stage...')
                p1=final_p.copy(); bnd2=list(bounds)
                bnd2[0]=(p1[0],p1[0])
                for j in range(1,1+2*N): bnd2[j]=(p1[j],p1[j])
                for idx in [IDX_SC,IDX_OF,IDX_IH]: bnd2[idx]=(p1[idx],p1[idx])
                # Pin all 5 W-nodes (w300,w335,w375,w400,w1100)
                for wi in range(1,6): bnd2[-wi]=(p1[-wi],p1[-wi])
                if state.lock_extra[0]: bnd2[IDX_DN]=(p1[IDX_DN],p1[IDX_DN])
                if state.lock_extra[1]: bnd2[IDX_PM]=(p1[IDX_PM],p1[IDX_PM])
                pm_starts=([p1[IDX_PM]] if state.lock_extra[1] else
                           np.linspace(-np.pi,np.pi,6,endpoint=False))
                best2={'chi2':_best['chi2'],'p':_best['p']}
                for pm_s in pm_starts:
                    if state.stop: break
                    pt=p1.copy(); pt[IDX_PM]=pm_s
                    try:
                        minimize(obj,pt,method='L-BFGS-B',bounds=bnd2,callback=cb_fit,
                                 options={'maxiter':800,'ftol':1e-18,'gtol':1e-12,'eps':1e-8})
                        if _best['chi2']<best2['chi2']:
                            best2={'chi2':_best['chi2'],'p':_best['p'].copy()}
                    except (StopIteration,Exception): pass
                if best2['p'] is not None:
                    _best.update({'chi2':best2['chi2'],'p':best2['p']}); final_p=best2['p']
            except (StopIteration,Exception): pass
            finally:
                state.iter_limit = saved_limit   # restore the original limit

        if final_p is None or _best['R'] is None: return {}

        w300f,w335f,w375f,w400f,w1100f=final_p[-5],final_p[-4],final_p[-3],final_p[-2],final_p[-1]
        cnf=ph._apply_w_nodes(c_nodes_0,w300f,w335f,w375f,w400f,w1100f)
        ext_nf=ph._build_ext_nodes(cnf)
        n_fin=np.clip(final_p[IDX_N],1.0,12.0); k_fin=np.clip(final_p[IDX_K],0.0,15.0)

        state.d=float(np.clip(final_p[0],10,6000))
        state.scatt=float(np.clip(final_p[IDX_SC],0,300))
        state.offs=float(np.clip(final_p[IDX_OF],0,20))
        state.inhom=float(np.clip(final_p[IDX_IH],0,15))
        state.km=float(np.clip(final_p[IDX_KM],0.2,2.0))
        state.ks=float(np.clip(final_p[IDX_KS],-4.0,4.0))
        state.kb=float(np.clip(final_p[IDX_KB],0.0,3.0))
        # pol/dn/pm are the fringe-contrast coeff. → clip to the fc ranges (0-3, ±2, ±2).
        state.pol=float(np.clip(final_p[IDX_POL],0.0,3.0))
        if ang in S:
            state.dn=float(np.clip(final_p[IDX_DN],-2.0,2.0))
            state.pm=float(np.clip(final_p[IDX_PM],-2.0,2.0))
        state.w300=w300f; state.w335=w335f; state.w375=w375f; state.w400=w400f; state.w1100=w1100f

        ext_ov=(ext_nf.copy(),n_fin.copy(),k_fin.copy(),_best['extra'].copy())
        state.ext_override=ext_ov
        state.suppress_override_clear=True
        for i in range(14):
            if not state.lock_n[i]:
                state.n_slider_vals[i]=ph.inv_n_map(np.clip(float(np.interp(cnf[i],ext_nf,n_fin)),1.0,12.0))
            if not state.lock_k[i]:
                state.k_slider_vals[i]=ph.inv_k_map(np.clip(float(np.interp(cnf[i],ext_nf,k_fin)),1e-4,15.0))
        state.suppress_override_clear=False

        if state.data_angles[ang]["exp"] is not None:
            _da=_best['extra'].get('da',0.0) if _best['extra'] else 0.0
            self._save_angle_state(ang, _best['R'], ext_ov,
                ang_params={"km":state.km,"ks":state.ks,"kb":state.kb,
                            "pol":state.pol,"dn":state.dn,"pm":state.pm}, da=_da)
            # Fields specific to run_fit (in addition to those saved by the helper):
            state.data_angles[ang].update({
                "km": state.km, "ks": state.ks, "final_p": final_p.copy(),
                "idxs": {"SC":IDX_SC,"OF":IDX_OF,"IH":IDX_IH,"KM":IDX_KM,
                         "KS":IDX_KS,"POL":IDX_POL,"DN":IDX_DN,"PM":IDX_PM,"KB":IDX_KB}})

        sigma_n = sigma_k = None
        if ph._film_mat[0] == 0:  # sigma meaningful only for PSi (free n/k)
            sigma_n, sigma_k = _compute_sigma_nodes(
                state, wl_exp, r_exp, _best['en'], _best['ek'], _best['ext'], ang,
                extra=_best['extra'])

        if self.on_progress:
            # `fit obj`, the same name and the same number the plot shows: the value
            # L-BFGS-B actually minimizes — data residuals plus regularization,
            # monotonicity, anti-zigzag and the peak penalties. It is NOT a chi²:
            # the spectra carry no per-point sigma.
            self.on_progress(f"fit obj: {_best['chi2']:.2e}")
        return {'R':_best['R'],'nf':_best['nf'],'kf':_best['kf'],
                'en':_best['en'],'ek':_best['ek'],'ext':_best['ext'],
                'chi2':_best['chi2'],'p':final_p,'extra':_best['extra'],
                'ext_override':state.ext_override,
                'sliders':state.data_angles[ang].get('sliders'),
                'n_iter':state.iter_count,'sigma_n':sigma_n,'sigma_k':sigma_k}

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-ANGLE FIT (run_fit_multi)
    # ─────────────────────────────────────────────────────────────────────────

    def _invalidate_kk(self, ang):
        """The KK curve and its figure of merit come from the 8° fit. When the 8°
        fit or its spectrum changes they no longer describe the current state, so
        this clears kk_cache and kk_metric."""
        if ang == 8:
            self.state.kk_cache = None
            self.state.kk_metric = None

    def _save_angle_state(self, ang, R_fit, ext_override, ang_params, da=0.0):
        """Save the results of a fit for angle `ang` into state.data_angles[ang]:
        writes "fit", "ext_override", and the "sliders" dict (parameters to restore
        when switching to that angle).
        `ang_params` is the dict of the per-angle instrumental parameters
        (km, ks, kb, pol, dn, pm — usually pulled from that angle's optimization
        vector, in run_fit instead from state.*). `da` is the angular spread
        (0 for base angles, a value for HOBL). `ext_override` is the complete tuple
        (ext_nd, n, k, extra_dict)."""
        state = self.state
        sliders = {
            "d": state.d, "scatt": state.scatt, "inhom": state.inhom, "offs": state.offs,
            "ps1": state.ps1, "lmin": state.lmin,
            "w300": state.w300, "w335": state.w335, "w375": state.w375,
            "w400": state.w400, "w1100": state.w1100,
            "da": float(da),
        }
        for k, v in ang_params.items():
            sliders[k] = float(v)
        state.data_angles[ang]["fit"] = R_fit.copy()
        state.data_angles[ang]["ext_override"] = ext_override
        state.data_angles[ang]["sliders"] = sliders
        self._invalidate_kk(ang)

    def _restore_angle_state(self, ang: int):
        """Restore AppState from the parameters saved for angle `ang`.
        Respects the slider locks — locked parameters are not overwritten.
        """
        state = self.state
        state.current_ang = ang
        sl = state.data_angles[ang].get("sliders")
        if sl:
            # Map parameter → lock_left index (None = lock_extra)
            # left_attrs: d(0),scatt(1),inhom(2),offs(3),km(4),ks(5),kb(6),
            #             w300(7),w335(8),w375(9),w400(10),w1100(11),pol(12)
            # lock_extra: dn(0),pm(1),ps1(2)
            lock_map = {
                'd':    ('left', 0),  'scatt': ('left', 1),
                'inhom':('left', 2),  'offs':  ('left', 3),
                'km':   ('left', 4),  'ks':    ('left', 5),
                'kb':   ('left', 6),  'w300':  ('left', 7),
                'w335': ('left', 8),  'w375':  ('left', 9),
                'w400': ('left',10),  'w1100': ('left',11),
                'pol':  ('left',12),
                'dn':   ('extra',0),  'pm':    ('extra',1),
                'ps1':  ('extra',2),
            }
            def is_locked(param):
                if param not in lock_map: return False
                kind, idx = lock_map[param]
                if kind == 'left':
                    return idx < len(state.lock_left) and state.lock_left[idx]
                else:
                    return idx < len(state.lock_extra) and state.lock_extra[idx]

            for attr in ['d','scatt','inhom','offs','km','ks','kb',
                         'w300','w335','w375','w400','w1100','pol','dn','pm','ps1','lmin']:
                if attr in sl and not is_locked(attr):
                    setattr(state, attr, sl[attr])

        ov = state.data_angles[ang].get("ext_override")
        if ov is not None:
            state.ext_override = ov

    def run_fit_multi(self, ang_avail: list = None) -> dict:
        """Simultaneous 3-stage multi-angle fit.

        Args:
            ang_avail: list of angles to fit (default: all those with data)
        Returns:
            dict with results per angle, or {} if it fails
        """
        state = self.state
        if ang_avail is None:
            ang_avail = [a for a in [8,20,40,60]
                         if state.data_angles[a]["exp"] is not None]
        if len(ang_avail) < 2:
            if self.on_progress:
                self.on_progress(f"FIT MULTI: need ≥2 angles ({len(ang_avail)} found)")
            return {}

        state.stop = False
        # iter_limit can "stay stuck" from a previous fit_auto (fit_rapido sets it
        # to N_ITER_RAPIDO). If not reset, the final refinement run_fits
        # (8/20/40/60) reach the limit in phase 1 → state.stop=True → they SKIP the
        # 2nd phase (IR fringe centering with dn/pm/spread). Explicit reset here.
        state.iter_limit = None
        ang_base = [a for a in ang_avail if a in (8,20)]
        ang_hobl  = [a for a in ang_avail if a in (40,60)]

        # Per-angle MEAS JUMP: measures the detector jump from each angle's data
        # and will keep it fixed (see make_p0_ang + pinning in the two stages).
        km_meas = {}
        if getattr(self, 'measure_jump_multi', True):
            for a in ang_avail:
                ex = state.data_angles[a].get("exp")
                if ex is not None:
                    kmv = ph.measure_ir_jump(ex[0], ex[1])
                    if kmv is not None:
                        km_meas[a] = kmv
            if km_meas and self.on_progress:
                self.on_progress("MEAS JUMP/angle: " +
                    " ".join(f"{a}°={v:.3f}" for a, v in km_meas.items()))

        # Peak-centering penalty: extract the exp peaks for each available angle
        # just once (then used in obj_s1 + obj_s2).
        _peaks_exp = {}
        for _a in ang_avail:
            try:
                pks = self._extract_peaks(_a, source='exp_only')
                _peaks_exp[_a] = self._filter_peaks_excluded(pks)
            except Exception:
                _peaks_exp[_a] = None

        c_nodes_0 = ph.build_c_nodes(state)
        ext_nodes_0 = ph._build_ext_nodes(c_nodes_0)
        N = len(ext_nodes_0)

        n_ext0, k_ext0 = ph._init_ext_nk(state, c_nodes_0, ext_nodes_0)

        d_start=state.d
        dm=state.d_mem_global
        d_from_peaks=dm if (dm is not None and 300<dm<6000) else None
        if d_from_peaks: d_start=d_from_peaks

        I_D=0; I_N=slice(1,1+N); I_K=slice(1+N,1+2*N)
        I_SC=1+2*N; I_OF=2+2*N; I_IH=3+2*N
        I_W0=4+2*N; I_W1=5+2*N; I_W2=6+2*N; I_W3=7+2*N; I_W4=8+2*N; N_CAMP=9+2*N; N_ANG=10

        def make_p0_ang(a):
            sl=state.data_angles[a].get("sliders") or {}
            ov=state.data_angles[a].get("ext_override")
            ex=(ov[3] if ov and len(ov)==4 else {}) or {}
            km0 = km_meas.get(a, sl.get('km',state.km))   # per-angle MEAS JUMP
            return [km0,sl.get('ks',state.ks),
                    sl.get('kb',state.kb),sl.get('pol',state.pol),
                    sl.get('dn',0.0),sl.get('pm',0.0),
                    ex.get('wc',0.5),ex.get('wl',0.25),ex.get('wr',0.25),ex.get('da',1.0)]

        bnd_ang_base=[(0.2,2.0),(-4,4),(0,3),(0,1),(0,0),(0,0),(0,1),(0,1),(0,1),(0.2,3.0)]
        # pol/dn/pm = fringe-contrast coeff. (fc0/fc1/fc2) → fc RANGE (0-3, ±2, ±2),
        # wide enough to hold a dilated fc0 without clipping. Same ranges as in run_fit.
        bnd_ang_hobl =[(0.2,2.0),(-4,4),(0,3),(0,3),(-2,2),(-2,2),(0,1),(0,1),(0,1),(0.2,3.0)]

        def calc_R_base(p, a, ki, ext_nd, n_p, k_p):
            base=N_CAMP+ki*N_ANG
            km_p,ks_p,kb_p,pol_p=p[base],p[base+1],p[base+2],p[base+3]
            dn_p,pm_p=p[base+4],p[base+5]
            wl_a,r_a=state.data_angles[a]["exp"]
            R,_,_=ph.calculate_refl_core(wl_a,p[I_D],ext_nd,n_p,k_p,p[I_SC],p[I_OF],p[I_IH],
                kjump=km_p,alpha=ks_p,beta=kb_p,pol=pol_p,theta_deg=a,
                delta_n=dn_p,phi_mix=pm_p,phi_s1=state.ps1)
            return R,r_a

        _ang_w={8:3.0,20:2.0,40:1.0,60:1.0}

        def obj_s1(p):
            if state.stop: raise StopIteration
            c_nd=ph._apply_w_nodes(c_nodes_0,p[I_W0],p[I_W1],p[I_W2],p[I_W3],p[I_W4])
            ext_nd=ph._build_ext_nodes(c_nd)
            n_p=np.clip(p[I_N],1.0,12.0); k_p=np.clip(p[I_K],0.0,15.0)
            chi2=0.0
            for ki,a in enumerate(ang_base):
                R,r_a=calc_R_base(p,a,ki,ext_nd,n_p,k_p)
                wl_a=state.data_angles[a]["exp"][0]
                w=np.ones_like(r_a)
                lmin=state.lmin; w[wl_a<lmin]=0.0
                w[(wl_a>=400)&(wl_a<600)]=15.0
                w[(wl_a>=600)&(wl_a<=1300)]=40.0; w[wl_a>1500]=0.5
                chi2+=_ang_w.get(a,1.0)*float(np.sum(w*(R-r_a)**2))
                # Peak-centering penalty (for the multi-angle case too)
                chi2 += _ang_w.get(a,1.0) * self._peak_penalty(R, wl_a, _peaks_exp.get(a))
            chi2+=50*np.sum(np.diff(n_p,2)**2)+30*np.sum(np.diff(k_p,2)**2)
            # Same shape priors as run_fit.obj — including the ANTI-ZIGZAG terms,
            # unmasked on purpose (a smooth maximum costs ~0, a zig-zag pays at
            # every node): see the long comment in run_fit.obj. Keep the two in
            # step: they are the same physics, written twice.
            if state.mono_n:
                mk=(ext_nd>=420)&(ext_nd<=2600); dn2=np.diff(n_p[mk])
                chi2+=1e9*np.sum(dn2[dn2>0]**2)
                d1n=np.diff(n_p); pn=d1n[:-1]*d1n[1:]
                chi2+=np.sum(np.where(pn<0,1e9*pn**2,0))
            if state.mono_k:
                mk=(ext_nd>=420)&(ext_nd<=2600); dk2=np.diff(k_p[mk])
                chi2+=1e9*np.sum(dk2[dk2>0]**2)
                d1k=np.diff(k_p); pk=d1k[:-1]*d1k[1:]
                chi2+=np.sum(np.where(pk<0,1e9*pk**2,0))
            # Extreme IR node floor (2550): PSi only (free n). See _ir_tail_floor_penalty.
            if ph._film_mat[0] == 0:
                chi2 += self._ir_tail_floor_penalty(n_p, ext_nd, 1e9)
            if chi2<_bm['chi2']: _bm['chi2']=chi2; _bm['p']=p.copy()
            return chi2

        bnd_d=(d_from_peaks*0.98,d_from_peaks*1.02) if d_from_peaks else (max(10,d_start*0.90),min(6000,d_start*1.10))
        # Physical bounds for n and k (same k-bound rule as run_fit → the two fit
        # paths agree)
        n_bds_m=[(1.0, 6.5) for _ in ext_nodes_0]
        k_bds_m=[ph._k_bounds(w, state.k_bound) for w in ext_nodes_0]
        bnd_s1=np.array([bnd_d]+n_bds_m+k_bds_m+
            [(0,300),(0,1.5),(0,15),(280,320),(315,355),(350,400),(390,450),(950,1200)]+
            [bnd_ang_base[j] for _ in ang_base for j in range(N_ANG)])
        for li,(idx,v) in {0:(I_D,state.d),1:(I_SC,state.scatt),2:(I_IH,state.inhom),
                            3:(I_OF,state.offs),7:(I_W0,state.w300),8:(I_W1,state.w335),9:(I_W2,state.w375),10:(I_W3,state.w400),11:(I_W4,state.w1100)}.items():
            if state.lock_left[li]: bnd_s1[idx]=(v,v)
        # Lock n/k nodes (base + sub-nodes between locked base pairs)
        ph._apply_lock_nk_bounds(bnd_s1, ext_nodes_0, c_nodes_0, n_ext0, k_ext0, state, N)
        # Lock per-angle parameters: km(off0),ks(off1),kb(off2),pol(off3),dn(off4),pm(off5)
        # pol is at lock_left[12], km/ks/kb at lock_left[4/5/6]
        for ki, a in enumerate(ang_base):
            base = N_CAMP + ki * N_ANG
            p0a = make_p0_ang(a)
            for li, off in {4:0, 5:1, 6:2, 12:3}.items():
                if li < len(state.lock_left) and state.lock_left[li]:
                    bnd_s1[base+off] = (p0a[off], p0a[off])
            for lei, off in {0:4, 1:5}.items():
                if lei < len(state.lock_extra) and state.lock_extra[lei]:
                    bnd_s1[base+off] = (p0a[off], p0a[off])
            if a in km_meas:   # per-angle MEAS JUMP: km fixed = measured from the data
                bnd_s1[base+0] = (km_meas[a], km_meas[a])

        p0_s1=np.array([d_start]+list(n_ext0)+list(k_ext0)+
            [state.scatt,state.offs,state.inhom,state.w300,state.w335,state.w375,state.w400,state.w1100]+
            [v for a in ang_base for v in make_p0_ang(a)])
        # Known film: pin n,k to the tabulated values (W-nodes stay free: in multi
        # they are in an intermediate position of bnd_s1, not at the tail)
        ph._lock_film_known_bounds(bnd_s1, ext_nodes_0, N)
        p0_s1=np.clip(p0_s1,bnd_s1[:,0],bnd_s1[:,1])
        _bm={'chi2':np.inf,'p':None}

        def cb(p,stage=""):
            state.iter_count+=1
            if self.on_progress: self.on_progress(f"fit obj: {_bm['chi2']:.2e} ({stage})")
            return bool(state.stop)

        try:
            minimize(obj_s1,p0_s1,method='L-BFGS-B',bounds=bnd_s1,
                     callback=lambda p:cb(p,"stage1"),
                     options={'maxiter':10000,'ftol':1e-18,'gtol':1e-12})
        except StopIteration: pass
        except Exception as e: print(f"S1 err:{e}")

        if state.stop or _bm['p'] is None: return {}
        p1=_bm['p'].copy()
        c_nd1=ph._apply_w_nodes(c_nodes_0,p1[I_W0],p1[I_W1],p1[I_W2],p1[I_W3],p1[I_W4])
        ext_nd1=ph._build_ext_nodes(c_nd1)
        n_fin1=np.clip(p1[I_N],1.0,12.0); k_fin1=np.clip(p1[I_K],0.0,15.0)

        # Update the sample state from stage 1
        state.d=float(np.clip(p1[I_D],10,6000)); state.scatt=float(np.clip(p1[I_SC],0,300))
        state.offs=float(np.clip(p1[I_OF],0,20)); state.inhom=float(np.clip(p1[I_IH],0,15))
        if not state.lock_left[7]:  state.w300=float(p1[I_W0])
        if not state.lock_left[8]:  state.w335=float(p1[I_W1])
        if not state.lock_left[9]:  state.w375=float(p1[I_W2])
        if not state.lock_left[10]: state.w400=float(p1[I_W3])
        if not state.lock_left[11]: state.w1100=float(p1[I_W4])
        state.ext_override=(ext_nd1.copy(),n_fin1.copy(),k_fin1.copy(),{})
        for i in range(14):
            if not state.lock_n[i]:
                state.n_slider_vals[i]=ph.inv_n_map(np.clip(float(np.interp(c_nd1[i],ext_nd1,n_fin1)),1.0,12.0))
            if not state.lock_k[i]:
                state.k_slider_vals[i]=ph.inv_k_map(np.clip(float(np.interp(c_nd1[i],ext_nd1,k_fin1)),1e-4,15.0))

        if not ang_hobl:
            for ki,a in enumerate(ang_base):
                wl_a,r_a=state.data_angles[a]["exp"]
                base=N_CAMP+ki*N_ANG
                km_a,ks_a,kb_a,pol_a=p1[base],p1[base+1],p1[base+2],p1[base+3]
                dn_a,pm_a=p1[base+4],p1[base+5]
                R_fit,_,_=ph.calculate_refl_core(wl_a,p1[I_D],ext_nd1,n_fin1,k_fin1,
                    p1[I_SC],p1[I_OF],p1[I_IH],kjump=km_a,alpha=ks_a,beta=kb_a,
                    pol=pol_a,theta_deg=a,delta_n=dn_a,phi_mix=pm_a)
                # Save the per-angle parameters in the extra dict so _refresh_simulation
                # uses the optimal values and not the slider ones
                extra_a={'km':float(km_a),'ks':float(ks_a),'kb':float(kb_a),
                         'pol':float(pol_a),'dn':float(dn_a),'pm':float(pm_a)}
                self._save_angle_state(a, R_fit,
                    (ext_nd1.copy(),n_fin1.copy(),k_fin1.copy(),extra_a),
                    ang_params={"km":km_a,"ks":ks_a,"kb":kb_a,
                                "pol":pol_a,"dn":dn_a,"pm":pm_a}, da=0.0)
            # Final single fits
            saved_ang=state.current_ang
            _km_lock_saved = state.lock_left[4]   # per-angle MEAS JUMP: see above
            for ang_s,nrep in [(8,2),(20,2)]:
                if state.stop or state.data_angles[ang_s]["exp"] is None: continue
                for _ in range(nrep):
                    if state.stop: break
                    self._restore_angle_state(ang_s)
                    if ang_s in km_meas:
                        state.km = km_meas[ang_s]
                        state.lock_left[4] = True
                    else:
                        state.lock_left[4] = _km_lock_saved
                    wl_a,r_a=state.data_angles[ang_s]["exp"]
                    self.run_fit(ang_s,wl_a,r_a)
            state.lock_left[4] = _km_lock_saved
            state.current_ang=saved_ang
            self._restore_angle_state(saved_ang)  # restore km/ks/kb/pol of the current angle
            return {'ang_base':ang_base,'n':n_fin1,'k':k_fin1,'d':state.d}

        # Stage 2: HOBL
        N_NEFF=len(n_fin1)
        p0_s2=np.array(list(p1[:N_CAMP])+[v for a in ang_hobl for v in make_p0_ang(a)]+
            [v for _ in ang_hobl for v in n_fin1])
        bnd_s2=np.array([(p1[i],p1[i]) for i in range(N_CAMP)]+
            [bnd_ang_hobl[j] for _ in ang_hobl for j in range(N_ANG)]+
            [(max(1.0,v*0.95),min(6.5,v*1.05)) for _ in ang_hobl for v in n_fin1])
        # Stage 2 MUST respect ALL the slider locks (lock_n, lock_left[4/5/6/12],
        # lock_extra[0/1]): otherwise it would refit n_eff and the per-angle HOBL
        # parameters, changing n/km/ks/kb/pol/dn/pm DESPITE the lock.
        _anchor_ir = getattr(self, 'anchor_ir_hobl', False)
        # Mask of locked n ext-nodes (base + sub-nodes between pairs both locked),
        # same logic as ph._apply_lock_nk_bounds.
        locked_n_ext = np.zeros(N, dtype=bool)
        for li in range(14):
            jj = np.where(np.isclose(ext_nodes_0, c_nodes_0[li]))[0]
            if len(jj) and state.lock_n[li]:
                locked_n_ext[jj[0]] = True
        for li in range(13):
            j0 = np.where(np.isclose(ext_nodes_0, c_nodes_0[li]))[0]
            j1 = np.where(np.isclose(ext_nodes_0, c_nodes_0[li+1]))[0]
            if len(j0) and len(j1) and state.lock_n[li] and state.lock_n[li+1]:
                for ji in range(j0[0]+1, j1[0]):
                    locked_n_ext[ji] = True
        neff_base = N_CAMP + len(ang_hobl) * N_ANG
        for ki, a in enumerate(ang_hobl):
            base = N_CAMP + ki * N_ANG
            # km/ks/kb: per-angle measured km (takes precedence) → IR anchor
            # (spectrometer constant) for ks/kb → manual lock.
            for off, li, sval in ((0, 4, state.km), (1, 5, state.ks), (2, 6, state.kb)):
                if off == 0 and a in km_meas:   # per-angle MEAS JUMP: km fixed
                    p0_s2[base+off] = km_meas[a]
                    bnd_s2[base+off] = (km_meas[a], km_meas[a])
                elif _anchor_ir:
                    p0_s2[base+off] = sval
                    bnd_s2[base+off] = (sval, sval)
                elif state.lock_left[li]:
                    bnd_s2[base+off] = (p0_s2[base+off], p0_s2[base+off])
            # pol: manual lock (lock_left[12])
            if state.lock_left[12]:
                bnd_s2[base+3] = (p0_s2[base+3], p0_s2[base+3])
            # dn/pm: lock_extra[0/1]
            if state.lock_extra[0]: bnd_s2[base+4] = (p0_s2[base+4], p0_s2[base+4])
            if state.lock_extra[1]: bnd_s2[base+5] = (p0_s2[base+5], p0_s2[base+5])
            # per-angle n_eff: respects lock_n
            for j in range(N_NEFF):
                if locked_n_ext[j]:
                    idx = neff_base + ki * N_NEFF + j
                    bnd_s2[idx] = (p0_s2[idx], p0_s2[idx])
        p0_s2=np.clip(p0_s2,bnd_s2[:,0],bnd_s2[:,1])
        _bm['chi2']=np.inf; _bm['p']=None

        def neff_s2(ki): return slice(N_CAMP+len(ang_hobl)*N_ANG+ki*N_NEFF,
                                      N_CAMP+len(ang_hobl)*N_ANG+(ki+1)*N_NEFF)
        def calc_R_s2(p,a,ki):
            base=N_CAMP+ki*N_ANG
            km_p,ks_p,kb_p,pol_p=p[base],p[base+1],p[base+2],p[base+3]
            dn_p,pm_p=p[base+4],p[base+5]
            wc_p,wlw,wr_p,da_p=p[base+6],p[base+7],p[base+8],p[base+9]
            n_eff=np.clip(p[neff_s2(ki)],1.0,6.5)
            wl_a,r_a=state.data_angles[a]["exp"]; ang_f=float(a)
            _kw=dict(scatt=p[I_SC],offset=p[I_OF],inhom=p[I_IH],
                     kjump=km_p,alpha=ks_p,beta=kb_p,pol=pol_p,
                     delta_n=dn_p,phi_mix=pm_p,phi_s1=state.ps1)
            R,_,_=ph.calc_refl_3angle(wl_a,p[I_D],ext_nd1,n_eff,k_fin1,
                                      ang_f,da_p,wc_p,wlw,wr_p,**_kw)
            return R,r_a

        def obj_s2(p):
            if state.stop: raise StopIteration
            chi2=0.0
            lmin=state.lmin
            for ki,a in enumerate(ang_hobl):
                R,r_a=calc_R_s2(p,a,ki)
                wl_a=state.data_angles[a]["exp"][0]
                w=np.ones_like(r_a); w[wl_a<lmin]=0.0
                w[(wl_a>=400)&(wl_a<600)]=15.0; w[(wl_a>=600)&(wl_a<=1300)]=40.0; w[wl_a>1500]=0.5
                chi2+=float(np.sum(w*(R-r_a)**2))
                # Peak-centering penalty for stage 2 too (HOBL 40/60)
                chi2 += self._peak_penalty(R, wl_a, _peaks_exp.get(a))
                # IR-peak position anchoring at 40° AND 60°: with the collapsed
                # model the 60° maxima are shifted, not deformed → the penalty
                # applies there too.
                if a in (40, 60):
                    wpp = self.peak_pos_w_hi if a == 60 else self.peak_pos_w
                    chi2 += self._peak_pos_penalty(R, wl_a, _peaks_exp.get(a),
                                                   lam_min=self.peak_pos_lam_min, w=wpp)
            if chi2<_bm['chi2']: _bm['chi2']=chi2; _bm['p']=p.copy()
            return chi2

        try:
            minimize(obj_s2,p0_s2,method='L-BFGS-B',bounds=bnd_s2,
                     callback=lambda p:cb(p,"stage2"),
                     options={'maxiter':10000,'ftol':1e-18,'gtol':1e-12})
        except StopIteration: pass
        except Exception as e: print(f"S2 err:{e}")
        if state.stop or _bm['p'] is None: return {}

        p2=_bm['p'].copy(); p2_best=p2.copy(); chi2_best=_bm['chi2']
        pm_starts=np.linspace(-np.pi,np.pi,6,endpoint=False)
        for ki,a in enumerate(ang_hobl):
            if state.stop: break
            idx_pm=N_CAMP+ki*N_ANG+5
            for pm_try in pm_starts:
                if state.stop: break
                pt=p2_best.copy(); pt[idx_pm]=float(np.clip(pm_try,bnd_s2[idx_pm,0],bnd_s2[idx_pm,1]))
                _bm['chi2']=np.inf; _bm['p']=None
                try:
                    minimize(obj_s2,pt,method='L-BFGS-B',bounds=bnd_s2,
                             callback=lambda p:cb(p,"stage3"),
                             options={'maxiter':800,'ftol':1e-18,'gtol':1e-12})
                except (StopIteration,Exception): pass
                if _bm['p'] is not None and _bm['chi2']<chi2_best:
                    chi2_best=_bm['chi2']; p2_best=_bm['p'].copy()

        # Save the per-angle results
        n_eff_hobl={}
        for ki,a in enumerate(ang_hobl):
            n_eff_hobl[a]=np.clip(p2_best[neff_s2(ki)],1.0,6.5)

        results={}
        ki_base=0; ki_hobl=0
        for a in ang_avail:
            if a in ang_base:
                base=N_CAMP+ki_base*N_ANG; ki_base+=1
                km_a,ks_a,kb_a,pol_a=p1[base],p1[base+1],p1[base+2],p1[base+3]
                dn_a,pm_a,da_a=p1[base+4],p1[base+5],p1[base+9]
                n_use=n_fin1; k_use=k_fin1
                wl_a,r_a=state.data_angles[a]["exp"]
                R_fit,_,_=ph.calculate_refl_core(wl_a,p1[I_D],ext_nd1,n_use,k_use,p1[I_SC],p1[I_OF],p1[I_IH],kjump=km_a,alpha=ks_a,beta=kb_a,pol=pol_a,theta_deg=a,delta_n=dn_a,phi_mix=pm_a)
            else:
                base=N_CAMP+ki_hobl*N_ANG; ki_hobl+=1
                km_a,ks_a,kb_a,pol_a=p2_best[base],p2_best[base+1],p2_best[base+2],p2_best[base+3]
                dn_a,pm_a=p2_best[base+4],p2_best[base+5]
                wc_a,wlw_a,wr_a,da_a=p2_best[base+6],p2_best[base+7],p2_best[base+8],p2_best[base+9]
                n_use=n_eff_hobl[a]; k_use=k_fin1
                wl_a,r_a=state.data_angles[a]["exp"]; ang_f=float(a)
                _kw=dict(scatt=p1[I_SC],offset=p1[I_OF],inhom=p1[I_IH],
                         kjump=km_a,alpha=ks_a,beta=kb_a,pol=pol_a,
                         delta_n=dn_a,phi_mix=pm_a,phi_s1=state.ps1)
                R_fit,_,_=ph.calc_refl_3angle(wl_a,p1[I_D],ext_nd1,n_use,k_use,
                                              ang_f,da_a,wc_a,wlw_a,wr_a,**_kw)

            ex_a=({'wc':float(wc_a),'wl':float(wlw_a),'wr':float(wr_a),'da':float(da_a)}
                  if a in (40,60) else {})
            _da_save = da_a if a in (40,60) else 0.0
            self._save_angle_state(a, R_fit,
                (ext_nd1.copy(),n_use.copy(),k_use.copy(),ex_a),
                ang_params={"km":km_a,"ks":ks_a,"kb":kb_a,
                            "pol":pol_a,"dn":dn_a,"pm":pm_a}, da=_da_save)
            results[a]=R_fit

        # Final single fits to refine
        saved_ang=state.current_ang
        _km_lock_saved = state.lock_left[4]
        for ang_s,nrep in [(8,2),(20,2),(40,1),(60,1)]:
            if state.stop or state.data_angles[ang_s]["exp"] is None: continue
            for _ in range(nrep):
                if state.stop: break
                self._restore_angle_state(ang_s)
                # per-angle MEAS JUMP: _restore_angle_state SKIPS km if locked,
                # so I set it to the measured value and lock it → run_fit does not
                # refit it (at 8/20 it would be free, at the HOBL angles the anchor
                # fixes it anyway to state.km = measured).
                if ang_s in km_meas:
                    state.km = km_meas[ang_s]
                    state.lock_left[4] = True
                else:
                    state.lock_left[4] = _km_lock_saved
                wl_a,r_a=state.data_angles[ang_s]["exp"]
                self.run_fit(ang_s,wl_a,r_a)
        state.lock_left[4] = _km_lock_saved
        # Restore the angle we started on COMPLETELY, not just current_ang: the
        # refine loop above leaves state.ext_override holding the LAST angle
        # refined (60°). The caller hands our return value to _on_fit_done, which
        # is written for run_fit's dict — ours is keyed by ANGLE, so its
        # result.get('ext_override') is None and it falls back on the global one,
        # writing 60°'s n(λ) into the selected angle's slot and clobbering what
        # run_fit had correctly saved there. _restore_angle_state puts back this
        # angle's parameters AND its ext_override, so the fallback becomes a no-op.
        self._restore_angle_state(saved_ang)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # SELF-CONSISTENT FIT (run_fit_auto)
    # ─────────────────────────────────────────────────────────────────────────

    def run_fit_auto(self, ang_avail: list = None) -> dict:
        """Self-consistent fit: d loop + fit_multi.
        Returns: a dict of results, or {}
        """
        import copy as _copy
        state=self.state
        N_ITER_RAPIDO=25; PASSO_NM=20.0; MAX_CICLI_D=20

        if ang_avail is None:
            ang_avail=[a for a in [8,20,40,60] if state.data_angles[a]["exp"] is not None]
        ha_8=state.data_angles[8]["exp"] is not None
        ha_20=state.data_angles[20]["exp"] is not None

        if not ha_8:
            if self.on_progress: self.on_progress("FIT AUTO: load the 8° spectrum first")
            return {}
        if not ha_20:
            if self.on_progress: self.on_progress("FIT AUTO: no 20° — direct FIT MULTI")
            return self.run_fit_multi(ang_avail)

        # Invalidate the d_from_peaks cache: it is renewed on the first cycle of
        # THIS fit_auto run (stable for the whole run).
        self._d_peaks_cache = None

        saved_ang=state.current_ang
        d_auto=[None]

        # ── PRE-LOOP: 8°→20° transfer, full fit at 20°, d_target computation with
        # state.d_source.
        # Premise: a reasonable fit at 8° already exists (IR maxima aligned to the
        # exp data). No automatic validation of the 8° fit.
        if self.on_progress:
            self.on_progress("FIT AUTO: pre-loop — copy 8°→20°...")
        sl8 = state.data_angles[8].get("sliders")
        if sl8 is not None:
            sl20 = state.data_angles[20].get("sliders") or {}
            for k in ['d','scatt','inhom','offs','km','ks','kb',
                       'w300','w335','w375','w400','w1100']:
                if k in sl8:
                    sl20[k] = sl8[k]
            state.data_angles[20]["sliders"] = sl20

        # FULL fit at 20° (standard run_fit, not fit_rapido): produces a
        # good-quality fit at 20° before the d_target computation. ~10-15s
        # typical. d stays free in this fit because we want n,k at 20° to find an
        # equilibrium consistent with those at 8°.
        if self.on_progress:
            self.on_progress("FIT AUTO: pre-loop — full fit at 20°...")
        self._restore_angle_state(20)
        wl_a20, r_a20 = state.data_angles[20]["exp"]
        state.iter_count = 0; state.iter_limit = None; state.stop = False
        try:
            self.run_fit(20, wl_a20, r_a20)
        except Exception as _exc:
            if self.on_progress:
                self.on_progress(f"FIT AUTO: full 20° fit failed ({_exc}), proceeding anyway")

        if state.stop:
            return {}

        # Compute d_target with state.d_source (the cache will be populated,
        # stable for the whole run). Premise: the 8° fit is reasonable → the fit
        # peaks (or exp, per d_source) are where they should be.
        d_target_pre = self.calcola_d_interno()
        if d_target_pre is None or not (300 < d_target_pre < 6000):
            if self.on_progress:
                self.on_progress("FIT AUTO: pre-loop d_target not computable, abort")
            return {}
        if self.on_progress:
            self.on_progress(
                f"FIT AUTO: pre-loop d_target = {d_target_pre:.0f} nm "
                f"(source={getattr(state, 'd_source', 'exp')}), "
                f"current state.d = {state.d:.0f} → adaptation loop")
        # NB: we do NOT set state.d=d_target_pre directly. The cycle loop will
        # nudge gradually (PASSO_NM=20). This preserves the n,k
        # consistency as d migrates.

        def fit_rapido(ang):
            if state.data_angles[ang]["exp"] is None: return
            self._restore_angle_state(ang)
            if d_auto[0] is not None: state.d=d_auto[0]
            # Pre-existing user lock (manual, before fit_auto)
            user_lock_d = state.lock_left[0]
            if user_lock_d:
                state.d = float(state.data_angles[ang].get("sliders",{}).get("d",state.d))
            # Internal lock of d during fit_rapido. Without the lock, fit_rapido
            # converges to the local χ² minimum near the initial state and CANCELS
            # the 20 nm nudge of the main loop (with bounds (10,6000) L-BFGS-B can
            # jump to any alternative minimum). With d locked, fit_rapido optimizes
            # only n,k → gradual migration of d driven by the main loop.
            state.lock_left[0] = True
            try:
                state.iter_count=0; state.iter_limit=N_ITER_RAPIDO; state.stop=False
                wl_a,r_a=state.data_angles[ang]["exp"]
                self.run_fit(ang,wl_a,r_a)
                state.iter_limit=None
                if state.iter_count>=N_ITER_RAPIDO: state.stop=False
            finally:
                state.lock_left[0] = user_lock_d
            # Update d_auto only if d was not locked manually
            if not user_lock_d:
                d_auto[0]=state.d

        for ciclo in range(MAX_CICLI_D):
            if state.stop: break
            if self.on_progress: self.on_progress(f"FIT AUTO cycle {ciclo+1}: quick fit 8°...")
            fit_rapido(8)
            if state.stop: break

            if ciclo==0:
                # Do NOT copy the 8° ext_override onto 20°: the UV optical
                # parameters are very different between the two angles and cause
                # instability. Copy only the angle-independent scalar slider
                # parameters (d, scatt, inhom, offs, km, ks, kb and the w-nodes),
                # not the 8° ext_override.
                sl8=state.data_angles[8].get("sliders")
                if sl8 is not None:
                    sl20 = state.data_angles[20].get("sliders") or {}
                    for k in ['d','scatt','inhom','offs','km','ks','kb',
                               'w300','w335','w375','w400','w1100']:
                        if k in sl8:
                            sl20[k] = sl8[k]
                    state.data_angles[20]["sliders"] = sl20

            if self.on_progress: self.on_progress(f"FIT AUTO cycle {ciclo+1}: quick fit 20°...")
            fit_rapido(20)
            if state.stop: break

            d_finestra=self.calcola_d_interno()
            d_fit=state.d
            if d_finestra is None or not (300<d_finestra<6000):
                if self.on_progress: self.on_progress("FIT AUTO: d_finestra not computable")
                break

            diff=d_finestra-d_fit
            if self.on_progress:
                self.on_progress(f"FIT AUTO cycle {ciclo+1}: d_fit={d_fit:.0f} d_fin={d_finestra:.0f} diff={diff:.0f}nm")

            if abs(diff)<20.0: break
            passo=min(abs(diff),PASSO_NM)*np.sign(diff)
            d_nuovo=float(np.clip(d_fit+passo,300,6000))
            d_auto[0]=d_nuovo; state.d=d_nuovo; state.d_mem_global=d_finestra
            if abs(diff)<30.0: break

        d_finale=state.d; d_auto[0]=None
        self._restore_angle_state(saved_ang)
        state.d=d_finale; state.d_mem_global=d_finale

        if state.stop:
            return {}

        # Final FIT MULTI
        if self.on_progress: self.on_progress("FIT AUTO: full FIT MULTI...")
        multi_result = self.run_fit_multi(ang_avail)

        # Verify post-fit_multi. After fit_multi has globally
        # optimized n,k,d, we recompute d_target on the peaks (with the cache
        # invalidated = a fresh read with the updated n,k). If state.d (post-multi)
        # is still farther from d_target than VERIFY_THRESH_NM, we launch a SECOND
        # full round of fit_auto — the first one did not yet have the n,k
        # parameters consolidated at the time m was identified. Limit: 1 verify
        # pass to avoid infinite loops (if they still diverge on the second round,
        # there is a structural problem that 30 extra cycles won't solve).
        VERIFY_THRESH_NM = 50.0
        if state.stop:
            return multi_result
        self._d_peaks_cache = None  # force recompute with the post-multi n,k
        d_target_post = self.calcola_d_interno()
        if d_target_post is None:
            return multi_result
        diff_verify = d_target_post - state.d
        if self.on_progress:
            self.on_progress(
                f"FIT AUTO verify post-multi: state.d={state.d:.0f} "
                f"d_target={d_target_post:.0f} diff={diff_verify:+.0f}nm")
        if abs(diff_verify) < VERIFY_THRESH_NM:
            if self.on_progress:
                self.on_progress(
                    f"FIT AUTO verify: convergence OK (diff<{VERIFY_THRESH_NM:.0f}nm)")
            return multi_result
        if self.on_progress:
            self.on_progress(
                f"FIT AUTO verify: diff>{VERIFY_THRESH_NM:.0f}nm → second round")
        # Second round: invalidate the cache (it will be recomputed on the first
        # cycle with fresh n,k), relaunch the cycles + final fit_multi
        self._d_peaks_cache = None
        saved_ang2 = state.current_ang
        d_auto2 = [None]
        for ciclo in range(MAX_CICLI_D):
            if state.stop: break
            if self.on_progress: self.on_progress(f"FIT AUTO[2] cycle {ciclo+1}: quick fit 8°...")
            # fit_rapido (defined in the first round) closes over d_auto; the
            # second round updates d_auto and state.d inline.
            fit_rapido(8)
            if state.stop: break
            if self.on_progress: self.on_progress(f"FIT AUTO[2] cycle {ciclo+1}: quick fit 20°...")
            fit_rapido(20)
            if state.stop: break
            d_fin2 = self.calcola_d_interno()
            if d_fin2 is None or not (300 < d_fin2 < 6000):
                break
            diff2 = d_fin2 - state.d
            if self.on_progress:
                self.on_progress(
                    f"FIT AUTO[2] cycle {ciclo+1}: d_fit={state.d:.0f} "
                    f"d_fin={d_fin2:.0f} diff={diff2:.0f}nm")
            if abs(diff2) < 20.0: break
            passo2 = min(abs(diff2), PASSO_NM) * np.sign(diff2)
            d_nuovo2 = float(np.clip(state.d + passo2, 300, 6000))
            d_auto[0] = d_nuovo2; state.d = d_nuovo2; state.d_mem_global = d_fin2
            if abs(diff2) < 30.0: break
        d_auto[0] = None
        self._restore_angle_state(saved_ang2)

        if state.stop:
            return multi_result

        if self.on_progress: self.on_progress("FIT AUTO[2]: final FIT MULTI...")
        return self.run_fit_multi(ang_avail)
