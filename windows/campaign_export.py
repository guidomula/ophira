"""campaign_export.py — per-fit campaign log of the Δn_eff read-outs (DATA EXPORT).

One CSV row per accepted fit. Does NOT touch the fitting engine: it reads the
main-window state and the curves already on screen (`l_sim`/`l_exp`), so it logs
exactly what is displayed. Auto-computed fields: sample, angle, d, fringe
POSITION residual in the transparency window, RMS residual, Δn from the fit and
from the maxima. The JUDGEMENT fields (run, m_resolved, kk_consistent,
locked_params, notes) are supplied by the caller via a dialog.

The misfit is logged as `rms_resid_pct`: an unweighted RMS in %R, NOT a chi²
reduced or otherwise — the spectra carry no per-point sigma, so there is nothing
to normalize by. Whether/how to use it is left to the operator (the judgement
fields carry that decision). Δn is a derived read-out (n_angle − n_8°), not a
quantity the fit is optimized toward.

All column names and strings are in English.
"""
import os
import csv
import datetime
import numpy as np
from scipy.signal import find_peaks, savgol_filter

COLUMNS = [
    "timestamp", "sample", "angle_deg", "run", "d_nm",
    "fringe_pos_residual_nm", "fringe_pos_scatter_nm", "rms_resid_pct",
    "kk_consistent", "kk_rms", "kk_snr", "kk_lmin_nm", "m_resolved", "locked_params",
    "dn_fit_offset", "dn_fit_min", "dn_fit_max", "dn_maxima",
    "session", "notes",
]

LAM_LO, LAM_HI = 900.0, 2550.0   # transparency window for the fringe positions
LAM_REF = 1400.0                 # reference wavelength for Δn from maxima


def _smooth(y):
    y = np.asarray(y, float)
    n = len(y)
    if n < 5:
        return y
    w = min(11, (n // 2) * 2 - 1)
    return savgol_filter(y, max(w, 5), 3)


def _fringe_position_stats(wl, R, r_exp):
    """(residual, scatter): residual = mean |Δλ| between model/exp maxima in the
    transparency window (the position mismatch that n depends on); scatter = std
    of the signed position offsets. ('', '') if not assessable."""
    wl = np.asarray(wl, float)
    R = np.asarray(R, float)
    r = np.asarray(r_exp, float)
    m = (wl >= LAM_LO) & (wl <= LAM_HI)
    if int(m.sum()) < 10:
        return "", ""
    wm = wl[m]
    ps, _ = find_peaks(_smooth(R[m]), prominence=1.0)
    pe, _ = find_peaks(_smooth(r[m]), prominence=1.0)
    if len(ps) == 0 or len(pe) == 0:
        return "", ""
    lam_s = wm[ps]
    d = np.array([lam_s[np.argmin(np.abs(lam_s - x))] - x for x in wm[pe]], float)
    return round(float(np.mean(np.abs(d))), 2), round(float(np.std(d)), 2)


def _n_maxima_at(mw, ang, d, ref=LAM_REF):
    """n_eff from the maxima at λ_ref for angle `ang` with the given `d`, or None."""
    try:
        arr = mw.engine.compute_n_from_maxima(ang, d)
    except Exception:
        return None
    if arr is None:
        return None
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return None
    if arr.shape[0] == 1:
        return float(arr[0, 1])
    # The maxima come out numbered from the IR, so λ DESCENDS. np.interp needs x
    # increasing: given a descending x it returns an edge value without a word —
    # here n at the bluest peak, at a λ that differs from angle to angle.
    o = np.argsort(arr[:, 0])
    lam, n_eff = arr[o, 0], arr[o, 1]
    # Outside the peaks np.interp would clamp to the edge value: that is an
    # extrapolation, not a measurement at λ_ref. Better an empty cell.
    if not (lam[0] <= ref <= lam[-1]):
        return None
    return float(np.interp(ref, lam, n_eff))


def _sample_name(state):
    names = [os.path.splitext(os.path.basename(f))[0]
             for f in getattr(state, 'ang_filenames', {}).values() if f]
    if not names:
        return "?"
    pref = os.path.commonprefix(names).rstrip("_- .")
    return pref or names[0]


def gather_auto(mw):
    """Collect the auto-computable fields from the main window `mw`."""
    s = mw.state
    ang = int(getattr(s, 'current_ang', 0) or 0)
    d = float(getattr(s, 'd', 0.0) or 0.0)

    # Model R + exp R from the displayed curves (what is shown on screen)
    wl_s, R = mw.l_sim.get_data()
    _, r = mw.l_exp.get_data()
    R = np.asarray(R, float)
    r = np.asarray(r, float)
    res_pos, scatter = _fringe_position_stats(wl_s, R, r)

    # RMS reflectance residual (%R) = sqrt(mean(res²)) — an unweighted least-squares
    # misfit, NOT a σ-normalized reduced χ² (the spectra carry no per-point σ).
    rms_resid = ""
    if len(R) == len(r) and len(r) > 1:
        rms_resid = round(float(np.sqrt(np.mean((R - r) ** 2))), 4)

    # Δn from the fit (offset, min, max over the transparency window)
    dn_off = dn_min = dn_max = ""
    if ang != 8:
        fit = mw._dn_slope_fit()
        if fit is not None:
            a, _b, dmn, dmx = fit
            dn_off, dn_min, dn_max = round(a, 4), round(dmn, 4), round(dmx, 4)

    # Δn from the maxima: n_maxima(ang) − n_maxima(8°) at λ_ref (NA if no peaks)
    dn_maxima = ""
    if ang != 8:
        n_a = _n_maxima_at(mw, ang, d)
        n_8 = _n_maxima_at(mw, 8, d)
        if n_a is not None and n_8 is not None:
            dn_maxima = round(n_a - n_8, 4)

    # User-chosen reference label (falls back to the derived sample name)
    label = getattr(s, 'sample_label', '') or _sample_name(s)

    # Formal KK metric from the KK window (state.kk_metric), if computed.
    # Only on the 8° row: the KK relation holds for the normal-incidence optical
    # constants, so the figure of merit belongs to the sample through its 8° fit
    # — on a 40°/60° row it would read as that angle's.
    km = (getattr(s, 'kk_metric', None) or {}) if ang == 8 else {}
    kk_rms = round(km['rms'], 5) if isinstance(km.get('rms'), (int, float)) else ""
    kk_snr = (round(km['snr'], 2)
              if isinstance(km.get('snr'), (int, float)) and np.isfinite(km['snr'])
              else "")
    kk_lmin = round(km['lmin']) if isinstance(km.get('lmin'), (int, float)) else ""

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample": label,
        "angle_deg": ang,
        "run": "",
        "d_nm": round(d, 1),
        "fringe_pos_residual_nm": res_pos,
        "fringe_pos_scatter_nm": scatter,
        "rms_resid_pct": rms_resid,
        # kk_consistent is a JUDGEMENT field (yes/no); the numeric kk_rms / kk_snr
        # (= RMS/σ_KK) / kk_lmin_nm below are the formal measure, auto-captured from
        # the KK window (state.kk_metric) if it has been (re)computed for this state.
        "kk_consistent": "",
        "kk_rms": kk_rms,
        "kk_snr": kk_snr,
        "kk_lmin_nm": kk_lmin,
        "m_resolved": "",
        "locked_params": "",
        "dn_fit_offset": dn_off,
        "dn_fit_min": dn_min,
        "dn_fit_max": dn_max,
        "dn_maxima": dn_maxima,
        "session": getattr(mw, '_last_session_path', "") or "",
        "notes": "",
    }


def append_row(csv_path, row):
    """Append `row` (dict) to the CSV; write the header if the file is new.
    Returns the path."""
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
    return csv_path
