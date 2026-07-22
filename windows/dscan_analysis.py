"""
dscan_analysis.py — analysis of a d-scan from several saved sessions.

The operator saves N sessions (curated fits) at different d; this module reads
them (READ ONLY, direct unpickle: does NOT touch the current state nor the
physics singletons) and produces two figures:
  1. overlay of the n(λ) dispersion curves (the smoothest / most Cauchy-regular
     one(s) highlighted as a d-bracket);
  2. the RMS reflectance residual(d) with a parabolic fit (minimum + bracket =
     uncertainty on d) and a table of each session's parameters.

Used by the "d-SCAN" button of the main window. A flat bottom of the residual(d) curve is the
uncertainty on d and steep branches set the bracket; the n(λ) regularity
corroborates it.
"""
import os
import pickle
import numpy as np
from matplotlib.figure import Figure

# Scalar parameters shown in the table (key → label)
_SCALAR_PARAMS = [
    # d + the instrumental params that vary with d (diagnostic). pol/dn are
    # excluded: they sit at default here and carry no diagnostic value.
    ("d", "d"), ("scatt", "scatt"), ("inhom", "inhom"), ("offs", "offs"),
    ("km", "IRjump"), ("ks", "IRexp"), ("kb", "IRdmp"),
]


# Spectral window where the KK consistency residual |n − n_KK| discriminates d
# (transparency edge: k small but rising → n maximally tied to k via KK). UV =
# structural/porosity (indep. of n), IR = k≈0: do NOT discriminate.
KK_REGION = (480.0, 750.0)

# ─── DRAWER ────────────────────────────────────────────────────────────────
# χ²(d)/KK-resid(d) d-scan figure, toggled by this flag. Set to False — or
# delete this flag plus the block guarded by it in main_window._on_dscan_analysis
# — to drop the χ²/KK figure from the build.
LEGACY_DSCAN_DRAWER = True

# ─── d-discriminants: two SIGNED metrics vs d ───────────────────────────────
# Read-only on the curated per-d sessions → deterministic, no re-fit, no
# initial-condition sensitivity (unlike χ²).
#   ΔC_IR = contrast(fit) − contrast(exp)  over IR_CONTRAST_BAND
#           contrast = mean(fringe maxima) − mean(fringe minima)   [%R]
#           d too large → n too small → fit under-contrasts → ΔC_IR < 0
#   ΔR500 = Σ (R_fit − R_exp)               over R500_BAND         [%R·pts]
# Their zero-crossings bracket the optimal d.
IR_CONTRAST_BAND = (1000.0, 2500.0)   # transparency: fringe DEPTH tracks n → d
R500_BAND        = (410.0, 530.0)     # absorption-onset region (default; slider-adjustable)
R500_SLIDER_MAX  = 880.0              # cap the ΔR band below the 890 nm detector jump

# EXPERIMENTAL / REMOVABLE: also expose the IR-contrast band as a slider so the
# best range can be tested. No physical justification yet → set to False (or delete
# this flag + the guarded IR slider spec below) to drop it.
EXPERIMENTAL_IR_BAND_SLIDER = True


def _kk_resid(ext_nd, n_arr, kk_cache, region=KK_REGION):
    """RMS(|n_fit − n_KK|) in the `region` window. None if kk_cache is missing."""
    if kk_cache is None or ext_nd is None:
        return None
    try:
        wl_kk = np.asarray(kk_cache[0], float)
        n_kk = np.asarray(kk_cache[1], float)
    except Exception:
        return None
    n_on = np.interp(wl_kk, ext_nd, n_arr)
    g = np.isfinite(n_kk) & (wl_kk >= region[0]) & (wl_kk <= region[1])
    if int(np.sum(g)) < 5:
        return None
    return float(np.sqrt(np.mean((n_on[g] - n_kk[g]) ** 2)))


# ─── n(λ) regularity discriminant ──────────────────────────────────────────
# A too-small or too-large d does NOT flip the sign of dn/dλ (the curve stays
# monotone) but it makes n(λ) IRREGULAR near the absorption edge:
# a flat shoulder/plateau (d too small) or a sharp overshoot (d too large). In the
# transparency region n(λ) physically follows a smooth normal-dispersion (Cauchy)
# law; the physically-consistent curve is the one closest to it. This is KK-FREE
# (the KK residual is unreliable on complex samples) and it catches the
# shoulder/overshoot the monotonicity test misses. Since d is a BRACKET (not a
# single number), the emphasis is a RANGE: the contiguous set of d whose Cauchy
# residual is within REG_TOL_FACTOR of the minimum are "equally regular" → their
# edges are marked.
REG_FIT_REGION  = (420.0, 1000.0)   # 420 nm = the anomalous-dispersion boundary (the SAME edge as
                                    # the n-monotonicity constraint): below it n is not regular by
                                    # construction, so the Cauchy fit starts exactly there.
REG_TOL_FACTOR  = 1.25              # d within ×1.25 of the min residual = equally regular
REG_LOWCONF_RMS = 0.007             # min residual above this → even the best curve is irregular


def _cauchy_residual(ext_nd, n_arr, region=REG_FIT_REGION):
    """RMS deviation of n(λ) from a smooth Cauchy dispersion n = A + B/λ² + C/λ⁴
    over `region` [nm]. Small = regular normal dispersion; a shoulder (too-small d)
    or an overshoot (too-large d) raises it. None if the curve is missing."""
    if ext_nd is None or n_arr is None:
        return None
    nd = np.asarray(ext_nd, float); nn = np.asarray(n_arr, float)
    o = np.argsort(nd); nd, nn = nd[o], nn[o]
    nd, iu = np.unique(nd, return_index=True); nn = nn[iu]
    if nd.size < 4:
        return None
    x = np.arange(region[0], region[1] + 1.0, 2.0)
    try:
        from scipy.interpolate import PchipInterpolator
        y = PchipInterpolator(nd, nn)(x)
    except Exception:
        y = np.interp(x, nd, nn)
    u = 1.0 / (x ** 2)
    A = np.vstack([np.ones_like(u), u, u ** 2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sqrt(np.mean((y - A @ coef) ** 2)))


def _regularity_bracket(residuals):
    """From r(d) (list ordered by d, entries may be None) return
    (i_lo, i_hi, i_min, low_conf): the CONTIGUOUS index bracket around the smallest
    residual whose members are within REG_TOL_FACTOR of it (the equally-regular
    range), the argmin index, and a low-confidence flag (the best residual is itself
    above REG_LOWCONF_RMS → no clearly regular curve). None if no valid residual.
    i_lo == i_hi is a single sharp minimum (one curve); i_lo < i_hi is a range."""
    n = len(residuals)
    valid = [i for i in range(n) if residuals[i] is not None]
    if not valid:
        return None
    i_min = min(valid, key=lambda i: residuals[i])
    T = residuals[i_min] * REG_TOL_FACTOR
    lo = i_min
    while lo - 1 >= 0 and residuals[lo - 1] is not None and residuals[lo - 1] <= T:
        lo -= 1
    hi = i_min
    while hi + 1 < n and residuals[hi + 1] is not None and residuals[hi + 1] <= T:
        hi += 1
    return lo, hi, i_min, bool(residuals[i_min] > REG_LOWCONF_RMS)


def read_session(path):
    """Read a .psi_sess session (direct unpickle) and extract d, n(λ), k(λ),
    RMS reflectance residual per angle (from the saved fit) and the scalar parameters. No side effects."""
    with open(path, "rb") as f:
        s = pickle.load(f)
    da = s.get("data_angles", {})
    # n(λ) curve: global ext_override, fallback to the first available per-angle one
    ov = s.get("ext_override")
    if ov is None:
        for ang in (8, 20, 40, 60):
            if da.get(ang, {}).get("ext_override") is not None:
                ov = da[ang]["ext_override"]
                break
    ext_nd = n_arr = k_arr = None
    if ov is not None and len(ov) >= 3:
        ext_nd = np.asarray(ov[0], float)
        n_arr = np.asarray(ov[1], float)
        k_arr = np.asarray(ov[2], float)
    # RMS reflectance residual (%R) from the saved fit, per angle (λ≥450)
    chi2 = {}
    for ang in (8, 20, 40, 60):
        dd = da.get(ang, {})
        exp = dd.get("exp")
        fit = dd.get("fit")
        if exp is not None and fit is not None:
            wl = np.asarray(exp[0], float)
            R = np.asarray(exp[1], float)
            fr = np.asarray(fit, float)
            if fr.shape == R.shape:
                m = wl >= 450
                # RMS reflectance residual (%R) over λ≥450 — an unweighted least-
                # squares misfit, NOT a σ-normalized reduced χ². Stored under 'chi2'.
                chi2[ang] = float(np.sqrt(np.mean((fr[m] - R[m]) ** 2)))
    params = {k: s.get(k) for k, _ in _SCALAR_PARAMS}
    kk_resid = _kk_resid(ext_nd, n_arr, s.get("kk_cache"))
    return {
        "path": path, "name": os.path.splitext(os.path.basename(path))[0],
        "d": float(s.get("d", 0.0)), "ext_nd": ext_nd, "n": n_arr, "k": k_arr,
        "chi2": chi2, "params": params, "kk_resid": kk_resid,
        "kk_cache": s.get("kk_cache"),
    }


def read_reflectance_set(paths):
    """For the first angle common to all sessions, return
    (angle, wl_exp, R_exp, [(d, R_fit), ...]) — exp from the first session (same
    sample/angle), sim for each. READ ONLY. None if no common angle."""
    data = []
    for p in paths:
        with open(p, "rb") as f:
            s = pickle.load(f)
        da = s.get("data_angles", {})
        angs = {a for a in (8, 20, 40, 60)
                if da.get(a, {}).get("exp") is not None and da.get(a, {}).get("fit") is not None}
        data.append((float(s.get("d", 0.0)), da, angs))
    common = None
    for _, _, angs in data:
        common = angs if common is None else (common & angs)
    if not common:
        return None
    ang = sorted(common)[0]
    wl0 = np.asarray(data[0][1][ang]["exp"][0], float)
    R0 = np.asarray(data[0][1][ang]["exp"][1], float)
    sims = [(d, np.asarray(da[ang]["fit"], float)) for d, da, _ in sorted(data, key=lambda x: x[0])]
    return ang, wl0, R0, sims


def _chi2_angle(sessions):
    """Angle to use for the parabola: the first present in ALL sessions."""
    common = None
    for s in sessions:
        ks = set(s["chi2"].keys())
        common = ks if common is None else (common & ks)
    if not common:
        return None
    return sorted(common)[0]


def make_n_overlay_figure(sessions):
    """Figure: n(λ) overlay (color per d) with KK edge (450) and the 480 line."""
    sessions = sorted(sessions, key=lambda s: s["d"])
    fig = Figure(figsize=(7.5, 4.6))
    ax = fig.add_subplot(111)
    import matplotlib.cm as cm
    cols = cm.viridis(np.linspace(0.0, 0.85, max(len(sessions), 1)))
    cols[:, :3] *= 0.85             # darken slightly → the faded curves keep decent contrast on white
    # Emphasize the MOST REGULAR dispersion curve(s). A too-small/too-large d keeps
    # n(λ) monotone but makes it IRREGULAR at the absorption edge (a flat shoulder or a
    # sharp overshoot); the physical curve is the one closest to a smooth normal
    # dispersion. Since d is a BRACKET, emphasise the contiguous range of d within
    # REG_TOL_FACTOR of the minimum Cauchy residual: ONE curve for a sharp minimum, the
    # TWO edges for a flat one. A low-confidence flag fires when even the best curve is
    # irregular (degenerate low-porosity samples). KK-free by design (the KK residual is
    # unreliable on complex samples).
    nS = len(sessions)
    xx = np.linspace(300, 1000, 500)
    regs = [_cauchy_residual(s["ext_nd"], s["n"]) for s in sessions]
    brk = _regularity_bracket(regs)              # (lo, hi, imin, low_conf) or None
    if brk is not None:
        reg_lo, reg_hi, reg_min, reg_lowconf = brk
        reg_idx = {reg_min} if reg_lowconf else {reg_lo, reg_hi}
    else:                                        # no valid curve → fall back to the middle
        reg_lo = reg_hi = reg_min = None
        reg_lowconf = False
        reg_idx = {nS // 2} if nS else set()

    for i, (s, c) in enumerate(zip(sessions, cols)):
        if s["ext_nd"] is None:
            continue
        emph = i in reg_idx
        tags = []
        if emph and brk is not None:
            tags.append("least-irregular, low conf." if reg_lowconf
                        else ("smoothest" if reg_lo == reg_hi else "smoothest range"))
        lab = f"d={s['d']:.0f}" + (("  (" + " / ".join(tags) + ")") if tags else "")
        ax.plot(xx, np.interp(xx, s["ext_nd"], s["n"]), color=c,
                lw=2.8 if emph else 1.4, alpha=1.0 if emph else 0.5,
                label=lab, zorder=5 if emph else 2)
        if emph:
            m = (s["ext_nd"] >= 300) & (s["ext_nd"] <= 1000)
            ax.plot(s["ext_nd"][m], s["n"][m], "o", color=c, ms=4, zorder=6)
    ax.axvline(450, color="orange", ls="--", lw=0.8, alpha=.6)
    ax.axvline(480, color="red", ls=":", lw=1.0, alpha=.6)
    # label just below the top frame (axes-fraction y), not AT ylim[1] where it
    # collides with the top curve / frame
    ax.text(452, 0.94, " KK edge", color="#E65100", fontsize=7, fontweight="bold", va="top",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("λ (nm)"); ax.set_ylabel("Refractive index, n")
    if brk is not None and reg_lowconf:
        sub = "emphasized: least-irregular (LOW confidence — no clearly smooth curve)"
    elif brk is not None and reg_lo == reg_hi:
        sub = f"emphasized: smoothest n(λ) — d = {sessions[reg_min]['d']:.0f} nm"
    elif brk is not None:
        sub = f"emphasized: smoothest range {sessions[reg_lo]['d']:.0f}–{sessions[reg_hi]['d']:.0f} nm"
    else:
        sub = "emphasized: smoothest n(λ)"
    ax.set_title("n(λ) dispersion overlay — d-scan\n" + sub, fontsize=10)
    ax.grid(alpha=.25)
    _lg = ax.legend(fontsize=8)
    try:
        _lg.set_draggable(True)     # user can drag the legend off overlapping curves
    except Exception:
        pass
    fig.tight_layout()
    fig._plot_axes = [ax]           # the only axes; used by the aspect-lock export/preview
    return fig


def make_chi2_figure(sessions):
    """Figure: RMS reflectance residual(d) + parabola (min+bracket), KK residual, table."""
    ang = _chi2_angle(sessions)
    sessions = sorted(sessions, key=lambda s: s["d"])
    fig = Figure(figsize=(11, 5.4))
    ax = fig.add_axes([0.075, 0.36, 0.40, 0.56])
    axt = fig.add_axes([0.62, 0.04, 0.37, 0.93]); axt.axis("off")
    ax2 = None                       # KK twin axis, created below if kk_cache present

    d_opt = None
    if ang is not None:
        d = np.array([s["d"] for s in sessions])
        cr = np.array([s["chi2"][ang] for s in sessions])
        ax.plot(d, cr, "o", color="#1565C0", ms=8, mec="white", mew=1, zorder=3)
        chi2_title = f"RMS residual vs d  ({ang}°)"
        if len(d) >= 3:
            a, b, c = np.polyfit(d, cr, 2)
            if a > 0:
                d_opt = -b / (2 * a)
                xx = np.linspace(d.min(), d.max(), 200)
                ax.plot(xx, a * xx ** 2 + b * xx + c, "-", color="#1565C0",
                        lw=1.4, alpha=.7, zorder=1)
                ax.axvline(d_opt, color="#1565C0", ls="--", lw=1.2)
                # bracket: where the parabola exceeds 2× the minimum
                cmin = a * d_opt ** 2 + b * d_opt + c
                disc = (cmin) / a  # solve a(d-d0)^2 = cmin -> Δ=sqrt(cmin/a)
                half = float(np.sqrt(max(cmin, 0) / a)) if cmin > 0 else 0.0
                ax.axvspan(d_opt - half, d_opt + half, color="#1565C0", alpha=.10)
                chi2_title = (f"RMS residual vs d  ({ang}°)   →   d = {d_opt:.0f} ± {half:.0f} nm"
                              if half >= 1 else
                              f"RMS residual vs d  ({ang}°)   →   d = {d_opt:.0f} nm")
        ax.set_xlabel("thickness d (nm)"); ax.set_ylabel(f"RMS resid [%R]  ({ang}°)")
        ax.grid(alpha=.25); ax.set_title(chi2_title, fontsize=10)

        # ── KK residual on a twin axis, recomputed live ────────────────────────
        # Complementary to χ²: where χ² and the KK residual agree → "easy" sample.
        # The residual is recomputed from the per-session n(λ)/n_KK over a
        # user-chosen band (RangeSlider, default 450-700 nm = the KK edge, excluding
        # the <440 anomalous region and the diluting IR), and shown either as a
        # MAGNITUDE (|Δn| RMS → a minimum) or SIGNED (mean(n_fit−n_KK) → a
        # zero-crossing, where n_fit passes from above to below n_KK). abs↔signed via
        # the radio, band via the slider; both built post-embed (see
        # _show_figure_dialog).
        st = {"region": (450.0, 700.0), "signed": False}
        def _kkv(s, region, signed):
            kc = s.get("kk_cache")
            if kc is None or s.get("ext_nd") is None:
                return np.nan
            wl_kk = np.asarray(kc[0], float); n_kk = np.asarray(kc[1], float)
            n_on = np.interp(wl_kk, s["ext_nd"], s["n"])
            g = np.isfinite(n_kk) & (wl_kk >= region[0]) & (wl_kk <= region[1])
            if int(np.sum(g)) < 5:
                return np.nan
            dn = n_on[g] - n_kk[g]
            return float(np.mean(dn)) if signed else float(np.sqrt(np.mean(dn ** 2)))
        def _kk_curve(region, signed):
            return np.array([_kkv(s, region, signed) for s in sessions])

        if any(s.get("kk_cache") is not None for s in sessions):
            ax2 = ax.twinx()
            l_kk, = ax2.plot(d, _kk_curve(st["region"], st["signed"]), "s--",
                             color="#C62828", ms=6, mec="white", mew=0.8, lw=1.2, zorder=4)
            zline = ax2.axhline(0, color="#C62828", lw=0.9, alpha=.45, zorder=1, visible=False)
            vmark = ax2.axvline(d[0], color="#C62828", ls=":", lw=1.2, alpha=0.8, visible=False)
            ax2.tick_params(axis="y", labelcolor="#C62828")

            def _refresh_kk():
                kr = _kk_curve(st["region"], st["signed"])
                l_kk.set_ydata(kr)
                lo, hi = int(st["region"][0]), int(st["region"][1])
                if st["signed"]:
                    zline.set_visible(True)
                    zc = _zero_cross(d, kr)
                    if zc is not None:
                        vmark.set_xdata([zc, zc]); vmark.set_visible(True)
                        note = f"KK signed crossing (above→below): d = {zc:.0f} nm   [{lo}-{hi} nm]"
                    else:
                        vmark.set_visible(False); note = f"KK signed: no crossing   [{lo}-{hi} nm]"
                    ax2.set_ylabel("KK signed  mean(n_fit − n_KK)", color="#C62828", fontsize=9)
                else:
                    zline.set_visible(False)
                    fin = np.isfinite(kr)
                    if fin.any():
                        i_kk = int(np.argmin(np.where(fin, kr, np.inf)))
                        vmark.set_xdata([d[i_kk], d[i_kk]]); vmark.set_visible(True)
                        note = f"KK-resid min: d = {d[i_kk]:.0f} nm   [{lo}-{hi} nm]"
                    else:
                        vmark.set_visible(False); note = f"[{lo}-{hi} nm]"
                    ax2.set_ylabel("KK residual  |Δn| RMS", color="#C62828", fontsize=9)
                ax2.relim(); ax2.autoscale_view()
                ax.set_title(chi2_title + (("\n" + note) if note else ""), fontsize=9)
                if fig.canvas is not None:
                    fig.canvas.draw_idle()
            _refresh_kk()

            def _update_kk_region(val):
                st["region"] = (float(val[0]), float(val[1])); _refresh_kk()
            def _update_kk_mode(label):
                st["signed"] = str(label).lower().startswith("sign"); _refresh_kk()

            _sax = fig.add_axes([0.34, 0.175, 0.17, 0.03])
            fig._slider_specs = [(_sax, 400.0, 900.0, tuple(st["region"]),
                                  "KK region (nm)", _update_kk_region)]
            _rax = fig.add_axes([0.05, 0.04, 0.095, 0.12])
            _rax.set_frame_on(False)
            fig._radio_specs = [(_rax, ("|Δn| (abs)", "signed"), 0, _update_kk_mode)]
    else:
        ax.text(0.5, 0.5, "no residual (saved fit missing)", ha="center")

    # parameter table (d and the RMS residual per angle as the FIRST columns, then
    # the scalar parameters). The dict key 'chi2' holds the RMS reflectance
    # residual, not a σ-normalized χ².
    chi_angs = sorted({a for s in sessions for a in s["chi2"]})
    lines = ["PARAMETERS per session:", ""]
    hdr = f"{'d':>6} " + " ".join(f"{'RMS'+str(a):>7}" for a in chi_angs) + "  " \
        + " ".join(f"{lab:>6}" for _, lab in _SCALAR_PARAMS[1:])
    lines.append(hdr)
    for s in sessions:
        p = s["params"]
        row = f"{s['d']:6.0f} "
        for a in chi_angs:
            cv = s["chi2"].get(a)
            row += f"{(cv if cv is not None else float('nan')):7.4f} "
        row += " "
        for k, _ in _SCALAR_PARAMS[1:]:
            v = p.get(k)
            row += f"{(v if v is not None else float('nan')):6.2f} "
        lines.append(row)
    axt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=6.2,
             va="top", ha="left", transform=axt.transAxes)
    fig._plot_axes = [ax] if ax2 is None else [ax, ax2]   # keep the plot (+twin) on export
    return fig, d_opt


# ═══════════════════════════════════════════════════════════════════════════
#  d-discriminants figure: IR fringe-contrast Δ + 500 nm Σ vs d
# ═══════════════════════════════════════════════════════════════════════════
def _fringe_contrast(wl, R, band=IR_CONTRAST_BAND):
    """Fringe depth = mean(local maxima R) − mean(local minima R) in `band` [%R].
    fit and exp are measured identically (same prominence). Requires ≥2 maxima AND
    ≥2 minima — a real fringe OSCILLATION, not one bump on a monotonic trend: with
    only 1 max/1 min far apart the "contrast" degenerates to the band's overall slope,
    a meaningless number (thin-sample failure). None when there are not enough fringe
    extrema (→ the figure then states the band has no measurable fringes, instead of
    plotting garbage or silently dropping the curve). Good samples are unaffected:
    they have many fringes."""
    from scipy.signal import find_peaks
    wl = np.asarray(wl, float); R = np.asarray(R, float)
    m = (wl >= band[0]) & (wl <= band[1])
    if int(np.sum(m)) < 5:
        return None
    x = R[m]
    ptp = float(np.nanmax(x) - np.nanmin(x))
    if ptp <= 0:
        return None
    prom = 0.02 * ptp                       # 2% of the band range
    pk, _ = find_peaks(x, prominence=prom)
    tr, _ = find_peaks(-x, prominence=prom)
    if len(pk) < 2 or len(tr) < 2:          # need a real oscillation, not 1 extremum on a trend
        return None
    return float(np.mean(x[pk]) - np.mean(x[tr]))


def _r500_sum(wl, R_fit, R_exp, band=R500_BAND):
    """Signed Σ(R_fit − R_exp) over `band` [%R·pts]. None if the band is empty."""
    wl = np.asarray(wl, float)
    m = (wl >= band[0]) & (wl <= band[1])
    if not m.any():
        return None
    return float(np.sum(np.asarray(R_fit, float)[m] - np.asarray(R_exp, float)[m]))


def _zero_cross(x, y):
    """First interpolated zero-crossing of y(x) (x need not be sorted). None if none."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None
    s = np.argsort(x); x, y = x[s], y[s]
    for i in range(len(x) - 1):
        if y[i] == 0.0:
            return float(x[i])
        if y[i] * y[i + 1] < 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def make_contrast_figure(sessions, refl):
    """Figure: two SIGNED d-discriminants vs d (twin axes) + parameter table.
       ΔC_IR = contrast(fit) − contrast(exp)  over the IR band
       ΔR    = Σ(R_fit − R_exp)               over the (slider-adjustable) ΔR band
    Their zero-crossings bracket the optimal d. READ ONLY.

    The ΔR band is adjustable via a RangeSlider (capped at R500_SLIDER_MAX, below the
    890 nm detector jump): the informative absorption-onset region moves with porosity,
    so the band is placed where the constrained (KK-consistent) fit is d-sensitive.
    The IR band is also slider-adjustable if EXPERIMENTAL_IR_BAND_SLIDER. The sliders
    are declared here on `fig._slider_specs` and instantiated by the host AFTER the Qt
    canvas exists (see main_window._show_figure_dialog).

    `refl` = read_reflectance_set(paths) output, or None.
    Returns (fig, d_contrast0, d_r500_0)."""
    fig = Figure(figsize=(11, 5.4))
    ax = fig.add_axes([0.12, 0.28, 0.40, 0.63])
    axt = fig.add_axes([0.66, 0.05, 0.33, 0.90]); axt.axis("off")
    if refl is None:
        ax.text(0.5, 0.5, "no common angle (exp/fit missing)", ha="center")
        fig._plot_axes = [ax]
        return fig, None, None
    ang, wl, R_exp, sims = refl
    wl = np.asarray(wl, float); R_exp = np.asarray(R_exp, float)
    ds = np.array([float(d) for d, _ in sims])
    Rfits = [np.asarray(Rf, float) for _, Rf in sims]

    def r500_curve(band):
        return np.array([(_r500_sum(wl, Rf, R_exp, band) or np.nan) for Rf in Rfits])

    def contrast_curve(band):
        ce = _fringe_contrast(wl, R_exp, band)
        out = []
        for Rf in Rfits:
            cf = _fringe_contrast(wl, Rf, band)
            out.append((cf - ce) if (cf is not None and ce is not None) else np.nan)
        return np.array(out)

    st = {"dC": contrast_curve(IR_CONTRAST_BAND), "dR": r500_curve(R500_BAND),
          "ir": tuple(IR_CONTRAST_BAND), "r500": tuple(R500_BAND)}

    # zero line PER axis, in the axis colour (a single shared line is misleading
    # because the two twin axes have different zero heights)
    ax.axhline(0, color="#1565C0", lw=0.8, alpha=.5, zorder=1)
    l1, = ax.plot(ds, st["dC"], "o-", color="#1565C0", ms=7, mec="white", mew=1,
                  zorder=3, label="ΔC_IR  contrast(fit−exp)")
    # shown when ΔC_IR is not measurable in the chosen band (no real IR fringes — e.g. a
    # thin / low-contrast sample); toggled live in _refresh() so it tracks the IR slider
    # instead of the curve silently vanishing.
    ir_note = ax.text(0.5, 0.5, "ΔC_IR: no measurable IR fringes in this band\n"
                      "(thin / low-contrast sample — use ΔR)",
                      transform=ax.transAxes, ha="center", va="center", color="#1565C0",
                      fontsize=9, fontstyle="italic", visible=False, zorder=6,
                      bbox=dict(boxstyle="round", fc="white", ec="#1565C0", alpha=.85))
    ax.set_xlabel("thickness d (nm)")
    ax.set_ylabel("IR fringe-contrast Δ  (fit − exp)  [%R]", color="#1565C0")
    ax.tick_params(axis="y", labelcolor="#1565C0"); ax.grid(alpha=.25)
    ax2 = ax.twinx()
    l2, = ax2.plot(ds, st["dR"], "s--", color="#C62828", ms=6, mec="white", mew=.8,
                   lw=1.2, zorder=4, label="ΔR  Σ(fit−exp)")
    ax2.set_ylabel("Σ(fit − exp) over ΔR band  [%R·pts]", color="#C62828")
    ax2.tick_params(axis="y", labelcolor="#C62828")
    ax2.axhline(0, color="#C62828", lw=0.8, alpha=.5, zorder=1)
    # legend on ax2 (the TOP twin axis), not ax: the twin above ax intercepts the pick,
    # so a legend on ax is un-draggable (looks identical either way).
    _lg = ax2.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8, loc="best")
    try:
        _lg.set_draggable(True)     # draggable off overlapping curves
    except Exception:
        pass

    # toggle-able crossing markers + bracket (updated live by the sliders)
    vC = ax.axvline(ds[0], color="#1565C0", ls="--", lw=1.2, visible=False, zorder=2)
    vR = ax2.axvline(ds[0], color="#C62828", ls=":", lw=1.4, visible=False, zorder=2)
    span = {"h": None}

    def _title(dC0, dR0):
        t = (f"d-discriminants vs d  ({ang}°)   "
             f"ΔR band {int(st['r500'][0])}-{int(st['r500'][1])} nm")
        line2 = []
        if dC0 is not None: line2.append(f"contrast→{dC0:.0f}")
        if dR0 is not None: line2.append(f"ΔR→{dR0:.0f}")
        if dC0 is not None and dR0 is not None:
            line2.append(f"[{min(dC0, dR0):.0f}, {max(dC0, dR0):.0f}]")
        if line2:                       # crossings + bracket on a 2nd line so the title never overflows
            t += "\n" + "   ".join(line2)
        return t

    def _refresh():
        dC0 = _zero_cross(ds, st["dC"]); dR0 = _zero_cross(ds, st["dR"])
        if dC0 is not None: vC.set_xdata([dC0, dC0])
        vC.set_visible(dC0 is not None)
        if dR0 is not None: vR.set_xdata([dR0, dR0])
        vR.set_visible(dR0 is not None)
        if span["h"] is not None:
            try: span["h"].remove()
            except Exception: pass
            span["h"] = None
        if dC0 is not None and dR0 is not None:
            span["h"] = ax.axvspan(min(dC0, dR0), max(dC0, dR0),
                                   color="green", alpha=.08, zorder=0)
        ir_note.set_visible(not np.any(np.isfinite(st["dC"])))   # no ΔC_IR → say so, don't vanish
        ax.set_title(_title(dC0, dR0), fontsize=9)
        return dC0, dR0

    dC0, dR0 = _refresh()

    # parameter table (d + the amplitude knobs; metric values are read off the plot,
    # since the band — hence the metrics — is slider-adjustable)
    pbyd = {round(s["d"]): s.get("params", {}) for s in sessions}
    def _g(p, k):
        v = p.get(k)
        return f"{v:7.2f}" if isinstance(v, (int, float)) else f"{'—':>7}"
    lines = ["PARAMETERS per session:", "",
             f"{'d':>6} {'scatt':>7} {'inhom':>7} {'offs':>7} {'pol':>7}"]
    for d in sorted(ds):
        p = pbyd.get(round(d), {})
        lines.append(f"{d:6.0f} {_g(p,'scatt')} {_g(p,'inhom')} {_g(p,'offs')} "
                     f"{_g(p,'pol')}")
    axt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=6.8,
             va="top", ha="left", transform=axt.transAxes)

    # ── slider callbacks (invoked live once the RangeSliders are built post-embed) ──
    def _update_r500(val):
        st["r500"] = (float(val[0]), float(val[1]))
        st["dR"] = r500_curve(st["r500"])
        l2.set_ydata(st["dR"]); ax2.relim(); ax2.autoscale_view()
        _refresh()
        if fig.canvas is not None:
            fig.canvas.draw_idle()

    def _update_ir(val):
        st["ir"] = (float(val[0]), float(val[1]))
        st["dC"] = contrast_curve(st["ir"])
        l1.set_ydata(st["dC"]); ax.relim(); ax.autoscale_view()
        _refresh()
        if fig.canvas is not None:
            fig.canvas.draw_idle()

    specs = []
    _wlmin = max(300.0, float(np.floor(wl.min())))
    sax_r = fig.add_axes([0.18, 0.12, 0.34, 0.03])
    specs.append((sax_r, _wlmin, float(R500_SLIDER_MAX), tuple(st["r500"]),
                  "ΔR band (nm)", _update_r500))
    if EXPERIMENTAL_IR_BAND_SLIDER:
        sax_i = fig.add_axes([0.18, 0.055, 0.34, 0.03])
        specs.append((sax_i, 700.0, float(np.ceil(wl.max())), tuple(st["ir"]),
                      "IR band (nm) [exp.]", _update_ir))
    fig._slider_specs = specs
    fig._plot_axes = [ax, ax2]      # keep the plot (+twin) on aspect-lock export/preview
    return fig, dC0, dR0
