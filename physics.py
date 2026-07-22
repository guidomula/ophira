"""
physics.py — OPHIRA physics engine.
Contains all the physics and math of the reflectometry model.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

# --- HIGHLY-OBLIQUE ANGLE CONFIGURATION ---
# HOBL = the two strongly oblique angles (40°, 60°) that use the fuller model:
# angular-spread averaging, birefringence as Δn_eff, stronger regularization. The
# base angles (8°, 20°) are oblique too, but near-normal for most purposes.
_HOBL_ANGLES = (40, 60)
_IR_ANGLES  = (8, 20, 40, 60)

# --- 1. PHYSICS ENGINE ---
# ── Crystalline Si optical constants — tabulated data (Palik + Green 2008) ───
# Tabulated because Si's UV index and its k≈3.3–3.5 are not captured by a closed-form model.
_SI_WL = np.array([250,255,258,262,265,270,275,280,285,290,295,300,310,320,
                   330,340,350,360,370,380,390,400,410,420,430,450,500,550,
                   600,650,700,800,900,1000,1200,1500,2000,2500], dtype=float)
_SI_N  = np.array([1.640,1.570,2.100,2.900,3.200,3.620,3.700,3.760,4.100,4.430,
                   4.720,5.000,5.340,5.530,5.570,5.930,6.080,6.410,6.590,6.490,
                   5.920,5.570,5.220,5.020,4.890,4.290,4.100,3.960,3.850,3.800,
                   3.750,3.690,3.614,3.570,3.524,3.495,3.476,3.465])
_SI_K  = np.array([3.330,3.348,3.520,3.720,3.820,3.895,3.650,3.410,3.480,3.560,
                   3.540,3.520,3.500,3.470,3.430,3.390,3.340,3.090,2.590,1.980,
                   1.250,0.390,0.175,0.107,0.073,0.072,0.036,0.016,0.009,0.004,
                   0.003,0.001,0.000,0.000,0.000,0.000,0.000,0.000])
_si_n_interp = PchipInterpolator(_SI_WL, _SI_N)
_si_k_interp = PchipInterpolator(_SI_WL, _SI_K)

def _nk_Si(wl_nm):
    """Complex crystalline-Si index from the Palik/Green table, Pchip interpolation."""
    wl_c = np.clip(wl_nm, _SI_WL[0], _SI_WL[-1])
    return (_si_n_interp(wl_c) + 1j * np.maximum(_si_k_interp(wl_c), 0.0))


# ── Materials database — CSV from the materials/ folder with Sellmeier fallback ─
import os as _os, csv as _csv

def _find_materials_dir():
    base = _os.path.dirname(_os.path.abspath(__file__))
    # look for materials/ next to this file, then one level up
    for d in [_os.path.join(base, 'materials'),
              _os.path.join(_os.path.dirname(base), 'materials')]:
        if _os.path.isdir(d): return d
    return None

_mat_cache = {}
# Materials we already warned about the hardcoded fallback for (warn only once:
# _nk_substrate is called on every reflectance evaluation).
_fallback_warned = set()

def _warn_material_fallback(kind, material):
    """Warn ONCE if the CSV is missing and the internal fallback is used.
    CRITICAL for Si: the internal _nk_Si table has the E1 transition at the
    wrong wavelength → without the Si CSV the fits on a Si substrate/film are
    silently wrong. An unpackaged CSV must not go unnoticed."""
    key = (kind, material)
    if key in _fallback_warned:
        return
    _fallback_warned.add(key)
    path = f"materials/{_material_subdir(kind)}/{material}.csv"
    if material == "Si":
        print(f"[physics] WARNING: {path} NOT found → fallback to the internal Si "
              f"table with WRONG E1. The fits on Si ({kind}) will be wrong. "
              f"Restore {path}.")
    else:
        print(f"[physics] Note: {path} not found → using internal fallback "
              f"(Sellmeier) for {kind} '{material}'.")

def _load_material_csv(path):
    """Read a `wl_nm,n,k` CSV. Leading '#' lines carry the attribution of the
    data and are skipped. None if the file cannot be used."""
    wl, n_arr, k_arr = [], [], []
    try:
        with open(path, 'r') as f:
            rows = [ln for ln in f if not ln.lstrip().startswith('#')]
        for row in _csv.DictReader(rows):
            try:
                wl.append(float(row['wl_nm']))
                n_arr.append(float(row['n']))
                k_arr.append(float(row.get('k', 0.0)))
            except (ValueError, KeyError, TypeError):
                continue
    except Exception:
        return None
    if len(wl) < 2: return None
    wl = np.array(wl); n_arr = np.array(n_arr); k_arr = np.array(k_arr)
    idx = np.argsort(wl)
    return wl[idx], n_arr[idx], k_arr[idx]

def _material_subdir(kind):
    """Folder holding the CSV for this role: 'sub' → substrates, 'film' → films."""
    return 'substrates' if kind == 'sub' else 'films'

def _get_material_data(kind, name):
    key = (kind, name)
    if key in _mat_cache: return _mat_cache[key]
    mdir = _find_materials_dir()
    if mdir:
        # The role subfolder first, then the flat directory (older installs)
        for path in (_os.path.join(mdir, _material_subdir(kind), name + '.csv'),
                     _os.path.join(mdir, name + '.csv')):
            if _os.path.isfile(path):
                data = _load_material_csv(path)
                if data is not None:
                    _mat_cache[key] = data
                    return data
    _mat_cache[key] = None
    return None

def _scan_materials(kind):
    """Materials available as CSV in materials/<role>/, in alphabetical order.

    A file is listed only if it really parses: a name in the menu whose CSV is
    unreadable would silently fall back to another material (Si for a substrate,
    a free fit for a film) and the wrong data would never show up in the UI."""
    mdir = _find_materials_dir()
    if mdir is None: return []
    subdir = _material_subdir(kind)
    d = _os.path.join(mdir, subdir)
    if not _os.path.isdir(d): return []
    names = []
    for f in sorted(_os.listdir(d)):
        if not f.endswith('.csv'): continue
        if _load_material_csv(_os.path.join(d, f)) is not None:
            names.append(_os.path.splitext(f)[0])
        else:
            print(f"[physics] WARNING: materials/{subdir}/{f} is not readable as "
                  f"'wl_nm,n,k' → material ignored (it will not appear in the menu).")
    return names

def _interp_material(wl_nm, data_tuple):
    wl_d, n_d, k_d = data_tuple
    wl = np.asarray(wl_nm, dtype=float)
    n = np.interp(wl, wl_d, n_d, left=n_d[0], right=n_d[-1])
    k = np.interp(wl, wl_d, k_d, left=k_d[0], right=k_d[-1])
    return (n + 1j*k).astype(complex)

def _reload_material_cache():
    _mat_cache.clear()
    _fallback_warned.clear()

# ── Substrates ────────────────────────────────────────────────────────────────

def _nk_substrate(wl_nm, material="Si"):
    wl = np.asarray(wl_nm, dtype=float)
    if material == "Aria" or material == "Air":
        return np.ones_like(wl, dtype=complex)
    # Try the CSV first
    data = _get_material_data('sub', material)
    if data is not None:
        return _interp_material(wl, data)
    # Sellmeier / internal-table fallback — warn (CSV missing)
    _warn_material_fallback('sub', material)
    wl_um = wl / 1000.0
    if material == "Si":
        return _nk_Si(wl)
    elif material == "BK7":
        n2 = (1 + 1.03961212*wl_um**2/(wl_um**2-0.00600069867)
                + 0.231792344*wl_um**2/(wl_um**2-0.0200179144)
                + 1.01046945*wl_um**2/(wl_um**2-103.560653))
    elif material in ("SiO2", "Zaffiro", "Sapphire"):
        if "SiO2" in material:
            n2 = (1 + 0.6961663*wl_um**2/(wl_um**2-0.0684043**2)
                    + 0.4079426*wl_um**2/(wl_um**2-0.1162414**2)
                    + 0.8974794*wl_um**2/(wl_um**2-9.896161**2))
        else:
            n2 = (1 + 1.4313493*wl_um**2/(wl_um**2-0.0726631**2)
                    + 0.6505471*wl_um**2/(wl_um**2-0.1193242**2)
                    + 5.3414021*wl_um**2/(wl_um**2-18.028251**2))
    elif material == "CaF2":
        n2 = (1 + 0.5675888*wl_um**2/(wl_um**2-0.050263605**2)
                + 0.4710914*wl_um**2/(wl_um**2-0.1003909**2)
                + 3.8484723*wl_um**2/(wl_um**2-34.649040**2))
    else:
        return _nk_Si(wl)
    return np.sqrt(np.maximum(n2, 1.0)).astype(complex)

def _nk_film_known(wl_nm, material):
    """n+ik for a known material. None if PSi (free fit)."""
    if material in ('PSi', None, ''):
        return None
    wl = np.asarray(wl_nm, dtype=float)
    # Try the CSV first
    data = _get_material_data('film', material)
    if data is not None:
        return _interp_material(wl, data)
    # Sellmeier / internal-table fallback — warn (CSV missing)
    _warn_material_fallback('film', material)
    wl_um = wl / 1000.0
    if material == "SiO2":
        n2 = (1 + 0.6961663*wl_um**2/(wl_um**2-0.0684043**2)
                + 0.4079426*wl_um**2/(wl_um**2-0.1162414**2)
                + 0.8974794*wl_um**2/(wl_um**2-9.896161**2))
        return np.sqrt(np.maximum(n2, 1.0)).astype(complex)
    elif material == "TiO2":
        n2 = 5.913 + 0.2441/(wl_um**2 - 0.0803)
        return np.sqrt(np.maximum(n2, 1.0)).astype(complex)
    elif material == "Si":
        return _nk_Si(wl)
    elif material == "Al2O3":
        n2 = (1 + 1.4313493*wl_um**2/(wl_um**2-0.0726631**2)
                + 0.6505471*wl_um**2/(wl_um**2-0.1193242**2)
                + 5.3414021*wl_um**2/(wl_um**2-18.028251**2))
        return np.sqrt(np.maximum(n2, 1.0)).astype(complex)
    return None

# Materials computed here in closed form, used when materials/ is missing. They
# are the minimum database the code carries on its own: every name below is one
# _nk_substrate / _nk_film_known can actually build — a name they cannot build
# would fall back to Si (substrate) or to a free fit (film) without a word.
_FALLBACK_SUBSTRATES = ["Si", "BK7", "SiO2", "Sapphire", "CaF2", "Air"]
_FALLBACK_FILMS      = ["SiO2", "TiO2", "Si", "Al2O3"]

def _build_substrate_list():
    names = _scan_materials('sub') or list(_FALLBACK_SUBSTRATES)
    # Si first: it is the substrate of essentially every sample
    if "Si" in names:
        names = ["Si"] + [n for n in names if n != "Si"]
    # Air has no CSV — it is n=1 by definition
    if "Air" not in names and "Aria" not in names:
        names.append("Air")
    return names

def _build_film_list():
    names = _scan_materials('film') or list(_FALLBACK_FILMS)
    # Fixed preferred order, then anything else the folder holds, alphabetically
    preferred = ["SiO2", "SiO2_thermal", "TiO2", "Si", "Al2O3"]
    ordered  = [n for n in preferred if n in names]
    ordered += [n for n in names if n not in preferred]
    # PSi has no CSV — it IS the free fit of n(λ), k(λ), and it must stay at
    # index 0: the whole engine reads `film index > 0` as "known material".
    return ["PSi"] + [n for n in ordered if n != "PSi"]

_SUBSTRATI = _build_substrate_list()
_FILM_MATS = _build_film_list()
_substrate  = [0]   # index into _SUBSTRATI
_film_mat   = [0]   # index into _FILM_MATS (0 = PSi = free fit)

# ── Name ↔ index resolution ───────────────────────────────────────────────────
# The lists above follow the materials/ folder, so a position in them is not
# stable: adding one CSV shifts every material after it. Sessions and .par files
# therefore travel by NAME. Older sessions can carry an index into these two frozen
# lists instead — do not reorder them.
_LEGACY_SUBSTRATES = ["Si", "BK7", "SiO2", "Zaffiro", "CaF2", "Aria"]
_LEGACY_FILMS      = ["PSi", "SiO2", "SiO2_thermal", "TiO2", "Si", "Al2O3"]

# Legacy material names mapped to the current names the menus use (the menus are
# built from the CSV file names).
_MATERIAL_ALIASES = {"Zaffiro": "Sapphire", "Aria": "Air"}

def resolve_material(kind, ref):
    """Index of a material in the current list, given its name — or the integer
    index an older session saved. None when it cannot be resolved (unknown name,
    or a CSV that is no longer there): the caller decides what that means."""
    names = _SUBSTRATI if kind == 'sub' else _FILM_MATS
    if isinstance(ref, str):
        name = ref
    elif isinstance(ref, (int, np.integer)):
        legacy = _LEGACY_SUBSTRATES if kind == 'sub' else _LEGACY_FILMS
        if not 0 <= int(ref) < len(legacy):
            return None
        name = legacy[int(ref)]
    else:
        return None
    name = _MATERIAL_ALIASES.get(name, name)
    return names.index(name) if name in names else None


# S/P consistency (URA cross-term) over the WHOLE spectrum vs VIS only.
# True = the cross-term is active in the IR too (w_cross=1) — needed for the 60°
# birefringence signal (Brewster) which lives only in the coherent channel.
# False = cross-term off for λ>1000nm.
_CROSS_FULL_SPECTRUM = [True]


def measure_ir_jump(wl, R, lam0=890.0, half=30.0, gap=2.0, deg=2,
                    km_min=0.2, km_max=2.0):
    """Measure the amplitude of the detector jump at lam0 (≈890 nm, PMT→PbS swap)
    directly from the raw experimental data.

    The true R(λ) is continuous across lam0 (the fringes are smooth); the jump is
    a sharp instrumental discontinuity. To separate them, a polynomial (quadratic
    by default) is fitted SEPARATELY on the two sides of lam0 — so the fringe
    curvature is captured — each side is extrapolated to lam0 and we take
    km = R_above / R_below. It is exactly the factor that, applied to the model
    above lam0 (R[λ>lam0] *= km), matches it to the measured data.

    Unlike a free fit of km in the chi², this measurement is anchored to the data
    and does not get confused with n/k/scatt. The jump is larger at high angle
    (≈0.4 at 60°) and modest near normal incidence (≈0.85 at 8°).

    Args:
        wl, R:   λ (nm) and measured reflectivity arrays (same scale as the slider)
        lam0:    jump position (default 890)
        half:    half-width of the band used for each side (nm)
        gap:     dead zone ±gap around lam0, excluded (transition points)
        deg:     polynomial degree per side (2 = captures the fringe slope)
        km_min/km_max: safety clamp (= IR Jump slider range)

    Returns:
        km (float) clamped to [km_min, km_max], or None if the data in the band
        are insufficient.
    """
    try:
        wl = np.asarray(wl, dtype=float)
        R = np.asarray(R, dtype=float)
        below = (wl >= lam0 - half) & (wl < lam0 - gap)
        above = (wl > lam0 + gap) & (wl <= lam0 + half)
        if below.sum() < deg + 1 or above.sum() < deg + 1:
            return None
        xb = (wl[below] - lam0) / 100.0   # numerical scale of the polynomial
        xa = (wl[above] - lam0) / 100.0
        R_below = float(np.polyval(np.polyfit(xb, R[below], deg), 0.0))
        R_above = float(np.polyval(np.polyfit(xa, R[above], deg), 0.0))
        if R_below <= 1e-6:
            return None
        return float(np.clip(R_above / R_below, km_min, km_max))
    except Exception:
        return None

def ir_instr_factor(wl_ir, kjump, alpha, beta, lam0=890.0):
    """IR instrumental factor (detector jump ≈890 nm) for the λ PROVIDED (expected
    >lam0). Forward model: R[λ>lam0] *= factor. Data de-correction: R[λ>lam0] /= factor.
    SINGLE source of truth for the formula: used by the forward (calculate_refl_core)
    and by the de-corrections (fit_engine._extract_peaks, peak_window). Keeping it in
    one place avoids a beta/kb mismatch drifting the peak markers.
    The caller is responsible for the λ>lam0 mask and its own guards (km>0, angle)."""
    wl_ir = np.asarray(wl_ir, dtype=float)
    return kjump * (lam0 / wl_ir) ** alpha * np.exp(-beta * (wl_ir - lam0) / 1000.0)

# Polarization-model collapse. When [True] the forward model IGNORES
# pol/delta_n/phi_mix/phi_s1 and uses the NON-polarized isotropic model
# (R = ½|Rs|² + ½|Rp|², no S/P split, no cross-term). Birefringence is then
# rendered as Δn_eff = n_ang − n_8°; the contrast stays with
# inhom/scatt/spread/offset.
_COLLAPSE_POL = [True]
# When _COLLAPSE_POL is active, the three sliders pol, delta_n, phi_mix are the
# params of the VERTEX-FORM FRINGE-CONTRAST envelope
# V(λ) = fc0 + fc_c·((λ−λv)/1000)², with λv = 1500 + 1000·fc_v — i.e. fc0=level,
# fc_v=vertex position (controllable: λv∈[-500,3500] nm for fc_v∈[-2,2]; put it out
# of band → monotonic V), fc_c=curvature/intensity. It scales the fringe amplitude
# (empirical: beam polarization + optical path, which we don't control/quantify).
# They remain normal fit parameters (free or lockable). The
# pol/delta_n/phi_mix/phi_s1 S/P parameters are set to their neutral values.
# Birefringence is now Δn_eff (n_ang=n_8+Δn).

def calculate_refl_core(wl, d, nodes_wl, n_v, k_v, scatt, offset, inhom,
                         kjump=1.0, alpha=0.0, beta=0.0, pol=0.5, theta_deg=8,
                         delta_n=0.0, phi_mix=0.0, phi_s1=0.0):
    """
    delta_n : uniaxial birefringence Δn = n_e - n_o (optical axis ⊥ surface).
              S sees n_o = n_f,  P sees n_eff(θ) = n_o*n_e/√(n_e²cos²θ₁+n_o²sin²θ₁).
              Default 0 (isotropic): S and P see the same index.
    kjump   : multiplicative jump of the instrumental response at the
              grating/detector change at 890 nm. R(λ>890) *= ir_instr_factor(λ; kjump, alpha, beta).
              Default 1.0: no extra jump.
    """
    # pol collapse: the 3 sliders (pol/delta_n/phi_mix) become the params of the
    # vertex-form fringe-contrast envelope; the pol/delta_n/phi_mix/phi_s1 S/P params go to neutral.
    # fc0 = level, fc_v = vertex-position param, fc_c = curvature. See _COLLAPSE_POL.
    if _COLLAPSE_POL[0]:
        fc0, fc_v, fc_c = pol, delta_n, phi_mix
        pol, delta_n, phi_mix, phi_s1 = 0.5, 0.0, 0.0, 0.0
    else:
        fc0, fc_v, fc_c = 1.0, 0.0, 0.0
    try:
        f_n = PchipInterpolator(nodes_wl, n_v)
        f_k = PchipInterpolator(nodes_wl, k_v)
        n_f = np.clip(f_n(wl), 1.0, 12.0)
        k_f = np.clip(f_k(wl), 0.0, 15.0)
    except Exception:
        n_f = np.interp(wl, nodes_wl, n_v)
        k_f = np.interp(wl, nodes_wl, k_v)

    # If the film is a known material (not PSi), overwrite n/k with the tabulated values
    film_name = _FILM_MATS[_film_mat[0]] if _film_mat[0] > 0 else None
    if film_name is not None:
        nk_known = _nk_film_known(wl, film_name)
        if nk_known is not None:
            n_f = nk_known.real
            k_f = nk_known.imag
    nk_f = n_f + 1j * k_f

    # Substrate — the selected material
    nk_sub = _nk_substrate(wl, _SUBSTRATI[_substrate[0]])

    # --- Rigorous Fresnel computation ---
    theta0 = np.radians(theta_deg)
    sin0 = np.sin(theta0)
    cos0 = np.cos(theta0)

    # Snell for complex angles in the media (S polarization = ordinary index)
    sin1 = sin0 / nk_f
    cos1 = np.sqrt(1 - sin1**2 + 0j)
    sin2 = sin0 / nk_sub
    cos2 = np.sqrt(1 - sin2**2 + 0j)

    # Uniaxial birefringence (optical axis perpendicular to the surface):
    # P polarization sees n_eff(θ) instead of n_o = n_f
    if delta_n != 0.0:
        n_e = n_f + float(delta_n)           # extraordinary index
        # n_eff for extraordinary rays: standard uniaxial formula
        cos1r = np.maximum(cos1.real, 1e-6)  # real part of cos θ₁
        sin1r = np.minimum(np.abs(sin1),  1.0 - 1e-9)
        denom_p = np.sqrt((n_e * cos1r)**2 + (n_f * sin1r)**2)
        nk_p = np.where(denom_p > 1e-9, n_f * n_e / denom_p, nk_f) + 1j * k_f
    else:
        nk_p = nk_f   # isotropic: P equal to S

    # Angles for P polarization (computed with nk_p)
    sin1p = sin0 / nk_p
    cos1p = np.sqrt(1 - sin1p**2 + 0j)

    # Fresnel coefficients, air-film interface (0-1)
    rs01 = (cos0 - nk_f  * cos1 ) / (cos0 + nk_f  * cos1 )   # S uses n_o
    rp01 = (nk_p * cos0  - cos1p) / (nk_p * cos0  + cos1p)   # P uses n_eff

    # Fresnel coefficients, film-substrate interface (1-2)
    rs12 = (nk_f  * cos1  - nk_sub * cos2) / (nk_f  * cos1  + nk_sub * cos2)
    rp12 = (nk_sub * cos1p - nk_p  * cos2) / (nk_sub * cos1p + nk_p  * cos2)

    # Internal phase: S and P have different phases if delta_n != 0
    phi_s = (2 * np.pi * nk_f  * d * cos1 ) / wl
    phi_p = (2 * np.pi * nk_p  * d * cos1p) / wl

    # ── Damping factors ──
    # The angular dependence uses θ₀ (external angle), not θ₁ (internal angle,
    # which varies little for n≈2).
    #
    # Three contributions to the phase variation, summed in quadrature:
    #
    # 1. Surface roughness σ_s — Debye-Waller formula with θ₀:
    #    Δφ_s = 4π·σ_s·cos(θ₀)/λ  →  f_s = exp(-2·Δφ_s²) (amplitude factor)
    #    σ_s = (scatt/130)·(500/λ)^0.8·λ_ref/4π·cos(θ₀)  (empirical scaling)
    #    Simplified: f_s = exp(-(scatt/130)·(500/λ)^1.6·cos(θ₀)²)
    #    At θ=0: cos(θ₀)=1.
    #
    # 2. Lateral inhomogeneity σ_d — acts on the amplitude, uses θ₀:
    #    σ_d = d·(inhom/100)
    #    Δφ_d(θ₀) = 2π·n·σ_d·cos(θ₁)/λ   (phase variation per Δd)
    #    + footprint contribution: 2π·n·σ_d·(1/cos(θ₀)-1)·sin²(θ₀)/λ
    #    Combined: f_d = exp(-2·(π·n_f·σ_d/λ)²·(cos(θ₁)² + (1/cos(θ₀)-1)²))
    #    Note: (1/cos(θ₀)-1) = 0 at θ=0.
    #
    # 3. Volume scattering — depends on the path in the film:
    #    Integrated in f_scatt with path=1/cos(θ₁) but now cos(θ₀) dominates
    #    the angular dependence via the roughness term.

    cos1r  = np.maximum(cos1.real, 0.01)
    cos0c  = max(cos0, 0.01)            # cos(θ₀) scalar
    path1  = 1.0 / cos1r               # 1/cos(θ₁) — path in the film
    path0  = 1.0 / cos0c               # 1/cos(θ₀) — external geometric path

    # f_scatt: damping from surface and volume scattering
    # Uses path1 = 1/cos(θ₁) — the physical path in the film
    f_scatt = np.exp(-(scatt/130) * (500/wl)**1.6 * path1)

    # Inhomogeneity — amplitude, quadrature sum of the two contributions:
    # normal contribution (Δd variation): proportional to cos(θ₁)
    # footprint contribution (sampled area): proportional to (1/cos(θ₀)-1)
    sigma_d  = d * (inhom / 100.0)
    phi_norm = (np.pi * n_f * sigma_d * cos1r) / wl      # normal contribution
    phi_foot = (np.pi * n_f * sigma_d * (path0 - 1.0)) / wl  # footprint contribution
    f_amp    = np.exp(-2 * (phi_norm**2 + phi_foot**2))
    # At θ=0: phi_foot=0, f_amp = exp(-2·phi_norm²).
    # At θ=60°: phi_foot contributes significantly (path0=2 → phi_foot=phi_norm).

    # Phase factor: only f_scatt acts on the phase (coherent transmission)
    pf_s = np.exp(2j * phi_s) * f_scatt
    pf_p = np.exp(2j * phi_p) * f_scatt

    # Total Airy reflection for S and P (separate phases)
    Rs_tot = (rs01 + rs12 * pf_s) / (1 + rs01 * rs12 * pf_s)
    Rp_tot = (rp01 + rp12 * pf_p) / (1 + rp01 * rp12 * pf_p)

    # S/P combination with the phi_mix coherence term (active only λ<900nm)
    Rs_abs2 = np.abs(Rs_tot)**2
    Rp_abs2 = np.abs(Rp_tot)**2
    # S/P coherence term: active only for the highly-oblique angles (40°/60°) where the
    # goniometer mirrors introduce a measurable S/P relative phase.
    # At 8°/20° the optical path is different — the term is physically negligible.
    if (phi_mix != 0.0 or phi_s1 != 0.0) and theta_deg in _HOBL_ANGLES:
        pm_lam  = phi_mix + phi_s1 * (500.0 / wl)
        cross   = 2.0 * np.sqrt(np.maximum(pol * (1.0 - pol), 0.0)) * \
                  np.real(Rs_tot * np.conj(Rp_tot) * np.exp(1j * pm_lam))
        # w_cross: by default active over the WHOLE spectrum
        # (_CROSS_FULL_SPECTRUM); otherwise off for λ>1000nm (VIS gate).
        if _CROSS_FULL_SPECTRUM[0]:
            w_cross = 1.0
        else:
            w_cross = np.clip((1000.0 - wl) / 200.0, 0.0, 1.0)
        R_coh   = (pol * Rs_abs2 + (1.0 - pol) * Rp_abs2 + cross * w_cross) * 100
    else:
        R_coh   = (pol * Rs_abs2 + (1.0 - pol) * Rp_abs2) * 100

    # Inhomogeneity damping: damps only the OSCILLATIONS around the mean, without
    # shifting the phase. Mean = fringe-free reflectance (Airy with r12→0).
    # Formula: R_avg = R_mean + (R_coh - R_mean) * f_amp
    # where R_mean = interface-only reflectance (no interference).
    # Empirical fringe contrast: VERTEX-FORM λ envelope
    #   V(λ) = fc0 + fc_c·((λ − λv)/1000)²,  λv = 1500 + 1000·fc_v
    # that scales the OSCILLATING part (R_coh-R_mean). The vertex position λv is
    # directly controllable (fc_v∈[-2,2] → λv∈[-500,3500] nm): putting λv OUTSIDE
    # [λmin,λmax] makes V monotonic over the band (increasing/decreasing), with
    # fc_c is the curvature/intensity. A contrast extremum at an arbitrary in-band λ
    # has no physical justification, so the vertex is placed out of band. Default
    # fc_c=0 → V=fc0 constant. Params from the 3 freed sliders (see _COLLAPSE_POL).
    Rs_mean = np.abs(rs01)**2
    Rp_mean = np.abs(rp01)**2
    R_mean  = (pol * Rs_mean + (1.0 - pol) * Rp_mean) * 100
    f_inh   = f_amp if inhom > 0.0 else 1.0
    _lamv   = 1500.0 + 1000.0 * fc_v
    _uv     = (wl - _lamv) / 1000.0
    V_fc    = np.clip(fc0 + fc_c * _uv * _uv, 0.0, 3.0)
    R_coh_d = R_mean + V_fc * (R_coh - R_mean) * f_inh

    # Debye-Waller over all of R — damps the Si peaks too
    # f_DW = exp(-C·(scatt/130)·(500/λ)²/cos²(θ₀))
    f_DW = np.exp(-0.013 * (scatt / 130.0) * (500.0 / wl)**2
                  / max(cos0c**2, 0.1))
    R = R_coh_d * f_DW + offset

    # IR instrumental correction with a discrete jump at the detector change at 890 nm
    # Comparison with a ±5° tolerance to handle the angular-spread rays
    if any(abs(theta_deg - a) <= 5 for a in _IR_ANGLES):
        mask_ir = (wl > 890)
        if np.any(mask_ir):
            R[mask_ir] *= ir_instr_factor(wl[mask_ir], kjump, alpha, beta)

    return R, n_f, k_f


def state_to_refl_kwargs(state):
    """Dict of all scalar parameters of calculate_refl_core sourced from `state`.
    Excludes wl/d/nodes_wl/n_v/k_v (per-call) and theta_deg (set per call).
    Includes scatt/offset/inhom (positionals 6-8) and all keyword args."""
    return dict(
        scatt=state.scatt, offset=state.offs, inhom=state.inhom,
        kjump=state.km, alpha=state.ks, beta=state.kb,
        pol=state.pol, delta_n=state.dn, phi_mix=state.pm,
        phi_s1=state.ps1,
    )


def calc_refl_3angle(wl, d, ext_nd, n_v, k_v, theta, da, wc, wl_w, wr, **kwargs):
    """Three calculate_refl_core calls at theta, theta-da, theta+da; weighted blend.
    Safe normalization: if wc+wl_w+wr ≈ 0 uses tot=1.0 (avoids NaN).
    Returns (R_blended, n_f_central, k_f_central)."""
    Rc, nf, kf = calculate_refl_core(wl, d, ext_nd, n_v, k_v, theta_deg=theta,    **kwargs)
    Rl, _, _  = calculate_refl_core(wl, d, ext_nd, n_v, k_v, theta_deg=theta-da, **kwargs)
    Rr, _, _  = calculate_refl_core(wl, d, ext_nd, n_v, k_v, theta_deg=theta+da, **kwargs)
    tot = wc + wl_w + wr
    tot = tot if tot > 1e-9 else 1.0
    return (wc * Rc + wl_w * Rl + wr * Rr) / tot, nf, kf


# --- FIT-INTERNAL NODE HELPERS ---

def _build_ext_nodes(base_nodes):
    extra = []
    for i in range(len(base_nodes) - 1):
        a, b = base_nodes[i], base_nodes[i+1]
        # Uniform densification into 3 sub-intervals across the whole spectrum
        extra += [a + (b-a)/3, a + 2*(b-a)/3]
    return np.sort(np.concatenate([base_nodes, extra]))


_K_ZERO_LAMBDA = 550.0   # nm: above this, PSi is transparent and k is a residue

def _k_bounds(wl, k_bound):
    """k bounds for one node. With k_bound True (default) k is clamped to ~0
    (1e-9, 1e-6) above _K_ZERO_LAMBDA — PSi is transparent there, so a finite k is
    a calculation residue, not physical information. With k_bound False, k follows
    the graded physical cap of _node_bounds. THE single definition of the k-bound
    rule, shared by every fit path so they cannot disagree."""
    if k_bound and wl > _K_ZERO_LAMBDA:
        return (1e-9, 1e-6)
    return _node_bounds(wl)[1]


def _node_bounds(wl):
    if wl <= 375:
        return (1.0, 5.0), (0.0, 2.5)
    elif wl <= 470:
        return (1.0, 4.0), (0.0, 1.5)
    elif wl <= 520:
        return (1.0, 4.0), (0.0, 0.20)
    elif wl <= 650:
        return (1.0, 4.0), (0.0, 0.05)
    elif wl <= 900:
        return (1.0, 4.0), (0.0, 0.010)
    elif wl <= 1300:
        return (1.0, 4.0), (0.0, 0.005)
    else:
        return (1.0, 4.0), (0.0, 0.002)


def _apply_lock_nk_bounds(bounds, ext_nodes_0, c_nodes_0, n_ext0, k_ext0, state, N):
    """Apply the n/k slider locks to the fit bounds.
    - For each base-node li=0..13: if state.lock_n[li]/lock_k[li] is True, pin
      bounds[1+j] / bounds[1+N+j] to the current value, where j is the position of
      the base-node in `ext_nodes_0` (found via np.where(np.isclose...) because
      ext_nodes_0 has interleaved sub-nodes, there is no direct base→index mapping).
    - Sub-nodes between pairs of adjacent base-nodes both locked: they too are
      pinned to their current values (lock_n[li] AND lock_n[li+1] → lock of all the
      PCHIP sub-nodes between base[li] and base[li+1]).
    Compatible with `bounds` both as a Python list of tuples and as an np.array (Ntot, 2)."""
    for li in range(14):
        j = np.where(np.isclose(ext_nodes_0, c_nodes_0[li]))[0]
        if len(j):
            if state.lock_n[li]: bounds[1+j[0]]   = (float(n_ext0[j[0]]), float(n_ext0[j[0]]))
            if state.lock_k[li]: bounds[1+N+j[0]] = (float(k_ext0[j[0]]), float(k_ext0[j[0]]))
    for li in range(13):
        j0 = np.where(np.isclose(ext_nodes_0, c_nodes_0[li]))[0]
        j1 = np.where(np.isclose(ext_nodes_0, c_nodes_0[li+1]))[0]
        if len(j0) and len(j1):
            for ji in range(j0[0]+1, j1[0]):
                if state.lock_n[li] and state.lock_n[li+1]:
                    bounds[1+ji]   = (float(n_ext0[ji]), float(n_ext0[ji]))
                if state.lock_k[li] and state.lock_k[li+1]:
                    bounds[1+N+ji] = (float(k_ext0[ji]), float(k_ext0[ji]))


def _lock_film_known_bounds(bounds, ext_nodes_0, N, state=None, fix_w_at_end=False):
    """If a known film (not PSi) is selected, pin the `n` and `k` bounds in the
    `bounds` vector to the material's tabulated values at the wavelengths of
    `ext_nodes_0`. Slots touched: bounds[1+j] and bounds[1+N+j] for j=0..len(ext_nodes_0)-1.

    If `fix_w_at_end=True` (requires `state`), also pin the last 5 elements of
    `bounds` to the values state.w300/w335/w375/w400/w1100 — the `run_fit`
    convention where the W-nodes are at the tail of the parameter vector. Do NOT
    use with `run_fit_multi`: there the W-nodes are in an intermediate position.

    Compatible with `bounds` both as a Python list of tuples and as an np.array of
    shape (Ntot, 2): the assignment `bounds[i] = (a, a)` works in both.

    Returns True if the locks were applied (known film active and tables
    available), False otherwise."""
    film_idx = _film_mat[0]
    if film_idx <= 0:
        return False
    film_name = _FILM_MATS[film_idx]
    nk_known = _nk_film_known(ext_nodes_0, film_name)
    if nk_known is None:
        return False
    n_known = nk_known.real
    k_known = np.maximum(nk_known.imag, 0.0)
    for j in range(len(ext_nodes_0)):
        bounds[1+j]   = (float(n_known[j]), float(n_known[j]))
        bounds[1+N+j] = (float(k_known[j]), float(k_known[j]))
    if fix_w_at_end and state is not None:
        bounds[-5] = (state.w300,  state.w300)
        bounds[-4] = (state.w335,  state.w335)
        bounds[-3] = (state.w375,  state.w375)
        bounds[-2] = (state.w400,  state.w400)
        bounds[-1] = (state.w1100, state.w1100)
    return True


nodes_wl_base = np.array([240, 270, 300, 335, 375, 430, 470, 520, 650, 900, 1100, 1583, 2067, 2550])

_W_MIN_GAP = 1.0   # nm — the least separation between two mobile nodes

def _apply_w_nodes(c_nodes, w300, w335, w375, w400, w1100):
    """Return a float copy of `c_nodes` with the 5 mobile W-nodes replaced by the
    explicit values provided. Fixed indices: 2=w300, 3=w335, 4=w375, 5=w400,
    10=w1100. Used both by `build_c_nodes(state)` (W taken from state) and by the
    fit functions (W taken from the optimized parameter vector).

    The UV chain w300→w335→w375→w400 is kept strictly increasing. Their allowed
    ranges OVERLAP (w300 up to 320, w335 from 315; w335 up to 355, w375 from 350;
    w375 up to 400, w400 from 390), so a crossing is inside the fit's own bounds —
    and n,k are indexed by node ORDINAL, n[i] belonging to c[i], so a crossing does
    not swap two equivalent nodes: it scrambles the (λ, n) pairing. PchipInterpolator
    then refuses ("x must be strictly increasing") and calculate_refl_core's except
    branch falls back to np.interp, which does NOT complain on unsorted x and returns
    a curve off by ~8 %R without a word.
    Only that chain can collide: the fixed neighbours (270 on the left of w300, 470
    on the right of w400) lie outside every range, and w1100 (950-1200) sits between
    the fixed 900 and 1583 — so no fixed node is ever moved.
    """
    c = c_nodes.copy().astype(float)
    c[2], c[3], c[4] = w300, w335, w375
    c[5]  = w400
    c[10] = w1100
    for i in (3, 4, 5):
        if c[i] <= c[i-1] + _W_MIN_GAP:
            c[i] = c[i-1] + _W_MIN_GAP
    return c


def build_c_nodes(state):
    """Build the vector of the 14 base nodes with the 5 mobile W-nodes
    (w300, w335, w375, w400, w1100) replaced by the current values from `state`."""
    return _apply_w_nodes(nodes_wl_base, state.w300, state.w335, state.w375,
                          state.w400, state.w1100)


def _init_ext_nk(state, c_nodes_0, ext_nodes_0):
    """Initialize the n,k arrays on the extended nodes (ext_nodes_0) as the fit's
    starting point. If `state.ext_override` is available (result of a previous fit),
    linear interpolation of the saved values. Otherwise PCHIP interpolation from the
    current sliders on c_nodes_0.
    Returns (n_ext0, k_ext0) clipped to physical bounds (n∈[1,12], k≥0)."""
    if state.ext_override is not None:
        eo, n_eo, k_eo, _ = state.ext_override
        n_ext0 = np.clip(np.interp(ext_nodes_0, eo, n_eo), 1.0, 12.0)
        k_ext0 = np.maximum(np.interp(ext_nodes_0, eo, k_eo), 0.0)
    else:
        n_u = np.array([n_map(v) for v in state.n_slider_vals])
        k_u = np.array([k_map(v) for v in state.k_slider_vals])
        n_ext0 = np.clip(PchipInterpolator(c_nodes_0, n_u)(ext_nodes_0), 1.0, 12.0)
        k_ext0 = np.maximum(PchipInterpolator(c_nodes_0, k_u)(ext_nodes_0), 0.0)
    return n_ext0, k_ext0

def _parse_spectral_file(path):
    """Parse a spectral data file. Tolerant to:
    - Tab, semicolon, comma, or whitespace delimiters
    - Decimal separator '.' (US locale) OR ',' (EU locale) — the PerkinElmer
      Lambda 950 software exports in both formats depending on the PC locale.
      The files use TAB between columns, so the comma must be interpreted as the
      DECIMAL separator, not as a column delimiter (otherwise "2500,000000" gets
      split and R is read as 0 for EU-locale files).
    - Comment lines starting with '#'
    - Multi-column files: takes first two numeric columns as (wl, R)
    - Optional '#DATA' marker (lines after the marker are forced into data block,
      bypassing the wavelength-range filter used in the no-marker fallback path).

    Backward compatible with previous raw 2-col files (with or without #DATA).
    """
    wls, refls = [], []
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()
        # If the file has a '#DATA' marker, everything BEFORE it is header and
        # must be skipped. Some PerkinElmer headers contain single comma-decimal
        # values (e.g. "889,8", "2500,000000") that the no-marker fallback would
        # otherwise misread as (889, 8)/(2500, 0) data rows — corrupting the
        # spectrum and breaking the ascending order (EU-locale PerkinElmer files
        # whose header numbers fall in the 150-3500 range).
        has_marker = any(l.lstrip().startswith("#DATA") for l in lines)
        in_data_block = not has_marker
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if l.startswith("#DATA"):
                in_data_block = True
                continue
            if l.startswith("#"):
                continue
            if has_marker and not in_data_block:
                continue
            # Split on EXPLICIT column delimiters (tab/semicolon) before the
            # comma, because the comma may be the decimal separator (EU locale).
            # Inside each field the decimal comma is then normalized to a dot.
            if '\t' in l:
                parts = [p.replace(',', '.') for p in l.split('\t')]
            elif ';' in l:
                parts = [p.replace(',', '.') for p in l.split(';')]
            else:
                ws = l.split()
                if len(ws) >= 2:
                    # whitespace-delimited: comma = decimal
                    parts = [p.replace(',', '.') for p in ws]
                else:
                    # single field: maybe pure CSV with comma delimiter
                    parts = l.split(',')
            parts = [p.strip() for p in parts if p.strip() != '']
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                r = float(parts[1])
            except ValueError:
                continue
            # Inside the #DATA block always accept; without the marker, filter plausible λ
            if in_data_block or (150 <= w <= 3500):
                wls.append(w)
                refls.append(r)
    wls = np.array(wls)
    refls = np.array(refls)
    if len(wls) > 1 and wls[0] > wls[-1]:
        wls = wls[::-1]
        refls = refls[::-1]
    return wls, refls

# Startup: no dialog — starts with an empty spectrum, loaded via LOAD
_fname_init = ''
ang_init    = 8
wl_exp = np.linspace(200, 2600, 500)
r_exp  = np.zeros(500)



def n_map(x): return 1.0 + (x**3 / 80.0)
def inv_n_map(n): return np.cbrt(np.maximum(0, (n - 1.0) * 80.0))
def k_map(x): return 10**x if x > -4 else 0.0
def inv_k_map(k): return np.log10(k) if k > 1e-4 else -4.0
