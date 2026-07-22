"""
app_state.py — Shared application state for OPHIRA
Contains ONLY data and flags. No logic, no UI dependency.
All windows (main, peak analysis) share the same instance.
"""
import numpy as np
import physics as ph


class AppState:
    """Centralized application state.

    A single instance is created in main.py and passed to all windows and to
    the FitEngine.
    """

    # ── Angles ────────────────────────────────────────────────────────────────
    HOBL_ANGLES = (40, 60)      # highly-oblique angles: angular spread + Δn_eff
    IR_ANGLES  = (8, 20, 40, 60)
    DN_ANGLES  = (20, 40, 60)   # angles with birefringence in the single-angle fit
    ANG_VALS   = [8, 20, 40, 60]

    @staticmethod
    def new_angle_dict(exp=None):
        """The per-angle record — THE definition, used both here and by
        fit_engine.load_spectral_file, which rebuilds it from scratch on a new
        spectrum."""
        return {
            "exp":          exp,    # np.array([wl, R_exp])
            "fit":          None,   # np.array R_fit
            "km":           None,
            "ks":           None,
            "dn":           0.0,
            "pm":           0.0,
            "kb":           0.0,
            "final_p":      None,
            "ext_override": None,   # (ext_nodes, n, k, extra_dict)
            "idxs":         None,
            "sliders":      None,   # dict of saved slider values
            "n_maxima":     None,   # np.array([[λ,n],...]) from maxima positions
            "view_mode":    "fit",  # 'fit' | 'maxima' (main-window view)
        }

    def __init__(self):
        # ── Experimental data and fit per angle ───────────────────────────────
        self.data_angles = {ang: self.new_angle_dict() for ang in self.ANG_VALS}
        self.ang_filenames = {8: None, 20: None, 40: None, 60: None}
        # User label for the table/figures (e.g. "A_15_200_9_1" = ref / HF% /
        # current / time / analysis no.). In the table/figures only this label is shown; the
        # real file names stay in ang_filenames (provenance, saved in the session).
        self.sample_label  = ""
        self.current_ang   = 8

        # ── Sample parameter values (match the sliders) ───────────────────────
        self.d       = 1000.0   # thickness nm
        self.scatt   = 5.0
        self.inhom   = 0.0
        self.offs    = 1.0
        self.km      = 1.0      # IR jump
        self.ks      = 0.0      # IR exp
        self.kb      = 0.0      # IR damp
        self.pol     = 1.0      # fc0 = fringe-contrast level (reused slider, neutral=1)
        self.dn      = 0.0      # fc_v = fringe-contrast vertex pos λv=1500+1000·dn (reused slider, neutral=0)
        self.pm      = 0.0      # fc_c = fringe-contrast curvature (reused slider, neutral=0)
        self.ps1     = 0.0      # phi_s1 — inactive in the fit
        self.lmin    = 450.0    # minimum fit wavelength
        self.w300    = 300.0
        self.w335    = 335.0
        self.w375    = 375.0
        self.w400    = 410.0
        self.w1100   = 1100.0   # mobile IR node (950-1200 nm)

        # ── n(λ) and k(λ) nodes (14 each) ────────────────────────────────────
        # Initial slider values (in mapped scale: n_map/k_map)
        self.n_slider_vals = [ph.inv_n_map(1.75)] * 14
        self.k_slider_vals = [ph.inv_k_map(0.01)] * 14

        # ── Fit override (optimized n/k) ──────────────────────────────────────
        self.ext_override = None        # (ext_nodes, n_arr, k_arr, extra_dict)
        self.kk_cache     = None        # (wl, n_kk, k_kk) — from the 8° fit
        self.kk_metric    = None        # {'rms','max_rel','snr','lmin'} — likewise
        self.d_mem_global = None        # thickness from the second window

        # ── Slider locks ──────────────────────────────────────────────────────
        self.lock_left  = [False] * 13  # left-column sliders
        self.lock_n     = [False] * 14  # n nodes
        self.lock_k     = [False] * 14  # k nodes
        self.lock_extra = [False] * 3   # [dn, pm, ps1]

        # ── Fit flags ─────────────────────────────────────────────────────────
        self.stop                   = False
        self.iter_count             = 0
        self.iter_limit             = None   # None = no limit
        self.last_p                 = None
        self.is_stadio1             = False
        self.suppress_update        = False
        self.suppress_override_clear = False

        # ── Mode flags ────────────────────────────────────────────────────────
        self.mono_n         = True    # n monotonicity
        self.mono_k         = True    # k monotonicity
        # k-bound: in the transparency region (λ > 550 nm) PSi is essentially
        # transparent, so k there is a calculation residue rather than physical
        # information. ON (default) clamps k to ~0 (1e-9, 1e-6) above 550 nm,
        # uniformly across single- and multi-angle fits; OFF lets k float up to
        # the graded physical cap (_node_bounds). Only affects the free fit (PSi):
        # known materials have n/k locked to the table.
        self.k_bound        = True
        self.use_pred_8     = False   # show the 8° prediction
        self.kk_on          = False   # show the KK curves
        # "Thin/intermediate sample" flag: lowers the NIR peak acceptance
        # threshold from 1000 to 600 nm in calcola_d_interno, so intermediate
        # samples (d~1000) whose higher-order fringes fall in the NIR are still
        # picked up. For truly thin samples (d<500 with a SINGLE coupled peak) it
        # assumes m=1. Enable from the UI or console: state.thin_film=True
        self.thin_film      = False
        # Peak source for the d calculation:
        #   'exp'    → exp-data peaks with a prominence filter (default, more
        #              robust on data well localized in the IR fringes)
        #   'fit'    → fit peaks refined with a parabolic 3pt fit
        #   'hybrid' → fit peaks as a positional guide, exp max within a
        #              ±30 nm window, parabolic 3pt on the exp data
        # Read by calcola_d_interno (fit_auto) and peak_window for consistency.
        self.d_source       = 'exp'
        # Maxima-centering aid for LOW-CONTRAST fringes. Default OFF: the
        # centering penalty (_peak_penalty) is opt-in. With well-contrasted
        # fringes it is useless and DANGEROUS: in the deep blue (<500 nm, low
        # contrast + noisy extraction) an exp peak landing on a model trough
        # makes the concavity term (w2=1e8) explode → the optimizer damps the
        # fringes (scatt/inhom up) and the fit diverges. When ON the penalty acts
        # only for λ≥peak_pen_lam_min (500 nm), skipping the deep blue. Enable
        # ONLY on few-fringe / low-contrast samples.
        self.low_contrast_aid = False

        # ── Materials ─────────────────────────────────────────────────────────
        # Only the current selection lives here. The lists themselves belong to
        # physics, which builds them from the materials/ folder: a second copy
        # would drift, and since the two sides exchange an INDEX, the same number
        # would then mean one material in the menu and another in the engine.
        self.substrate  = 0    # index into ph._SUBSTRATI
        self.film_mat   = 0    # index into ph._FILM_MATS (0 = PSi = free fit)

    # ── Convenience accessors ─────────────────────────────────────────────────
    @property
    def substrati(self):
        return ph._SUBSTRATI

    @property
    def film_mats(self):
        return ph._FILM_MATS

    @property
    def substrate_name(self):
        return self.substrati[self.substrate]

    @property
    def film_name(self):
        return self.film_mats[self.film_mat]

    @property
    def is_psi(self):
        """True if the film is PSi (free fit of n and k)."""
        return self.film_mat == 0

    def get_angle_data(self, ang):
        return self.data_angles[ang]

    def has_data(self, ang):
        return self.data_angles[ang]["exp"] is not None

    def reset_fit_state(self):
        """Reset the fit control flags (called before every fit)."""
        self.stop        = False
        self.iter_count  = 0
        self.iter_limit  = None
        self.is_stadio1  = False

    def to_dict(self):
        """Serialize the state for session saving (pickle-safe)."""
        return {
            'data_angles':   self.data_angles,
            'ang_filenames': self.ang_filenames,
            'sample_label':  getattr(self, 'sample_label', ''),
            'current_ang':   self.current_ang,
            'd': self.d, 'scatt': self.scatt, 'inhom': self.inhom,
            'offs': self.offs, 'km': self.km, 'ks': self.ks,
            'kb': self.kb, 'pol': self.pol, 'dn': self.dn,
            'pm': self.pm, 'ps1': self.ps1, 'lmin': self.lmin,
            'w300': self.w300, 'w335': self.w335, 'w375': self.w375, 'w400': self.w400, 'w1100': self.w1100,
            'n_slider_vals': self.n_slider_vals,
            'k_slider_vals': self.k_slider_vals,
            'ext_override':  self.ext_override,
            'kk_cache':      self.kk_cache,
            'd_mem_global':  self.d_mem_global,
            'lock_left': self.lock_left, 'lock_n': self.lock_n,
            'lock_k': self.lock_k, 'lock_extra': self.lock_extra,
            'mono_n': self.mono_n, 'mono_k': self.mono_k,
            'k_bound': getattr(self, 'k_bound', True),
            # Materials by NAME: the lists follow the materials/ folder, so an
            # index would point at a different material as soon as a CSV is
            # added or removed. (Some sessions instead carry an index —
            # see from_dict.)
            'substrate_name': self.substrate_name, 'film_name': self.film_name,
            # Persist thin_film and d_source so a reloaded session keeps the
            # peak-detection behavior that was enabled in the UI before saving.
            'thin_film': getattr(self, 'thin_film', False),
            'd_source':  getattr(self, 'd_source', 'exp'),
            # Low-contrast aid: default OFF; old sessions without the key stay OFF.
            'low_contrast_aid': getattr(self, 'low_contrast_aid', False),
        }

    @classmethod
    def from_dict(cls, d):
        """Restore the state from a dict (session loading)."""
        state = cls()
        d = dict(d)
        # Materials, out of the generic loop: they are read-only properties here,
        # and the values need translating (name → position in the current list;
        # older sessions saved a position in the list of their time).
        sub_name,  sub_idx  = d.pop('substrate_name', None), d.pop('substrate', None)
        film_name, film_idx = d.pop('film_name',      None), d.pop('film_mat',  None)
        mat_refs = {
            'sub':  sub_name  if sub_name  is not None else sub_idx,
            'film': film_name if film_name is not None else film_idx,
        }
        for key, val in d.items():
            if hasattr(state, key):
                setattr(state, key, val)
        for kind, ref in mat_refs.items():
            if ref is None:
                continue
            idx = ph.resolve_material(kind, ref)
            if idx is None:
                # The material of the session is not available any more (its CSV
                # left the folder). Say so: silently fitting on another material
                # would corrupt the result.
                print(f"[app_state] WARNING: {kind} material '{ref}' of the session "
                      f"is not available → falling back to "
                      f"'{(ph._SUBSTRATI if kind=='sub' else ph._FILM_MATS)[0]}'.")
                idx = 0
            if kind == 'sub':
                state.substrate = idx
            else:
                state.film_mat = idx
        # Backward-compat with old sessions — extend all lists to the right size
        while len(state.n_slider_vals) < 14:
            state.n_slider_vals.append(ph.inv_n_map(2.0))
        while len(state.k_slider_vals) < 14:
            state.k_slider_vals.append(ph.inv_k_map(0.0))
        while len(state.lock_n) < 14:
            state.lock_n.append(False)
        while len(state.lock_k) < 14:
            state.lock_k.append(False)
        while len(state.lock_left) < 13:
            state.lock_left.append(False)
        while len(state.lock_extra) < 3:
            state.lock_extra.append(False)
        if not hasattr(state, 'w400'):  state.w400  = 410.0
        if not hasattr(state, 'w1100'): state.w1100 = 1100.0
        # Per-angle Fit/Maxima view fields: back-fill on old sessions whose
        # data_angles lacked these keys.
        for _adict in state.data_angles.values():
            _adict.setdefault('n_maxima', None)
            _adict.setdefault('view_mode', 'fit')
        return state
