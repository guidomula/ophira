"""
kk_window.py — Kramers-Kronig analysis window
Shows:
  - n(λ) vs n_KK(λ) plot over the whole KK grid
  - Δn = n - n_KK node-by-node deviation plot (extra nodes included)
  - λ_min cursor for selective RMS/max_rel computation
  - Node-by-node value table
"""
try:
    from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QGroupBox, QSizePolicy, QStatusBar,
        QLineEdit, QScrollArea)
    from PyQt6.QtCore import Qt, QTimer
    PYQT = 6
except ImportError:
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QGroupBox, QSizePolicy, QStatusBar,
        QLineEdit, QScrollArea)
    from PyQt5.QtCore import Qt, QTimer
    PYQT = 5

import numpy as np
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import physics as ph


class KKWindow(QMainWindow):
    """Kramers-Kronig analysis window with plot and selective metric."""

    def __init__(self, state, engine, parent=None):
        super().__init__(parent)
        self.state  = state
        self.engine = engine
        self.setWindowTitle("OPHIRA — KK Analysis")
        self.resize(1100, 700)

        # Reference to the main window's n sliders (for set_value)
        self._sliders_n = []
        if parent is not None and hasattr(parent, '_sliders_n'):
            self._sliders_n = parent._sliders_n

        # Display wavelength range (view + SAVE CURVES). KK itself is ALWAYS computed
        # on the FULL spectrum (the transform is non-local); this only limits what is
        # shown / exported.
        self._view_lo = 200.0
        self._view_hi = 2600.0

        # Current KK data (computed on open)
        self._wl_kk   = None   # dense KK grid (nm)
        self._n_kk    = None   # n_KK on the dense grid
        self._n_fit   = None   # n_fit interpolated on the same grid
        self._wl_nodes = None  # node positions (base + extra)
        self._dn_nodes = None  # Δn at the nodes
        self._n_nodes  = None  # n_fit at the nodes
        self._nkk_nodes= None  # n_KK at the nodes
        self._sigma_kk = None  # local n_KK noise
        self._sigma_kk_wl = None

        self._build_ui()
        self._compute_and_refresh()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.canvas.draw_idle)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # NB: no QScrollArea here. The matplotlib canvas redraws on every mouse
        # move (crosshair); inside an auto-resizing scroll those per-move redraws
        # make the plots "dance"/grow on mouse-over.
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(8)

        # ── Plot panel (left) ─────────────────────────────────────────────────
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding if PYQT==6 else QSizePolicy.Expanding,
            QSizePolicy.Policy.Expanding if PYQT==6 else QSizePolicy.Expanding)
        # Crosshair label — must be created BEFORE being added to the layout
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(
            "color: #1A237E; font-size: 9pt; font-family: Courier; "
            "background: #F3F4FF; padding: 1px 6px; border-radius: 3px;")
        self._cursor_label.setFixedHeight(20)
        # This readout is a QLabel in the same column as the canvas; with no width bound
        # its long text would make the left column (and thus the canvas/plots) grow
        # horizontally. Ignore its width in the layout so it takes the canvas width and
        # clips overflow instead.
        self._cursor_label.setSizePolicy(
            QSizePolicy.Policy.Ignored if PYQT == 6 else QSizePolicy.Ignored,
            QSizePolicy.Policy.Fixed if PYQT == 6 else QSizePolicy.Fixed)

        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setSpacing(2)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self._cursor_label)
        main.addWidget(left_col, stretch=4)

        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1],
                                   hspace=0.35, top=0.95, bottom=0.1,
                                   left=0.09, right=0.97)
        self.ax_n  = self.fig.add_subplot(gs[0])   # n vs n_KK
        self.ax_dn = self.fig.add_subplot(gs[1])   # Δn node by node

        self.ax_n.set_ylabel("Refractive index, n")
        self.ax_n.set_xlabel("Wavelength (nm)")
        self.ax_n.grid(True, alpha=0.2)
        self.ax_n.set_title("n(λ) — fit vs KK", fontsize=10)

        self.ax_dn.set_ylabel("Δn = n−n_KK")
        self.ax_dn.set_xlabel("Wavelength (nm)")
        self.ax_dn.grid(True, alpha=0.2)
        self.ax_dn.axhline(0, color='gray', lw=0.7, ls='--')
        self.ax_dn.set_title("Node-by-node deviation", fontsize=10)

        # Initial lines
        self.l_n_fit,  = self.ax_n.plot([], [], 'darkgreen', lw=2,   label='n fit')
        self.l_n_kk,   = self.ax_n.plot([], [], '#006400',   lw=1.5, ls='--', label='n KK')
        self.l_n_dots, = self.ax_n.plot([], [], 'og', markersize=5,  label='nodes')
        # Uncertainty bands (empty initially)
        self._band_prop  = [self.ax_n.fill_between([], [], [], alpha=0)]  # not in legend
        self._band_local = [self.ax_n.fill_between([], [], [],
                             color='royalblue', alpha=0.12, label='±σ_KK local')]
        self.ax_n.legend(fontsize=8)

        self.l_dn_stem_data = []   # created dynamically
        self.l_dn_line, = self.ax_dn.plot([], [], 'b-', lw=0.7, alpha=0.4)

        # λ_min vertical line (RMS threshold)
        self._vline_lmin = self.ax_n.axvline(x=420, color='tomato',
                                              ls=':', lw=1.5, alpha=0.7,
                                              label='λ_min RMS')
        self._vline_lmin2 = self.ax_dn.axvline(x=420, color='tomato',
                                                ls=':', lw=1.5, alpha=0.7)

        # Metric text in the n plot
        self.txt_metric = self.ax_n.text(
            0.02, 0.05, '', transform=self.ax_n.transAxes,
            fontsize=9, color='#1B5E20', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.85))

        # Crosshair vertical lines (invisible initially)
        self._ch_n  = self.ax_n.axvline(x=0,  color='gray', lw=0.7, ls='--', alpha=0.6, visible=False)
        self._ch_dn = self.ax_dn.axvline(x=0, color='gray', lw=0.7, ls='--', alpha=0.6, visible=False)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event', lambda e: self._cursor_label.setText(""))

        # ── Controls panel (right) ────────────────────────────────────────────
        # Right control panel inside a scroll area. Its content (RMS range, display
        # range, metric, node table) can exceed a laptop screen height; without the
        # scroll the window minimum height would exceed the screen and overflow.
        # Scrolling bounds the window minimum. Safe here, unlike the canvas: these are
        # static widgets that do NOT redraw on mouse-move, so there is no "dancing".
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        # FIXED width (standard for a control panel): the panel keeps a constant width,
        # so resizing the window grows only the plots — no re-flow lag and the panel is
        # always shown in full.
        right_scroll.setFixedWidth(300)
        right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff if PYQT == 6 else Qt.ScrollBarAlwaysOff)
        main.addWidget(right_scroll)

        # λ_min threshold for RMS
        g1 = QGroupBox("RMS range")
        v1 = QVBoxLayout(g1)
        v1.addWidget(QLabel("λ_min for RMS/max (nm):"))
        # Default λ_min = 420 nm (= monotonicity boundary; below 420 PSi has anomalous
        # dispersion → outside the KK test). Adjustable for exploration, but the value
        # used is logged (state.kk_metric['lmin']).
        self._lmin_edit = QLineEdit("420")
        self._lmin_edit.setFixedHeight(28)
        self._lmin_edit.editingFinished.connect(self._on_lmin_changed)
        v1.addWidget(self._lmin_edit)

        # λ_min slider (250–600 nm)
        self._lmin_slider = QSlider(
            Qt.Orientation.Horizontal if PYQT==6 else Qt.Horizontal)
        self._lmin_slider.setMinimum(200)
        self._lmin_slider.setMaximum(600)
        self._lmin_slider.setValue(420)
        self._lmin_slider.setTickInterval(50)
        self._lmin_slider.valueChanged.connect(self._on_lmin_slider)
        v1.addWidget(self._lmin_slider)

        # Show λ_max (fixed, 2550 nm)
        self._lbl_range = QLabel("λ_max: 2550 nm (fixed)")
        self._lbl_range.setStyleSheet("color:gray; font-size:8pt;")
        v1.addWidget(self._lbl_range)

        # n_KK smoothing (Savitzky-Golay)
        v1.addWidget(QLabel("Smoothing window (pts, odd):"))
        hw = QHBoxLayout()
        self._smooth_edit = QLineEdit("1")
        self._smooth_edit.setFixedWidth(50)
        self._smooth_edit.setFixedHeight(26)
        self._smooth_edit.editingFinished.connect(self._on_smooth_changed)
        self._lbl_smooth_info = QLabel("(1 = no smooth)")
        self._lbl_smooth_info.setStyleSheet("color:gray; font-size:8pt;")
        hw.addWidget(self._smooth_edit); hw.addWidget(self._lbl_smooth_info)
        v1.addLayout(hw)
        right.addWidget(g1)

        # Display wavelength range (view only — KK is computed on the full spectrum).
        # Limits what the plots show and what SAVE CURVES exports, e.g. to match the
        # Scout comparison interval. Fields (exact, reproducible) linked to sliders.
        _mk_slider = lambda *_: (
            QSlider(Qt.Orientation.Horizontal if PYQT == 6 else Qt.Horizontal))
        gV = QGroupBox("Display range (λ, view/export)")
        vV = QVBoxLayout(gV)
        hmin = QHBoxLayout(); hmin.addWidget(QLabel("min (nm):"))
        self._view_min_edit = QLineEdit("200"); self._view_min_edit.setFixedHeight(26)
        self._view_min_edit.editingFinished.connect(self._on_view_min_edit)
        hmin.addWidget(self._view_min_edit); vV.addLayout(hmin)
        self._view_min_slider = _mk_slider(200, 2600, 200)
        self._view_min_slider.setMinimum(200); self._view_min_slider.setMaximum(2600)
        self._view_min_slider.setValue(200)
        self._view_min_slider.valueChanged.connect(self._on_view_min_slider)
        vV.addWidget(self._view_min_slider)
        hmax = QHBoxLayout(); hmax.addWidget(QLabel("max (nm):"))
        self._view_max_edit = QLineEdit("2600"); self._view_max_edit.setFixedHeight(26)
        self._view_max_edit.editingFinished.connect(self._on_view_max_edit)
        hmax.addWidget(self._view_max_edit); vV.addLayout(hmax)
        self._view_max_slider = _mk_slider(200, 2600, 2600)
        self._view_max_slider.setMinimum(200); self._view_max_slider.setMaximum(2600)
        self._view_max_slider.setValue(2600)
        self._view_max_slider.valueChanged.connect(self._on_view_max_slider)
        vV.addWidget(self._view_max_slider)
        _bfull = QPushButton("Full range")
        _bfull.clicked.connect(self._on_view_full)
        vV.addWidget(_bfull)
        _lbl_v = QLabel("(view only; full-spectrum KK)")
        _lbl_v.setStyleSheet("color:gray; font-size:8pt;")
        vV.addWidget(_lbl_v)
        right.addWidget(gV)

        # Textual metric — includes σ_KK
        g2 = QGroupBox("KK self-consistency metric")
        v2 = QVBoxLayout(g2)
        self.lbl_rms    = QLabel("RMS(n−n_KK): --")
        self.lbl_maxrel = QLabel("max|Δn|/n: --")
        self.lbl_sigma  = QLabel("σ_KK (noise): --")
        self.lbl_snr    = QLabel("Δn/σ_KK: --")
        for lbl in (self.lbl_rms, self.lbl_maxrel):
            lbl.setStyleSheet("font-size:10pt; font-weight:bold; color:#00E676;")
        for lbl in (self.lbl_sigma, self.lbl_snr):
            lbl.setStyleSheet("font-size:10pt; font-weight:bold; color:#CE93D8;")
        v2.addWidget(self.lbl_rms)
        v2.addWidget(self.lbl_maxrel)
        v2.addWidget(_lbl_sep := QLabel("— KK noise —"))
        _lbl_sep.setStyleSheet("color:#AAAAAA; font-size:9pt;")
        v2.addWidget(self.lbl_sigma)
        v2.addWidget(self.lbl_snr)
        v2.addWidget(_lbl_note := QLabel("(computed on λ_min → 2550 nm)"))
        _lbl_note.setStyleSheet("color:#AAAAAA; font-size:8pt;")
        right.addWidget(g2)

        # Node table (scroll) with the σ_KK column
        g3 = QGroupBox("Node values")
        v3 = QVBoxLayout(g3)
        hdr = QHBoxLayout()
        for h in ["λ (nm)", "n_fit", "n_KK", "Δn", "σ_KK", "Δn/σ"]:
            lbl = QLabel(h)
            lbl.setStyleSheet("font-weight:bold; font-size:8pt; color:white;")
            hdr.addWidget(lbl)
        v3.addLayout(hdr)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._table_widget = QWidget()
        self._table_layout = QVBoxLayout(self._table_widget)
        self._table_layout.setSpacing(1)
        scroll.setWidget(self._table_widget)
        scroll.setMinimumHeight(180)
        v3.addWidget(scroll)
        right.addWidget(g3)

        right.addStretch()

        # Actions — a HORIZONTAL bar BELOW the plots (left column), always visible
        # (rather than in the scrolling right panel, where they could be scrolled
        # out of view).
        btn_bar = QHBoxLayout()
        for _text, _bg, _slot in [
            ("RECALCULATE",    "#AED6F1", self._compute_and_refresh),
            ("APPLY KK→NODES", "#A9DFBF", self._apply_kk_uv),
            ("SAVE CURVES",    "#D7BDE2", self._save_curves_csv),
            ("SAVE ALL",       "#C39BD3", self._save_csv),
            ("EXPORT FIG",     "#FFE082", self._on_export_figure),
            ("CLOSE",          "#FADBD8", self.close),
        ]:
            _b = QPushButton(_text)
            _b.setStyleSheet(f"background:{_bg}; color:black; font-weight:bold; font-size:10pt;")
            _b.setFixedHeight(28)          # match the main-window buttons (10 pt, 28 px high)
            _b.clicked.connect(_slot)
            btn_bar.addWidget(_b)
        left_layout.addLayout(btn_bar)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _on_export_figure(self):
        """Export the KK window figure (n vs n_KK + Δn).
        Formats: vector PDF (default), PNG 600 DPI, vector SVG.
        """
        from .export_fig import export_figure
        path = export_figure(
            self.fig, parent=self, default_name="kk_check",
            title="Export KK figure",
            metadata={"Title": "OPHIRA Kramers-Kronig check",
                      "Creator": "OPHIRA"})
        if path:
            self.status.showMessage(f"Figure exported: {path}")

    # ── Computation and refresh ───────────────────────────────────────────────

    def _compute_and_refresh(self):
        """Compute n_KK and Δn on all nodes, refresh the plot and table."""
        from scipy.interpolate import PchipInterpolator
        s = self.state

        # Build the current n/k (override or sliders)
        if s.ext_override is not None:
            ext_nd, n_arr, k_arr, _ = s.ext_override
        else:
            c_nd = ph.build_c_nodes(s)
            curr_n = np.array([ph.n_map(v) for v in s.n_slider_vals])
            curr_k = np.array([ph.k_map(v) for v in s.k_slider_vals])
            ext_nd = ph._build_ext_nodes(c_nd)
            n_arr  = np.clip(PchipInterpolator(c_nd, curr_n)(ext_nd), 1.0, 12.0)
            k_arr  = np.maximum(PchipInterpolator(c_nd, curr_k)(ext_nd), 0.0)

        # Compute KK
        result = self.engine.calcola_kk(ext_nd, n_arr, k_arr)
        if result is None:
            self.status.showMessage("KK calculation failed"); return
        wl_kk, n_kk_raw, k_kk = result
        s.kk_cache = (wl_kk, n_kk_raw, k_kk)

        # Savitzky-Golay smoothing on n_KK
        n_kk = self._apply_smooth(wl_kk, n_kk_raw)

        # σ_KK: local standard deviation of the raw n_KK (sliding window)
        # Measures the intrinsic noise of the KK computation, independent of smoothing
        win_std = 15   # window points (about ±7 points, ~10-15 nm typical)
        n_kk_raw_arr = self.state.kk_cache[1]   # raw n_KK from the cache
        sigma_arr = np.zeros(len(n_kk_raw_arr))
        half = win_std // 2
        for i in range(len(n_kk_raw_arr)):
            lo = max(0, i - half)
            hi = min(len(n_kk_raw_arr), i + half + 1)
            sigma_arr[i] = float(np.std(n_kk_raw_arr[lo:hi]))
        self._sigma_kk_wl = wl_kk.copy()
        self._sigma_kk    = sigma_arr

        # n_fit interpolated on the same KK grid
        n_fit_on_kk = np.interp(wl_kk, ext_nd, n_arr)

        # Δn on all nodes (base + extra)
        n_nodes    = np.interp(ext_nd, ext_nd, n_arr)    # = n_arr
        nkk_nodes  = np.interp(ext_nd, wl_kk, n_kk)
        dn_nodes   = n_nodes - nkk_nodes

        # σ_KK propagated from σ_k (analytic uncertainty from the fit)
        self._sigma_prop_wl = None
        self._sigma_prop    = None
        if hasattr(self.state, '_last_sigma_k') and self.state._last_sigma_k is not None:
            sigma_k_ext = self.state._last_sigma_k
            ext_sigma   = getattr(self.state, '_last_sigma_ext', ext_nd)
            if len(sigma_k_ext) == len(ext_sigma):
                # n_arr and k_arr on the base nodes (interpolate from the extended nodes)
                n_base = np.interp(ext_nd, ext_sigma, np.interp(ext_sigma, ext_nd, n_arr))
                k_base = np.interp(ext_nd, ext_sigma, np.interp(ext_sigma, ext_nd, k_arr))
                sk_base = np.interp(ext_nd, ext_sigma, np.maximum(sigma_k_ext, 0.0))
                try:
                    # pass the transform already computed above (RAW n_KK: the
                    # Jacobian belongs to the transform, not to the smoothing)
                    wl_prop, sig_prop = self.engine.calcola_kk_uncertainty(
                        ext_nd, n_base, k_base, sk_base,
                        kk=(wl_kk, n_kk_raw, k_kk))
                    self._sigma_prop_wl = wl_prop
                    self._sigma_prop    = sig_prop
                except Exception as e:
                    print(f"[KK uncertainty propagation] {e}")
        self._n_fit    = n_fit_on_kk
        self._wl_kk    = wl_kk
        self._n_kk     = n_kk          # n_KK smoothed (or raw if smooth=1)
        self._wl_nodes = ext_nd
        self._n_nodes  = n_nodes
        self._nkk_nodes= nkk_nodes
        self._dn_nodes = dn_nodes

        self._update_plot()
        self._update_table()
        self._update_metric()
        self.status.showMessage(f"KK computed: {len(ext_nd)} nodes, {len(wl_kk)} grid points")

    def _get_lmin(self):
        try:
            v = float(self._lmin_edit.text())
            return max(200.0, min(600.0, v))
        except ValueError:
            return 250.0

    def _get_smooth_window(self):
        """Return the smoothing window (odd, ≥1)."""
        try:
            v = int(self._smooth_edit.text())
            v = max(1, v)
            if v % 2 == 0: v += 1   # must be odd
            return v
        except ValueError:
            return 1

    def _apply_smooth(self, wl, n_kk):
        """Apply Savitzky-Golay to n_KK if the window > 1."""
        win = self._get_smooth_window()
        if win <= 1:
            return n_kk.copy()
        from scipy.signal import savgol_filter
        # The window cannot exceed the signal length
        win = min(win, len(n_kk) if len(n_kk) % 2 == 1 else len(n_kk)-1)
        win = max(3, win)
        if win % 2 == 0: win += 1
        try:
            return savgol_filter(n_kk, window_length=win, polyorder=3)
        except Exception:
            return n_kk.copy()

    def _on_smooth_changed(self):
        win = self._get_smooth_window()
        self._smooth_edit.setText(str(win))
        self._lbl_smooth_info.setText(f"({'no smooth' if win==1 else 'SG poly=3'})")
        if self._wl_kk is not None and self.state.kk_cache is not None:
            # Recompute only the smoothed n_KK — sigma stays the local std (invariant)
            n_kk_raw = self.state.kk_cache[1]
            n_kk = self._apply_smooth(self._wl_kk, n_kk_raw)
            self._n_kk = n_kk
            nkk_nodes = np.interp(self._wl_nodes, self._wl_kk, n_kk)
            self._nkk_nodes = nkk_nodes
            self._dn_nodes  = self._n_nodes - nkk_nodes
            self._update_plot()
            self._update_metric()
            self._update_table()

    def _update_metric(self):
        if self._wl_kk is None: return
        lmin = self._get_lmin()
        mask = (self._wl_nodes >= lmin) & (self._wl_nodes <= 2550)
        if not np.any(mask):
            self.lbl_rms.setText("RMS: -- (no nodes in range)")
            self.lbl_maxrel.setText("max|Δn|/n: --")
            self.lbl_sigma.setText("σ_KK: --")
            self.lbl_snr.setText("Δn/σ_KK: --")
            return
        dn  = self._dn_nodes[mask]
        n   = np.maximum(self._n_nodes[mask], 0.1)
        rms = float(np.sqrt(np.mean(dn**2)))
        mxr = float(np.max(np.abs(dn) / n))

        # σ_KK: mean n_KK noise at the nodes in the interval
        sigma_at_nodes = np.interp(self._wl_nodes[mask], self._sigma_kk_wl, self._sigma_kk)
        sigma_mean = float(np.mean(sigma_at_nodes))
        sigma_max  = float(np.max(sigma_at_nodes))
        # Δn/σ_KK ratio: how significant the difference is relative to the noise
        snr = rms / sigma_mean if sigma_mean > 1e-8 else float('inf')

        self.lbl_rms.setText(f"RMS(n−n_KK): {rms:.5f}")
        self.lbl_maxrel.setText(f"max|Δn|/n: {mxr:.5f}")
        self.lbl_sigma.setText(f"σ_KK local mean/max: {sigma_mean:.5f} / {sigma_max:.5f}")
        self.lbl_snr.setText(f"RMS/σ_KK: {snr:.2f}{'  ✓ significant' if snr > 3 else '  ~ noise level'}")

        self.txt_metric.set_text(
            f"λ≥{lmin:.0f}nm:  RMS={rms:.4f}  max|Δn|/n={mxr:.4f}  "
            f"σ_local={sigma_mean:.4f}  RMS/σ={snr:.1f}")
        self.state.kk_metric = {'rms': rms, 'max_rel': mxr,
                                 'sigma_kk': sigma_mean, 'snr': snr, 'lmin': lmin}
        self.canvas.draw_idle()

    def _update_plot(self):
        if self._wl_kk is None: return
        lmin = self._get_lmin()

        # n vs n_KK plot
        self.l_n_fit.set_data(self._wl_kk, self._n_fit)
        self.l_n_kk.set_data(self._wl_kk,  self._n_kk)
        # Nodes: show only the 14 base nodes as filled green,
        # extra nodes as open circles
        c_nd = ph.build_c_nodes(self.state)
        n_base = np.interp(c_nd, self._wl_nodes, self._n_nodes)
        self.l_n_dots.set_data(c_nd, n_base)

        # Extra dots (non-base nodes)
        mask_extra = np.ones(len(self._wl_nodes), dtype=bool)
        for wl in c_nd:
            idx = np.argmin(np.abs(self._wl_nodes - wl))
            mask_extra[idx] = False
        if not hasattr(self, 'l_n_extra'):
            self.l_n_extra, = self.ax_n.plot([], [], 'o', markerfacecolor='none',
                                              markeredgecolor='darkgreen',
                                              markersize=4, alpha=0.6,
                                              label='extra nodes')
            self.ax_n.legend(fontsize=8)
        self.l_n_extra.set_data(self._wl_nodes[mask_extra],
                                self._n_nodes[mask_extra])

        self.ax_n.relim(); self.ax_n.autoscale_view()
        self.ax_n.set_xlim(self._view_lo, self._view_hi)

        # Uncertainty bands on n_KK
        try:
            self._band_prop[0].remove()
            self._band_local[0].remove()
        except Exception: pass
        self._band_prop[0] = self.ax_n.fill_between([], [], [], alpha=0)
        # σ_local (light blue) — local std of the KK noise
        if self._sigma_kk is not None:
            self._band_local[0] = self.ax_n.fill_between(
                self._wl_kk,
                self._n_kk - self._sigma_kk,
                self._n_kk + self._sigma_kk,
                color='royalblue', alpha=0.15, label='±σ_KK local')

        # Δn node-by-node plot — stem plot via vertical lines
        for line in self.l_dn_stem_data:
            try: line.remove()
            except Exception: pass
        self.l_dn_stem_data = []
        for i, (wl, dn) in enumerate(zip(self._wl_nodes, self._dn_nodes)):
            col = '#C0392B' if wl < lmin else '#2471A3'
            ln, = self.ax_dn.plot([wl, wl], [0, dn], color=col, lw=2, alpha=0.8)
            mk, = self.ax_dn.plot([wl], [dn], 'o', color=col, markersize=5)
            self.l_dn_stem_data += [ln, mk]

        # Δn connecting line
        self.l_dn_line.set_data(self._wl_nodes, self._dn_nodes)
        self.ax_dn.relim(); self.ax_dn.autoscale_view()
        self.ax_dn.set_xlim(self._view_lo, self._view_hi)

        # λ_min line
        self._vline_lmin.set_xdata([lmin, lmin])
        self._vline_lmin2.set_xdata([lmin, lmin])

        self.canvas.draw_idle()

    def _update_table(self):
        """Refresh the node-by-node table."""
        if self._wl_nodes is None: return
        # Remove the old rows
        while self._table_layout.count():
            child = self._table_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        lmin = self._get_lmin()
        c_nd = ph.build_c_nodes(self.state)

        for wl in c_nd:
            n_f  = float(np.interp(wl, self._wl_nodes, self._n_nodes))
            n_k  = float(np.interp(wl, self._wl_kk,    self._n_kk))
            dn   = n_f - n_k
            sig  = float(np.interp(wl, self._sigma_kk_wl, self._sigma_kk))
            snr  = abs(dn)/sig if sig > 1e-8 else float('inf')
            in_range = wl >= lmin

            row = QHBoxLayout()
            for txt, col, w in [
                (f"{wl:.0f}",   '#333333', 55),
                (f"{n_f:.4f}",  '#1A5276', 62),
                (f"{n_k:.4f}",  '#1B5E20', 62),
                (f"{dn:+.4f}",  '#C0392B' if abs(dn)>2*sig else '#117A65', 62),
                (f"{sig:.4f}",  '#7B1FA2', 62),
                (f"{snr:.1f}",  '#C0392B' if snr>2 else '#555555', 45),
            ]:
                lbl = QLabel(txt)
                lbl.setStyleSheet(f"color:{col}; font-size:8pt; font-family:Courier;")
                lbl.setFixedWidth(w)
                row.addWidget(lbl)
            row_w = QWidget()
            row_w.setLayout(row)
            row_w.setStyleSheet(
                "background:#F0FFF0;" if in_range else "background:#FFF9F0;")
            self._table_layout.addWidget(row_w)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_lmin_slider(self, val):
        self._lmin_edit.setText(str(val))
        self._on_lmin_changed()

    def _on_lmin_changed(self):
        try:
            val = float(self._lmin_edit.text())
            val = max(200.0, min(600.0, val))
            self._lmin_slider.blockSignals(True)
            self._lmin_slider.setValue(int(val))
            self._lmin_slider.blockSignals(False)
        except ValueError:
            pass
        self._update_plot()
        self._update_metric()
        self._update_table()

    # ── display λ range (view/export only; KK stays full-spectrum) ─────────────
    def _apply_view_range(self):
        self.ax_n.set_xlim(self._view_lo, self._view_hi)
        self.ax_dn.set_xlim(self._view_lo, self._view_hi)
        if getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()

    def _set_view(self, lo, hi):
        lo = max(200.0, min(lo, 2580.0))
        hi = max(lo + 20.0, min(hi, 2600.0))     # keep min < max with a 20 nm floor
        self._view_lo, self._view_hi = lo, hi
        for w, v in ((self._view_min_slider, lo), (self._view_max_slider, hi)):
            w.blockSignals(True); w.setValue(int(round(v))); w.blockSignals(False)
        self._view_min_edit.setText(f"{lo:.0f}")
        self._view_max_edit.setText(f"{hi:.0f}")
        self._apply_view_range()

    def _on_view_min_edit(self):
        try: v = float(self._view_min_edit.text())
        except ValueError: v = self._view_lo
        self._set_view(v, self._view_hi)

    def _on_view_max_edit(self):
        try: v = float(self._view_max_edit.text())
        except ValueError: v = self._view_hi
        self._set_view(self._view_lo, v)

    def _on_view_min_slider(self, val):
        self._set_view(float(val), self._view_hi)

    def _on_view_max_slider(self, val):
        self._set_view(self._view_lo, float(val))

    def _on_view_full(self):
        self._set_view(200.0, 2600.0)

    def _on_mouse_move(self, event):
        """Crosshair and curve-value readout in the KK window."""
        if event.inaxes not in (self.ax_n, self.ax_dn) or self._wl_kk is None:
            self._ch_n.set_visible(False)
            self._ch_dn.set_visible(False)
            self.canvas.draw_idle()
            return
        x = event.xdata
        if x is None: return
        self._ch_n.set_xdata([x, x]);  self._ch_n.set_visible(True)
        self._ch_dn.set_xdata([x, x]); self._ch_dn.set_visible(True)

        parts = [f"λ = {x:.1f} nm"]
        n_f = float(np.interp(x, self._wl_kk, self._n_fit))
        n_k = float(np.interp(x, self._wl_kk, self._n_kk))
        dn  = n_f - n_k
        dn_over_n = dn / n_f if abs(n_f) > 1e-6 else 0.0
        parts.append(f"n_fit={n_f:.4f}")
        parts.append(f"n_KK={n_k:.4f}")
        parts.append(f"Δn={dn:+.4f}")
        parts.append(f"Δn/n={dn_over_n:+.4f}")
        if self._sigma_kk is not None:
            sig = float(np.interp(x, self._sigma_kk_wl, self._sigma_kk))
            snr = abs(dn)/sig if sig > 1e-8 else float('inf')
            parts.append(f"σ_KK={sig:.4f}")
            parts.append(f"Δn/σ={snr:.1f}")
        if self._wl_nodes is not None:
            idx = int(np.argmin(np.abs(self._wl_nodes - x)))
            dn_node = self._dn_nodes[idx]
            n_node  = self._n_nodes[idx]
            dn_n_node = dn_node / n_node if abs(n_node) > 1e-6 else 0.0
            parts.append(f"Δn(node@{self._wl_nodes[idx]:.0f}nm)={dn_node:+.4f}  Δn/n={dn_n_node:+.4f}")
        self._cursor_label.setText("   |   ".join(parts))
        self.canvas.draw_idle()

    def _save_curves_csv(self):
        """SAVE CURVES — only the curves shown in the plot, in a single CSV.

        Lightweight output for replotting in external software: wl, n_fit,
        n_KK_smooth, n_KK ± σ_KK_local (the shown band). Smoothing window and lmin
        written as metadata in the header (parsable keys).
        """
        if self._wl_kk is None:
            self.status.showMessage("Compute KK first"); return
        try:
            from PyQt6.QtWidgets import QFileDialog
        except ImportError:
            from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save KK curves", "", "CSV (*.csv);;All (*)")
        if not path: return

        import csv, os
        s = self.state
        lmin = self._get_lmin()
        smooth_win = self._get_smooth_window()
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["# OPHIRA — KK curves (plot data)"])
            w.writerow([f"# smooth_window={smooth_win}  lmin_RMS_nm={lmin:.0f}"
                        f"  view_nm={self._view_lo:.0f}-{self._view_hi:.0f}"])
            if hasattr(s, 'kk_metric') and s.kk_metric:
                km = s.kk_metric
                w.writerow([f"# RMS(n-n_KK)={km.get('rms','-'):.5f}  "
                            f"max|Dn|/n={km.get('max_rel','-'):.5f}  "
                            f"RMS/sigma_KK={km.get('snr','-'):.2f}"])
            w.writerow(["wl_nm", "n_fit", "n_KK_smooth",
                        "n_KK_minus_sigma", "n_KK_plus_sigma"])
            has_sig = self._sigma_kk is not None
            for i in range(len(self._wl_kk)):
                wl  = float(self._wl_kk[i])
                if not (self._view_lo <= wl <= self._view_hi):
                    continue                        # SAVE CURVES honours the display range
                nf  = float(self._n_fit[i])
                nk  = float(self._n_kk[i])
                sig = float(self._sigma_kk[i]) if has_sig else 0.0
                w.writerow([f"{wl:.2f}", f"{nf:.6f}", f"{nk:.6f}",
                            f"{nk-sig:.6f}", f"{nk+sig:.6f}"])
        self.status.showMessage(f"Saved curves: {os.path.basename(path)}")

    def _save_csv(self):
        """SAVE ALL — complete KK data for external reconstruction of the analysis.

        Three files:
          {base}_kk_grid.csv  — dense grid: wl, n_fit, n_KK_raw, n_KK_smooth,
                                k_input_Hilbert, σ_KK_local, σ_prop (propagated
                                from the fit's σ_k, if available), Δn
          {base}_kk_nodes.csv — base+extended nodes with Δn, σ_KK, snr, in_range
          {base}_kk_meta.csv  — parameters/metadata (smoothing, lmin, metric)

        Allows the KK analysis to be fully reconstructed in external software.
        """
        if self._wl_kk is None:
            self.status.showMessage("Compute KK first"); return
        try:
            from PyQt6.QtWidgets import QFileDialog
        except ImportError:
            from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save KK data", "", "CSV (*.csv);;All (*)")
        if not path: return

        import csv, os
        s = self.state
        lmin = self._get_lmin()
        smooth_win = self._get_smooth_window()
        base = os.path.splitext(path)[0]

        # k input to the Hilbert computation (kk_cache[2]) and raw n_KK (kk_cache[1])
        k_input_dense = None
        n_kk_raw_dense = None
        if s.kk_cache is not None:
            _, n_kk_raw_dense, k_input_dense = s.kk_cache

        # σ propagated from the fit's σ_k (if available)
        sigma_prop_on_kk = None
        if self._sigma_prop_wl is not None and self._sigma_prop is not None:
            sigma_prop_on_kk = np.interp(self._wl_kk,
                                         self._sigma_prop_wl, self._sigma_prop)

        # ── File 1: extended dense grid ─────────────────────────────────────
        grid_path = base + "_kk_grid.csv"
        with open(grid_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["# OPHIRA — KK analysis (full grid)"])
            w.writerow([f"# smooth_window={smooth_win}  lmin_RMS_nm={lmin:.0f}"])
            if hasattr(s, 'kk_metric') and s.kk_metric:
                km = s.kk_metric
                w.writerow([f"# RMS(n-n_KK)={km.get('rms','-'):.5f}  "
                            f"max|Dn|/n={km.get('max_rel','-'):.5f}  "
                            f"sigma_KK_local={km.get('sigma_kk','-'):.5f}  "
                            f"RMS/sigma={km.get('snr','-'):.2f}"])
            cols = ["wl_nm", "n_fit", "n_KK_raw", "n_KK_smooth",
                    "k_input_Hilbert", "sigma_KK_local"]
            if sigma_prop_on_kk is not None: cols.append("sigma_KK_propagated")
            cols.append("delta_n")
            w.writerow(cols)
            for i in range(len(self._wl_kk)):
                wl   = float(self._wl_kk[i])
                nf   = float(self._n_fit[i])
                nk_s = float(self._n_kk[i])
                nk_r = float(n_kk_raw_dense[i]) if n_kk_raw_dense is not None else nk_s
                kin  = float(k_input_dense[i]) if k_input_dense is not None else 0.0
                sigL = float(self._sigma_kk[i]) if self._sigma_kk is not None else 0.0
                row = [f"{wl:.2f}", f"{nf:.6f}", f"{nk_r:.6f}", f"{nk_s:.6f}",
                       f"{kin:.6e}", f"{sigL:.6f}"]
                if sigma_prop_on_kk is not None:
                    row.append(f"{float(sigma_prop_on_kk[i]):.6f}")
                row.append(f"{nf-nk_s:+.6f}")
                w.writerow(row)

        # ── File 2: nodes ───────────────────────────────────────
        nodes_path = base + "_kk_nodes.csv"
        c_nd = ph.build_c_nodes(s)
        with open(nodes_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["wl_nm", "n_fit", "n_KK", "delta_n",
                        "sigma_KK_local", "delta_n_over_sigma", "in_RMS_range"])
            for wl in c_nd:
                nf  = float(np.interp(wl, self._wl_nodes, self._n_nodes))
                nk  = float(np.interp(wl, self._wl_kk,   self._n_kk))
                dn  = nf - nk
                sig = float(np.interp(wl, self._sigma_kk_wl, self._sigma_kk)) \
                      if self._sigma_kk is not None else 0.0
                snr = abs(dn)/sig if sig > 1e-8 else 0.0
                in_range = wl >= lmin
                w.writerow([f"{wl:.1f}", f"{nf:.6f}", f"{nk:.6f}",
                            f"{dn:+.6f}", f"{sig:.6f}",
                            f"{snr:.2f}", str(in_range)])

        # ── File 3: meta (structured key/value parameters) ─────────────────
        meta_path = base + "_kk_meta.csv"
        with open(meta_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["# OPHIRA — KK analysis metadata"])
            w.writerow(["key", "value"])
            w.writerow(["smooth_window", smooth_win])
            w.writerow(["smooth_polyorder", 3])
            w.writerow(["lmin_RMS_nm", f"{lmin:.1f}"])
            if hasattr(s, 'kk_metric') and s.kk_metric:
                km = s.kk_metric
                for k in ('rms', 'max_rel', 'sigma_kk', 'snr'):
                    if k in km:
                        w.writerow([f"kk_{k}", f"{km[k]:.6f}"])
            w.writerow(["n_grid_points", len(self._wl_kk)])
            w.writerow(["n_nodes", len(c_nd)])
            w.writerow(["sigma_prop_available", sigma_prop_on_kk is not None])

        self.status.showMessage(
            f"Saved: {os.path.basename(grid_path)} + "
            f"{os.path.basename(nodes_path)} + {os.path.basename(meta_path)}")

    def _apply_kk_uv(self):
        """Apply the n_KK values to the base nodes — always works, even after a slider move.

        By construction the transfer is iterative (applying n_KK changes n, so
        ε₂=2nk changes, so the new n_KK is different). To approach the fixed point
        "n satisfies KK" without requiring several clicks, we run **3 internal
        iterations** per click:
        apply → recompute KK → apply → recompute → apply → recompute.
        """
        if self._wl_nodes is None or self._nkk_nodes is None:
            # Recompute first
            self._compute_and_refresh()
        if self._wl_nodes is None or self._wl_kk is None:
            self.status.showMessage("Compute KK first"); return

        N_ITER_INTERNAL = 3
        s = self.state
        n_applied = 0

        for _ in range(N_ITER_INTERNAL):
            c_nd = np.asarray(ph.build_c_nodes(s), dtype=float)
            wl_kk = np.asarray(self._wl_kk, dtype=float)
            n_kk = np.asarray(self._n_kk, dtype=float)
            # Update the base nodes (display + no-fit case), skipping the extremes
            s.suppress_update = True
            s.suppress_override_clear = True
            n_applied = 0
            for i, wl in enumerate(c_nd):
                # Does NOT transfer the FIRST node (UV extreme ~240nm, outside the
                # KK range and in the anomalous-dispersion zone <420nm). The LAST IR
                # node IS included: np.interp clamps to the IR-edge KK value if the
                # node exceeds the data.
                if i == 0:
                    continue
                n_kk_at_node = float(np.interp(wl, wl_kk, n_kk))
                internal = ph.inv_n_map(max(1.0, min(12.0, n_kk_at_node)))
                s.n_slider_vals[i] = internal
                if i < len(self._sliders_n):
                    self._sliders_n[i].set_value(internal)
                n_applied += 1
            s.suppress_update = False
            s.suppress_override_clear = False
            # If there is a fit, apply the KK n INSIDE ext_override (k and the ~40
            # ext-nodes PRESERVED) instead of clearing it. Clearing it would lose the
            # fitted curve and k → degraded fit, and with n locked it would stay in
            # the bad minimum. Without a fit: ext_override stays None and the
            # simulation uses the just-updated base nodes.
            ov = s.ext_override
            if ov is not None and len(ov) >= 4:
                ext_nd, n_arr, k_arr, extra = ov
                ext_nd = np.asarray(ext_nd, dtype=float)
                lo = max(float(c_nd[1]), float(wl_kk.min()))
                # c_nd[-1] to also include the last IR node (it stays capped by
                # wl_kk.max() anyway if the data don't reach there → no
                # extrapolation beyond the KK range).
                hi = min(float(c_nd[-1]), float(wl_kk.max()))
                in_kk = (ext_nd >= lo) & (ext_nd <= hi)
                n_new = np.asarray(n_arr, dtype=float).copy()
                if in_kk.any():
                    n_new[in_kk] = np.clip(
                        np.interp(ext_nd[in_kk], wl_kk, n_kk), 1.0, 12.0)
                s.ext_override = (ext_nd, n_new, k_arr, extra)
            # Recompute the KK view with the new values (updates self._n_kk)
            self._compute_and_refresh()

        # LOCK d after the KK→nodes transfer: restarting from the KK n, the next fit
        # — via the n·d degeneracy (anchor opt_thick_ref = mean(n)·d) + the landscape
        # — can move d by hundreds of nm as its first step, forcing a restart. Locking
        # d keeps it stable until it is unlocked from its slider. KK serves to fix n;
        # d has already been determined.
        s.lock_left[0] = True

        # Refresh the main plot only once at the end of the loop
        parent = self.parent()
        if parent is not None and hasattr(parent, '_refresh_simulation'):
            # Reflect the d lock on the slider (set_locked doesn't touch the state,
            # so we already set the state above).
            if hasattr(parent, '_sliders_left') and parent._sliders_left:
                parent._sliders_left[0].set_locked(True)
            parent._refresh_simulation()

        self.status.showMessage(
            f"KK→nodes applied: {n_applied} nodes × {N_ITER_INTERNAL} iter "
            f"— d locked (unlock from its slider to refit it)")
