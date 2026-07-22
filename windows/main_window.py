"""
main_window.py — OPHIRA main window

Architecture:
  - QMainWindow with an embedded matplotlib canvas (FigureCanvasQTAgg)
  - PyQt sliders (custom QSlider with n_map/k_map mapping)
  - QPushButton buttons + QComboBox menus
  - Callbacks → FitEngine (no physics logic here)
"""

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QSplitter, QPushButton, QLabel, QComboBox,
        QSlider, QGroupBox, QGridLayout, QScrollArea, QLineEdit,
        QCheckBox, QStatusBar, QSizePolicy, QToolBar, QFileDialog,
        QMessageBox, QDialog, QFormLayout, QSpinBox,
        QDialogButtonBox)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QAction, QFont
    PYQT = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QSplitter, QPushButton, QLabel, QComboBox,
        QSlider, QGroupBox, QGridLayout, QScrollArea, QLineEdit,
        QCheckBox, QStatusBar, QSizePolicy, QToolBar, QFileDialog,
        QMessageBox, QAction, QDialog, QFormLayout, QSpinBox,
        QDialogButtonBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont
    PYQT = 5

import numpy as np
import matplotlib
matplotlib.use('QtAgg' if PYQT == 6 else 'Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app_state import AppState
from fit_engine import FitEngine
import physics as ph


# ─────────────────────────────────────────────────────────────────────────────
# Thread for the fit (does not block the UI during optimization)
# ─────────────────────────────────────────────────────────────────────────────

class FitThread(QThread):
    """Run run_fit in a separate thread so as not to block the UI."""
    progress    = pyqtSignal(str)               # text message
    fit_step    = pyqtSignal(float, int,        # chi2, n_iter
                             object, object, object,  # R, nf, kf
                             object, object, object,  # ext, en, ek
                             int)                     # ang being fitted
    finished_ok = pyqtSignal(dict)              # complete result
    finished_err= pyqtSignal(str)               # error message

    def __init__(self, engine: FitEngine, ang: int,
                 wl_exp: np.ndarray, r_exp: np.ndarray):
        super().__init__()
        self.engine  = engine
        self.ang     = ang
        self.wl_exp  = wl_exp
        self.r_exp   = r_exp

    def run(self):
        self.engine.on_progress = lambda m: self.progress.emit(m)
        self.engine.on_fit_step = lambda chi2,n_iter,R,nf,kf,ext,en,ek,ang: \
            self.fit_step.emit(chi2,n_iter,R,nf,kf,ext,en,ek,ang)
        try:
            result = self.engine.run_fit(self.ang, self.wl_exp, self.r_exp)
            self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


class FitMultiThread(QThread):
    """Run run_fit_multi in a separate thread."""
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err= pyqtSignal(str)

    def __init__(self, engine: FitEngine, ang_avail: list):
        super().__init__()
        self.engine    = engine
        self.ang_avail = ang_avail

    def run(self):
        self.engine.on_progress = lambda m: self.progress.emit(m)
        try:
            result = self.engine.run_fit_multi(self.ang_avail)
            self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


class FitAutoThread(QThread):
    """Run run_fit_auto in a separate thread."""
    progress    = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err= pyqtSignal(str)

    def __init__(self, engine: FitEngine, ang_avail: list):
        super().__init__()
        self.engine    = engine
        self.ang_avail = ang_avail

    def run(self):
        self.engine.on_progress = lambda m: self.progress.emit(m)
        try:
            result = self.engine.run_fit_auto(self.ang_avail)
            self.finished_ok.emit(result)
        except Exception as e:
            self.finished_err.emit(str(e))


# Realistic first-launch curve (spectrum + parameters). Single source of truth
# shared by startup, the slider construction, and _on_reset — so RESET restores
# exactly the first-launch curve, not the flat AppState defaults. Values from a
# clean PSi fit at d≈1200 nm, chosen as the default starting curve.
# N_INIT/K_INIT = n/k at the 14 base nodes; PARAM_INIT = the scalar fit parameters
# (so the default sim reproduces that fit, not just its n/k with a generic d).
N_INIT = [1.294, 1.8238, 2.3509, 2.4708, 2.4749, 2.2364, 2.0208, 1.9319, 1.8275, 1.7610, 1.7411, 1.7249, 1.7237, 1.7204]
K_INIT = [0.8527, 1.2313, 1.0456, 0.5532, 0.4087, 0.1087, 0.0300, 0.0122, 0.0023, 0.0, 0.0, 0.0, 0.0, 0.0001]
PARAM_INIT = {
    'd': 1199.852, 'scatt': 15.6806, 'inhom': 2.2501, 'offs': 0.9578,
    'km': 0.9797, 'ks': -0.282, 'kb': 0.2234,
    'pol': 1.0, 'dn': 0.0, 'pm': 0.0, 'ps1': 0.0, 'lmin': 450.0,
    'w300': 298.7566, 'w335': 334.2253, 'w375': 374.5787,
    'w400': 410.6773, 'w1100': 1100.1321,
}
# Full fit curve (n,k at the 40 EXT nodes = 14 base + 2 sub-nodes/gap) of the same
# d≈1200 nm fit. Applied as ext_override at startup/RESET so the default sim shows
# the fine structure the 14 base nodes miss (the Si-resonance detail near 310-410 nm);
# the ext-node wavelengths are rebuilt from the base nodes (no need to store them).
# Cleared to the 14-node representation as soon as an n/k slider is moved.
EXT_N = [1.291, 1.4603, 1.6455, 1.8591, 2.0841, 2.2444, 2.3302, 2.3717, 2.4074, 2.4654,
         2.5278, 2.5347, 2.4715, 2.3776, 2.2932, 2.2395, 2.1496, 2.0797, 2.0212, 1.9888,
         1.9571, 1.9319, 1.8845, 1.8514, 1.8275, 1.7966, 1.7777, 1.7608, 1.7521, 1.7456,
         1.7413, 1.7349, 1.7285, 1.7255, 1.7256, 1.7251, 1.7212, 1.7192, 1.7136, 1.7046]
EXT_K = [0.8535, 0.959, 1.0986, 1.2456, 1.272, 1.2462, 1.0464, 0.8273, 0.6585, 0.5537,
         0.5104, 0.5019, 0.4102, 0.2921, 0.1886, 0.1096, 0.0452, 0.0376, 0.0302, 0.0235,
         0.0176, 0.0124, 0.0079, 0.0047, 0.0025, 0.0009, 0.0, 0.0, 0.0, 0.0,
         0.0003, 0.0008, 0.0013, 0.0018, 0.002, 0.0017, 0.0015, 0.001, 0.0, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """OPHIRA main window.

    Layout:
      ┌────────────────────────────────┬──────────────────┐
      │   matplotlib canvas (plots)    │  Right panel     │
      │   - Reflectance                │  - Angles        │
      │   - Residuals                  │  - Substrate     │
      │   - n/k curves                 │  - Film          │
      ├────────────────────────────────┴──────────────────┤
      │   Slider panel (left)          │   n/k sliders    │
      ├────────────────────────────────┴──────────────────┤
      │   Button bar (3 rows)                             │
      └───────────────────────────────────────────────────┘
    """

    def __init__(self, state: AppState, engine: FitEngine, parent=None):
        super().__init__(parent)
        self.state  = state
        self.engine = engine
        self._fit_thread = None

        # first-launch scalar parameters: set them BEFORE building the UI so the
        # sliders — and the startup sim — reproduce that fit (its n/k come from
        # N_INIT/K_INIT applied in the slider construction).
        for _k, _v in PARAM_INIT.items():
            setattr(self.state, _k, _v)

        from version import full_title
        self.setWindowTitle(full_title())
        # Tall enough to contain everything (content ~866px) → it opens WITHOUT a
        # scrollbar; with the QScrollArea (in _build_central_widget) the window can
        # still be shrunk below this size and the scrollbars appear.
        self.resize(1500, 940)

        # Tooltip style (button comments): readable and consistent — black text on
        # pale yellow, 10pt, with padding. Scoped to QToolTip → doesn't touch the
        # other widgets.
        self.setStyleSheet(
            "QToolTip { color:#000000; background-color:#FFFFDC; "
            "border:1px solid #999999; padding:5px; font-size:10pt; }")

        # Application icon — colored spectrum generated programmatically
        self._set_app_icon()

        # Build the UI
        self._build_plots()
        self._build_central_widget()
        self._build_status_bar()
        self._build_windows_menu()

        # Initialize the plots with empty data
        self._init_plot_data()

        # Default sim = the full fit curve (fine structure near the Si resonance),
        # not just the 14 base nodes. Applied after the plot lines exist; cleared to
        # the 14-node curve as soon as an n/k slider is moved.
        self.state.ext_override = self._default_ext_override()
        self._refresh_simulation()

        # Bring the window to the front at startup
        self.show()
        self.raise_()
        self.activateWindow()

    def _set_app_icon(self):
        """Create a colored-spectrum icon with a reflectance curve."""
        try:
            from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush
            from PyQt6.QtCore import QRect, QPoint
        except ImportError:
            from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush
            from PyQt5.QtCore import QRect, QPoint

        size = 64
        px = QPixmap(size, size)
        px.fill(QColor(30, 30, 50))          # dark blue background

        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing
                        if hasattr(QPainter, 'RenderHint') else QPainter.Antialiasing)

        # Colored spectral bands (UV→IR)
        colors = ['#8B00FF','#4400FF','#0044FF','#00AAFF',
                  '#00FF88','#AAFF00','#FFFF00','#FF8800','#FF2200']
        bw = size // len(colors)
        for i, c in enumerate(colors):
            p.fillRect(i*bw, 0, bw, size, QColor(c))

        # Dark semi-transparent overlay for readability
        p.fillRect(0, 0, size, size, QColor(0, 0, 0, 80))

        # Simulated reflectance curve (decreasing sinusoid)
        pen = QPen(QColor(255, 255, 255), 3)
        p.setPen(pen)
        import math
        pts = []
        for ix in range(size):
            t = ix / (size - 1)
            y = 0.75 - 0.45 * t + 0.18 * math.sin(t * math.pi * 5) * (1 - t)
            pts.append((ix, int((1 - y) * size)))
        for i in range(len(pts) - 1):
            p.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

        # Border
        pen2 = QPen(QColor(200, 200, 255), 2)
        p.setPen(pen2)
        p.drawRect(1, 1, size-2, size-2)
        p.end()

        self.setWindowIcon(QIcon(px))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot setup
    # ─────────────────────────────────────────────────────────────────────────

    def _build_plots(self):
        """Create the matplotlib figure with 3 subplots (reflectance, residuals, n/k)."""
        self.fig = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)

        gs = self.fig.add_gridspec(3, 1,
                                   height_ratios=[3, 1, 2],
                                   hspace=0.45,
                                   top=0.93, bottom=0.10,   # bottom=0.10 lifts the n/k x-axis labels off the panel border
                                   left=0.08, right=0.80)
        self.ax1    = self.fig.add_subplot(gs[0])
        self.ax_res = self.fig.add_subplot(gs[1])
        self.ax2    = self.fig.add_subplot(gs[2])
        self.ax2_k  = self.ax2.twinx()

        # Axes
        self.ax1.set_ylabel("Reflectance (%)")
        self.ax_res.set_ylabel("Res. (%)", fontsize=8)
        self.ax_res.axhline(0, color='gray', lw=0.6, ls='--')
        self.ax_res.grid(True, alpha=0.2)
        self.ax_res.set_ylim(-5, 5)
        self.ax2.set_ylabel("n", color='darkgreen',
                             fontweight='bold', fontsize=12)
        self.ax2_k.set_ylabel("k", color='purple',
                              fontweight='bold', fontsize=12)

        # Fit-status texts — on the RIGHT in the space between the two plots'
        # legends: they free up the plot area and remove clutter from the top-left
        # corner of the reflectance panel.
        self.txt_chi  = self.fig.text(0.995, 0.62, 'RMS resid: --',
                                      fontweight='bold', fontsize=9,
                                      ha='right', va='top', linespacing=1.4)
        self.txt_iter = self.fig.text(0.995, 0.47, 'Iter: 0',
                                      fontsize=8, color='dimgray',
                                      ha='right', va='top')
        # KK metric text in the n/k plot
        self.txt_kk_metric = self.ax2.text(
            0.02, 0.97, '', transform=self.ax2.transAxes,
            fontsize=8, color='#1B5E20', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.0))

    def _init_plot_data(self):
        """Initialize the plot lines with the default grid."""
        # Default wl grid — always available even before loading. Limited to the
        # valid n/k node range (240–2550 nm): outside it the nodes are clamped/
        # extrapolated, which would give the default (no-data) sim an unphysical low-λ
        # spike and a tail past 2550 nm.
        self._wl_default = np.linspace(float(ph.nodes_wl_base[0]),
                                       float(ph.nodes_wl_base[-1]), 500)
        wl0  = self._wl_default
        zero = np.zeros(len(wl0))

        # Realistic initial n/k values (single source of truth: N_INIT/K_INIT)
        _N_IN = N_INIT
        _K_IN = K_INIT
        nodes = ph.build_c_nodes(self.state)
        # Update AppState with the correct initial values
        for i in range(14):
            self.state.n_slider_vals[i] = ph.inv_n_map(float(_N_IN[i]))
            self.state.k_slider_vals[i] = ph.inv_k_map(float(_K_IN[i]))

        # Compute the initial curve
        from scipy.interpolate import PchipInterpolator
        n_arr = np.array(_N_IN, dtype=float)
        k_arr = np.array(_K_IN, dtype=float)
        ext0  = ph._build_ext_nodes(nodes)
        n_ext = np.clip(PchipInterpolator(nodes, n_arr)(ext0), 1.0, 12.0)
        k_ext = np.maximum(PchipInterpolator(nodes, k_arr)(ext0), 0.0)
        R0, nf0, kf0 = ph.calculate_refl_core(
            wl0, self.state.d, ext0, n_ext, k_ext,
            self.state.scatt, self.state.offs, self.state.inhom,
            kjump=self.state.km, alpha=self.state.ks, beta=self.state.kb,
            pol=self.state.pol, theta_deg=self.state.current_ang)

        self.l_sim, = self.ax1.plot(wl0, R0,  'b',   lw=2,         label='sim')
        self.l_exp, = self.ax1.plot(wl0, zero, 'r--', alpha=0.3,   label='exp')
        self.ax1.set_ylim(0, 65)
        self.ax1.set_xlim(200, 2600)
        self.ax1.legend(fontsize=9, loc='upper left',
                        bbox_to_anchor=(1.08, 1), borderaxespad=0)

        self.l_res, = self.ax_res.plot(wl0, zero, color='seagreen', lw=1.0)
        self.ax_res.set_xlim(200, 2600)

        self.l_n, = self.ax2.plot(wl0, nf0, 'darkgreen', lw=1.8, label='n')
        self.l_k, = self.ax2_k.plot(wl0, kf0, 'purple', alpha=0.4, ls='--', label='k')
        self.l_n_kk, = self.ax2.plot([], [], color='#006400',
                                      lw=1.2, ls=':', alpha=0.85, label='n_KK')
        self.l_k_kk, = self.ax2_k.plot([], [], color='#4B0082',
                                         lw=1.2, ls=':', alpha=0.85, label='k_KK')
        self.ax2.set_xlim(200, 2600)

        self.dots_n, = self.ax2.plot(nodes, n_arr, 'og', markersize=6)
        self.dots_k, = self.ax2_k.plot(nodes, k_arr, 'o', markersize=6, color='purple')
        self.extra_dots_n, = self.ax2.plot([], [], 'o', markerfacecolor='none',
                                            markeredgecolor='green', markersize=5, alpha=0.7)
        self.extra_dots_k, = self.ax2_k.plot([], [], 'o', markerfacecolor='none',
                                              markeredgecolor='purple', markersize=5, alpha=0.7)
        self._band_n = [self.ax2.fill_between([], [], [], color='darkgreen', alpha=0.15)]
        self._band_k = [self.ax2_k.fill_between([], [], [], color='purple',  alpha=0.15)]
        self.vlines  = [self.ax2.axvline(x=n, color='gray', linestyle=':',
                         linewidth=0.8, alpha=0.3) for n in nodes]

        # Combined n/k legend inside ax2 (top right, inside the plot)
        lines_n, labels_n = self.ax2.get_legend_handles_labels()
        lines_k, labels_k = self.ax2_k.get_legend_handles_labels()
        self.ax2.legend(lines_n + lines_k, labels_n + labels_k,
                        fontsize=9, loc='upper left',
                        bbox_to_anchor=(1.08, 1), borderaxespad=0)

        # ── Fit/Maxima view: artists for "n from maxima" ──────────
        # n_eff from the maxima (PCHIP line + markers) on ax2; maxima markers on
        # ax1 (reflectance); text notes (disclaimer + residual). All hidden until
        # the angle is in view_mode='maxima'. On-screen text ALWAYS in English
        # (international UI).
        self.l_n_max, = self.ax2.plot([], [], color='#0066CC', lw=1.6, ls='-',
                                      label='n from maxima', visible=False)
        self.dots_n_max, = self.ax2.plot([], [], 'o', ms=6, color='#0066CC',
                                         mec='white', mew=0.7, visible=False,
                                         label='_nolegend_')
        self.dots_peaks, = self.ax1.plot([], [], 'v', ms=7, color='#0066CC',
                                         mec='white', mew=0.7, visible=False,
                                         label='_nolegend_')
        self._maxima_note = self.ax1.text(
            0.5, 0.05, '', transform=self.ax1.transAxes, ha='center', va='bottom',
            fontsize=8, color='#7A0000', fontweight='bold', wrap=True,
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3F3', ec='#CC0000', alpha=0.9),
            visible=False, zorder=20)
        self._maxima_res_note = self.ax_res.text(
            0.5, 0.5, 'n from maxima (no fit residual)', transform=self.ax_res.transAxes,
            ha='center', va='center', fontsize=8, color='gray', style='italic',
            visible=False)
        self._DISCLAIMER = {
            60: "60°: n = n_eff from maxima positions. "
                "Fit unreliable (deformed maxima); k not shown.",
            40: "40°: n from maxima (cross-check). Fit reliable.",
            20: "",
            8:  "",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Central layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_central_widget(self):
        """Build the main layout with canvas + panels."""
        central = QWidget()
        # Scroll: lets the window shrink BELOW the content's minimum (the
        # scrollbars appear only when needed). At full size it is transparent and
        # the content fills normally (setWidgetResizable). Useful for those who
        # work with many apps and don't always want a huge window.
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QScrollArea.Shape.NoFrame if PYQT == 6
                              else QScrollArea.NoFrame)
        _scroll.setWidget(central)
        self.setCentralWidget(_scroll)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # ── matplotlib toolbar (zoom, pan, cursor coordinates) ────────────────
        self.toolbar = NavToolbar(self.canvas, central)
        self.toolbar.setMaximumHeight(28)
        from windows.widgets import style_nav_toolbar
        style_nav_toolbar(self.toolbar)
        # Trim the toolbar: drop Subplots and Save; KEEP 'Customize' (Edit axis, curve &
        # image parameters) — the manual view-parameter editor the d-scan windows expose.
        for action in self.toolbar.actions():
            if action.text() in ('Subplots', 'Save'):
                self.toolbar.removeAction(action)
        main_layout.addWidget(self.toolbar)

        # ── Cursor coordinate label (crosshair) ───────────────────────────────
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(
            "color: #1A237E; font-size: 9pt; font-family: Courier; "
            "background: #F3F4FF; padding: 1px 6px; border-radius: 3px;")
        self._cursor_label.setFixedHeight(20)
        main_layout.addWidget(self._cursor_label)

        # Connect mouse movement to the three axes
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event',
                                lambda e: self._cursor_label.setText(""))

        # Crosshair vertical lines (invisible initially)
        self._crosshair_lines = []
        for ax in (self.ax1, self.ax_res, self.ax2):
            ln = ax.axvline(x=0, color='gray', lw=0.7, ls='--', alpha=0.6, visible=False)
            self._crosshair_lines.append(ln)

        # Horizontal splitter: canvas (left) | right panel
        h_split = QSplitter(Qt.Orientation.Horizontal
                            if PYQT == 6 else Qt.Horizontal)
        h_split.addWidget(self.canvas)
        h_split.addWidget(self._build_right_panel())
        h_split.setSizes([1100, 120])
        # Limit the plot-area height: it is the effective way to lower the window
        # (the canvas sizeHint doesn't drop with the internal maximumHeight alone).
        # Without the cap the window would exceed the screen height and the last
        # button rows would fall off-screen.
        h_split.setMaximumHeight(290)
        main_layout.addWidget(h_split, stretch=5)

        # Slider panel (below the canvas) — compact
        main_layout.addWidget(self._build_slider_panel(), stretch=2)

        # Button bar — more visible
        main_layout.addWidget(self._build_button_bar())

    def _build_right_panel(self):
        """Right panel: [angle | 📂] pairs + substrate + film."""
        panel = QWidget()
        panel.setFixedWidth(155)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        self._ang_buttons  = {}
        self._load_buttons = {}

        for ang in [8, 20, 40, 60]:
            row = QHBoxLayout()
            row.setSpacing(4)

            # Angle button — gray/disabled without data
            btn_ang = QPushButton(f"{ang}°")
            btn_ang.setCheckable(True)
            btn_ang.setEnabled(False)
            btn_ang.setFixedHeight(28)
            btn_ang.setFixedWidth(80)
            btn_ang.setStyleSheet(
                "QPushButton { background-color: #D0D0D0; color: #888888; "
                "font-size: 11pt; font-weight: bold; border-radius: 4px; }"
                "QPushButton:enabled:!checked { background-color: #90EE90; color: black; }"
                "QPushButton:checked { background-color: #2E8B57; color: white; }")
            btn_ang.clicked.connect(lambda checked, a=ang: self._select_ang(a))
            self._ang_buttons[ang] = btn_ang
            row.addWidget(btn_ang)

            # Load button — always clickable
            btn_load = QPushButton("📂")
            btn_load.setFixedSize(44, 28)
            btn_load.setToolTip(f"Load {ang}° spectrum")
            btn_load.setStyleSheet(
                "QPushButton { background-color: #5B9BD5; color: white; "
                "font-size: 15pt; border-radius: 4px; }"
                "QPushButton:hover { background-color: #3A7ABF; }"
                "QPushButton:pressed { background-color: #2A5A9F; }")
            btn_load.clicked.connect(lambda checked, a=ang: self._load_ang(a))
            self._load_buttons[ang] = btn_load
            row.addWidget(btn_load)

            layout.addLayout(row)

        # Fit/Maxima view toggle for the SELECTED angle (n from fit vs n from
        # n_eff maxima). Per-angle state in data_angles[ang]['view_mode'].
        self.btn_view_mode = QPushButton("View: FIT")
        self.btn_view_mode.setFixedHeight(24)
        self.btn_view_mode.setToolTip(
            "Toggle n display for the selected angle: model fit vs n from interference maxima")
        self.btn_view_mode.clicked.connect(self._on_toggle_view_mode)
        layout.addWidget(self.btn_view_mode)

        layout.addSpacing(10)

        lbl_sub = QLabel("Substrate")
        lbl_sub.setStyleSheet("color: white; font-size: 10pt; font-weight: bold;")
        layout.addWidget(lbl_sub)
        self.combo_substrate = QComboBox()
        self.combo_substrate.addItems(ph._SUBSTRATI)
        self.combo_substrate.setCurrentIndex(self.state.substrate)
        self.combo_substrate.setFixedHeight(28)
        self.combo_substrate.setStyleSheet(
            "QComboBox { background-color: #D6EAF8; color: black; "
            "font-size: 12pt; font-weight: bold; border-radius: 3px; padding: 2px 4px; }"
            "QComboBox QAbstractItemView { color: black; background: white; font-size:12pt; }")
        self.combo_substrate.currentIndexChanged.connect(self._on_substrate_changed)
        layout.addWidget(self.combo_substrate)

        lbl_film = QLabel("Film")
        lbl_film.setStyleSheet("color: white; font-size: 10pt; font-weight: bold;")
        layout.addWidget(lbl_film)
        self.combo_film = QComboBox()
        self.combo_film.addItems(ph._FILM_MATS)
        self.combo_film.addItem("— Load user material... —")
        last_idx = self.combo_film.count() - 1
        item = self.combo_film.model().item(last_idx)
        item.setEnabled(False)
        try:
            from PyQt6.QtGui import QColor
        except ImportError:
            from PyQt5.QtGui import QColor
        item.setForeground(QColor('#888888'))
        self.combo_film.setCurrentIndex(self.state.film_mat)
        self.combo_film.setFixedHeight(28)
        self.combo_film.setStyleSheet(
            "QComboBox { background-color: #D5F5E3; color: black; "
            "font-size: 12pt; font-weight: bold; border-radius: 3px; padding: 2px 4px; }"
            "QComboBox QAbstractItemView { color: black; background: white; font-size:12pt; }")
        self.combo_film.currentIndexChanged.connect(self._on_film_changed)
        layout.addWidget(self.combo_film)

        layout.addStretch()
        return panel

    def _build_slider_panel(self):
        """Slider panel with 3 columns (left / n / k)."""
        from windows.widgets import MappedSlider, SliderGroup
        import physics as ph

        self._sliders_left  = []   # 15 left-column sliders (12 parameters + Fr contrast / Fr ctr vtx / Fr ctr curv)
        self._sliders_n     = []   # 14 n-node sliders
        self._sliders_k     = []   # 14 k-node sliders

        # ── Outer panel ───────────────────────────────────────────────────────
        panel = QWidget()
        outer = QHBoxLayout(panel)
        outer.setSpacing(6)
        outer.setContentsMargins(2, 2, 2, 2)

        # Handle color: dark purple for the left column (#7B2D8B)
        _STYLE_LEFT  = "QSlider::handle:horizontal{background:#7B2D8B;width:10px;border-radius:4px}" \
                       "QSlider::sub-page:horizontal{background:#7B2D8B;border-radius:2px}"
        _STYLE_N     = "QSlider::handle:horizontal{background:#2E7D32;width:10px;border-radius:4px}" \
                       "QSlider::sub-page:horizontal{background:#2E7D32;border-radius:2px}"
        _STYLE_K     = "QSlider::handle:horizontal{background:#6A0DAD;width:10px;border-radius:4px}" \
                       "QSlider::sub-page:horizontal{background:#6A0DAD;border-radius:2px}"

        # Initial n/k values (single source of truth: N_INIT/K_INIT)
        _N_IN = N_INIT
        _K_IN = K_INIT

        def make_col(title, color):
            """Slider column with a centered, always-visible title."""
            col_widget = QWidget()
            col_widget.setStyleSheet(
                f"QWidget {{ background-color: #FFFFFF; border: 2px solid {color}; "
                f"border-radius: 4px; }}")
            col_layout = QVBoxLayout(col_widget)
            col_layout.setSpacing(0)
            col_layout.setContentsMargins(2, 2, 2, 2)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT == 6 else Qt.AlignCenter)
            lbl.setStyleSheet(
                f"color: {color}; font-size: 12pt; font-weight: bold; "
                f"background: transparent; border: none; padding: 1px 0px;")
            col_layout.addWidget(lbl)
            return col_widget, col_layout

        # ── Left column: 12 parameters + Fr contrast + Fr ctr vtx + Fr ctr curv ──
        col_left, lay_left = make_col("Parameters", "#BB86FC")
        specs_left = [
            ("d",        10,     6000,   self.state.d,     None,  "{:.0f}"),
            ("Scatt.",    0,      300,    self.state.scatt, None,  "{:.1f}"),
            ("Inhom. %",  0,      10,     self.state.inhom, None,  "{:.2f}"),
            ("Offs",     0,      20,     self.state.offs,  None,  "{:.2f}"),
            ("IR Jump",  0.2,    2.0,    self.state.km,    None,  "{:.3f}"),
            ("IR Exp",  -4.0,    4.0,    self.state.ks,    None,  "{:.3f}"),
            ("IR Dmp",   0.0,    3.0,    self.state.kb,    None,  "{:.3f}"),
            ("W 300",    280,    320,    self.state.w300,  None,  "{:.1f}"),
            ("W 335",    315,    355,    self.state.w335,  None,  "{:.1f}"),
            ("W 375",    350,    400,    self.state.w375,  None,  "{:.1f}"),
            ("W 400",    390,    450,    self.state.w400,  None,  "{:.1f}"),
            ("W 1100",   950,   1200,   self.state.w1100, None,  "{:.0f}"),
            ("Fr contrast", 0.0,  3.0,      self.state.pol,   None,  "{:.3f}"),
            # vtx: display the vertex WAVELENGTH λv=1500+1000·value (nm) instead of
            # the raw -2..2 param — readability only, engine still gets the internal
            # value (typing a λ inverts back to the param).
            ("Fr ctr vtx", -2.0,  2.0,     self.state.dn,
                                            (lambda x: 1500.0 + 1000.0 * x),  "{:.0f}"),
            ("Fr ctr curv", -2.0,  2.0,     self.state.pm,    None,  "{:.3f}"),
            # No UV Phase (ps1) slider: phi_s1 is held at 0 under _COLLAPSE_POL.
        ]
        for name, vmin, vmax, init, todisp, fmt in specs_left:
            sl = MappedSlider(name, vmin, vmax, init,
                              to_display=todisp, fmt=fmt, slider_style=_STYLE_LEFT)
            lay_left.addWidget(sl)
            self._sliders_left.append(sl)
        outer.addWidget(col_left)

        def make_bulk_row(btn):
            """Row with the bulk-lock button aligned left (above the lock column)."""
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            row.addWidget(btn)
            row.addStretch()
            return row

        # ── n(λ) column: only 14 nodes ────────────────────────────────────────
        self._btn_bulk_lock_n = QPushButton("🔒")
        self._btn_bulk_lock_n.setFixedSize(24, 22)
        self._btn_bulk_lock_n.setToolTip("Lock/unlock all n sliders")
        self._btn_bulk_lock_n.clicked.connect(self._on_bulk_lock_n)
        col_n, lay_n = make_col("n(λ)", "#4CAF50")
        lay_n.addLayout(make_bulk_row(self._btn_bulk_lock_n))
        for i, wl in enumerate(ph.nodes_wl_base):
            sl = MappedSlider(f"{wl:.0f}nm", 0.0, 7.37, ph.inv_n_map(_N_IN[i]),
                              to_display=ph.n_map, fmt="{:.4f}", slider_style=_STYLE_N)
            lay_n.addWidget(sl)
            self._sliders_n.append(sl)
        # RIGID shift of the n nodes at λ>460 (initial quick optimization).
        # Below the n column (more vertical space). Buttons with auto-repeat:
        # hold down = continuous shift (smooth and predictable). "Rigid n" =
        # uniform shift; "Tilt" = tilt (pivot ~1500nm). They skip the locked
        # nodes, preserve the steep UV nodes (λ≤460), update the real n sliders.
        def _mk_bulk_btn(label, color, shift, slope):
            b = QPushButton(label)
            b.setFixedHeight(24)
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(300)
            b.setAutoRepeatInterval(55)
            b.setStyleSheet(
                f"QPushButton {{ background-color:{color}; color:black; "
                f"font-weight:bold; font-size:10pt; border:1px solid #8a8a8a; "
                f"border-radius:4px; padding:1px 2px; }}")
            b.clicked.connect(
                lambda _=False, sh=shift, sl=slope: self._apply_n_bulk(sh, sl))
            return b
        # ── Bulk n shift + Δn/slope boxes (birefringence) ──
        # Narrow buttons (n −/+, slope −/+) with, in between, the boxes: Δn₀
        # (uniform offset) and slope of the n(θ)−n(8°) difference, fitted as a
        # straight line over the TRANSPARENT window (700–2550 nm, k≈0), plus the
        # Δn min/max range over that window. At θ≠8° the Δn₀/slope boxes are
        # editable (type a value → it gets applied). At 8° (=reference, n_eff=n_o)
        # the boxes are off. The straight line is an approximation valid only in
        # transparency: near absorption Δn is not a straight line
        # and is not quantifiable.
        def _mk_box(editable, tip):
            if editable:
                # explicit color: without it, in dark mode the text is white on the
                # white background forced by the column → invisible.
                w = QLineEdit(); w.setStyleSheet("font-size:8pt; padding:0px; font-weight:bold; color:#000; background:#FFF;")
            else:
                w = QLabel("—"); w.setStyleSheet("font-size:8pt; color:#7030A0; font-weight:bold;")
                w.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT == 6 else Qt.AlignCenter)
            w.setMaximumHeight(22); w.setToolTip(tip)
            return w
        # No fixed width: buttons and boxes FILL the width of the n column via
        # stretch (the column stays as wide as the sliders — wide sliders = easier
        # manual adjustment, especially at large values).
        self._btn_rigid_m = _mk_bulk_btn("n −",     "#D7BDE2", -0.01, 0.0)
        self._btn_rigid_p = _mk_bulk_btn("n +",     "#D7BDE2", +0.01, 0.0)
        self._btn_tilt_m  = _mk_bulk_btn("slope −", "#A9CCE3", 0.0, -0.01)
        self._btn_tilt_p  = _mk_bulk_btn("slope +", "#A9CCE3", 0.0, +0.01)
        self._ed_dn      = _mk_box(True,  "Δn₀ vs 8° (uniform offset) — editable at θ≠8°")
        self._ed_slope   = _mk_box(True,  "slope of Δn(λ) vs 8° (per µm) — editable at θ≠8°")
        self._lbl_dn_max = _mk_box(False, "max Δn over the transparent window (700–2550 nm)")
        self._lbl_dn_min = _mk_box(False, "min Δn over the transparent window (700–2550 nm)")
        grid_dn = QGridLayout(); grid_dn.setSpacing(2)
        for _c in range(4): grid_dn.setColumnStretch(_c, 1)  # 4 equally-wide columns → fill the n column
        hA = QLabel("Δn₀ / slope"); hB = QLabel("Δn max/min")
        for h in (hA, hB):
            h.setStyleSheet("font-size:8pt; color:#333333; font-weight:bold;")
            h.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT == 6 else Qt.AlignCenter)
        grid_dn.addWidget(hA, 0, 1); grid_dn.addWidget(hB, 0, 2)
        grid_dn.addWidget(self._btn_rigid_m, 1, 0); grid_dn.addWidget(self._ed_dn,      1, 1); grid_dn.addWidget(self._lbl_dn_max, 1, 2); grid_dn.addWidget(self._btn_rigid_p, 1, 3)
        grid_dn.addWidget(self._btn_tilt_m,  2, 0); grid_dn.addWidget(self._ed_slope,   2, 1); grid_dn.addWidget(self._lbl_dn_min, 2, 2); grid_dn.addWidget(self._btn_tilt_p,  2, 3)
        lay_n.addLayout(grid_dn)
        self._ed_dn.editingFinished.connect(lambda: self._apply_dn_box('dn'))
        self._ed_slope.editingFinished.connect(lambda: self._apply_dn_box('slope'))
        outer.addWidget(col_n)

        # ── k(λ) column: only 14 nodes ────────────────────────────────────────
        self._btn_bulk_lock_k = QPushButton("🔒")
        self._btn_bulk_lock_k.setFixedSize(24, 22)
        self._btn_bulk_lock_k.setToolTip("Lock/unlock all k sliders")
        self._btn_bulk_lock_k.clicked.connect(self._on_bulk_lock_k)
        col_k, lay_k = make_col("k(λ)", "#CE93D8")
        lay_k.addLayout(make_bulk_row(self._btn_bulk_lock_k))
        for i, wl in enumerate(ph.nodes_wl_base):
            sl = MappedSlider(f"{wl:.0f}nm", -4.0, 1.0, ph.inv_k_map(_K_IN[i]),
                              to_display=ph.k_map, fmt="{:.4f}", slider_style=_STYLE_K)
            lay_k.addWidget(sl)
            self._sliders_k.append(sl)
        outer.addWidget(col_k)

        # ── Connect signals → AppState and plot refresh ───────────────────────
        self._connect_sliders()
        self._update_bulk_lock_labels()  # apply the initial style to the bulk-locks
        return panel

    def _connect_sliders(self):
        """Connect all slider signals → AppState → plot refresh."""
        s = self.state

        # Left column: 12 parameters + Fr contrast(12) + Fr ctr vtx(13) + Fr ctr curv(14)
        left_attrs = ['d','scatt','inhom','offs','km','ks','kb',
                      'w300','w335','w375','w400','w1100','pol','dn','pm']
        for i, (sl, attr) in enumerate(zip(self._sliders_left, left_attrs)):
            def make_cb(a, idx):
                def cb(v):
                    setattr(s, a, v)
                    if not s.suppress_update:
                        self._refresh_simulation()
                return cb
            sl.value_changed.connect(make_cb(attr, i))
            def make_lock(idx):
                def cb(locked):
                    if idx < 13:
                        s.lock_left[idx] = locked
                    else:  # 13=dn (Fr ctr vtx), 14=pm (Fr ctr curv)
                        s.lock_extra[idx - 13] = locked
                return cb
            sl.lock_changed.connect(make_lock(i))

        # n sliders (14 nodes)
        for i, sl in enumerate(self._sliders_n):
            def make_n_cb(idx):
                def cb(v):
                    s.n_slider_vals[idx] = v
                    if not s.suppress_override_clear:
                        # Clear the override: _refresh_simulation will use n_slider_vals
                        s.ext_override = None
                    if not s.suppress_update:
                        self._refresh_simulation()
                return cb
            sl.value_changed.connect(make_n_cb(i))
            def make_n_lock(idx):
                def cb(locked): s.lock_n[idx] = locked
                return cb
            sl.lock_changed.connect(make_n_lock(i))
            sl.lock_changed.connect(lambda _: self._update_bulk_lock_labels())

        # k sliders (14 nodes)
        for i, sl in enumerate(self._sliders_k):
            def make_k_cb(idx):
                def cb(v):
                    s.k_slider_vals[idx] = v
                    if not s.suppress_override_clear:
                        s.ext_override = None
                    if not s.suppress_update:
                        self._refresh_simulation()
                return cb
            sl.value_changed.connect(make_k_cb(i))
            def make_k_lock(idx):
                def cb(locked): s.lock_k[idx] = locked
                return cb
            sl.lock_changed.connect(make_k_lock(i))
            sl.lock_changed.connect(lambda _: self._update_bulk_lock_labels())

    # Bulk-lock button styles — constants to avoid recreation on every update
    _BULK_STYLE_LOCKED = (
        "QPushButton { background-color: #FF9800; color: white; "
        "border: 1px solid #C66900; border-radius: 4px; padding: 0px; "
        "font-size: 11pt; font-weight: bold; }"
        "QPushButton:hover { background-color: #F57C00; }")
    _BULK_STYLE_UNLOCKED = (
        "QPushButton { background-color: #B0B0B0; color: black; "
        "border: 1px solid #555; border-radius: 4px; padding: 0px; "
        "font-size: 11pt; font-weight: bold; }"
        "QPushButton:hover { background-color: #909090; }")

    def _bulk_lock_toggle(self, sliders, lock_list):
        """Bulk lock/unlock of all the passed sliders.
        If all already locked → unlock all; otherwise lock all.
        NB: set_locked() suppresses lock_changed, so I update
        lock_list[i] directly."""
        all_locked = all(lock_list[i] for i in range(len(sliders)))
        new_state = not all_locked
        for i, sl in enumerate(sliders):
            lock_list[i] = new_state
            sl.set_locked(new_state)
        self._update_bulk_lock_labels()

    def _on_bulk_lock_n(self):
        self._bulk_lock_toggle(self._sliders_n, self.state.lock_n)

    def _on_bulk_lock_k(self):
        self._bulk_lock_toggle(self._sliders_k, self.state.lock_k)

    def _apply_n_bulk(self, shift=0.0, slope=0.0, lam_min=460.0, lam_ref=1500.0):
        """Bulk-shift the n nodes at λ>lam_min (default 460 → includes node 470,
        excludes the steep UV nodes): uniform shift + tilt (pivot lam_ref).
        For the initial quick optimization. Skips the locked nodes, clears
        ext_override (n changed) and redraws once."""
        s = self.state
        c_nodes = ph.build_c_nodes(s)
        s.suppress_update = True
        # Sync slider ← the angle's ext_override (if present): the shift starts from
        # the REAL displayed curve, not from another angle's frozen values. Exact on
        # the 14 base nodes (they are inside ext_nodes → np.interp recovers them).
        ov = s.ext_override
        if ov and len(ov) >= 2:
            n_disp = np.interp(np.asarray(c_nodes, float),
                               np.asarray(ov[0], float), np.asarray(ov[1], float))
            for i in range(min(len(s.n_slider_vals), len(n_disp), len(self._sliders_n))):
                iv = float(np.clip(ph.inv_n_map(n_disp[i]), 0.0, 7.37))
                s.n_slider_vals[i] = iv
                self._sliders_n[i].set_value(iv)
        changed = False
        for i, lam in enumerate(c_nodes):
            if i >= len(s.n_slider_vals) or lam <= lam_min:
                continue
            if i < len(s.lock_n) and s.lock_n[i]:
                continue
            cur = ph.n_map(s.n_slider_vals[i])
            new = cur + shift + slope * (lam - lam_ref) / 1000.0
            internal = float(np.clip(ph.inv_n_map(new), 0.0, 7.37))
            s.n_slider_vals[i] = internal
            if i < len(self._sliders_n):
                self._sliders_n[i].set_value(internal)
            changed = True
        s.suppress_update = False
        if changed:
            s.ext_override = None
            self._refresh_simulation()
            self._refresh_dn_boxes()

    def _dn_slope_fit(self):
        """Straight line of Δn(λ)=n(θ)−n(8°) over the TRANSPARENT window
        (700–2550 nm, k≈0): returns (Δn₀@1500, slope/µm, Δn_min, Δn_max) or None.
        Current n from the SLIDERS (holds even after a shift that clears
        ext_override). Transparency only: near absorption Δn is not a straight line
        and is not quantifiable."""
        s = self.state
        ov8 = s.data_angles.get(8, {}).get('ext_override')
        if not ov8 or len(ov8) < 2:
            return None
        try:
            c = np.asarray(ph.build_c_nodes(s), float)
            # The angle's CURRENT n = the REAL displayed curve (ext_override), NOT the
            # global sliders (which are not per-angle → they'd lie). After a manual
            # shift ext_override=None → it falls back to the sliders (now updated by the shift).
            ov_cur = s.ext_override
            if ov_cur and len(ov_cur) >= 2:
                n_cur = np.interp(c, np.asarray(ov_cur[0], float), np.asarray(ov_cur[1], float))
            else:
                n_cur = np.array([ph.n_map(v) for v in s.n_slider_vals], float)
            if len(n_cur) != len(c):
                return None
            e8, n8 = np.asarray(ov8[0], float), np.asarray(ov8[1], float)
            res = n_cur - np.interp(c, e8, n8)
            m = (c >= 700.0) & (c <= 2550.0)
            if int(m.sum()) < 2:
                return None
            lam, r = c[m], res[m]
            A = np.vstack([np.ones_like(lam), (lam - 1500.0) / 1000.0]).T
            a, b = np.linalg.lstsq(A, r, rcond=None)[0]
            dn = a + b * (lam - 1500.0) / 1000.0
            return float(a), float(b), float(dn.min()), float(dn.max())
        except Exception:
            return None

    def _refresh_dn_boxes(self):
        """Populate Δn₀/slope/min/max. At 8° (=reference, n_eff=n_o) they are off."""
        if not hasattr(self, '_ed_dn'):
            return
        if self.state.current_ang == 8:
            for w in (self._ed_dn, self._ed_slope):
                w.blockSignals(True); w.setText(""); w.setPlaceholderText("8°=ref")
                w.setEnabled(False); w.setModified(False); w.blockSignals(False)
            self._lbl_dn_max.setText("—"); self._lbl_dn_min.setText("—")
            return
        for w in (self._ed_dn, self._ed_slope):
            w.setEnabled(True)
        fit = self._dn_slope_fit()
        if fit is None:
            for w in (self._ed_dn, self._ed_slope):
                w.blockSignals(True); w.setText(""); w.setPlaceholderText("fit 8°?")
                w.setModified(False); w.blockSignals(False)
            self._lbl_dn_max.setText("—"); self._lbl_dn_min.setText("—")
            return
        a, b, dmin, dmax = fit
        for w, val in ((self._ed_dn, a), (self._ed_slope, b)):
            w.blockSignals(True); w.setText(f"{val:+.4f}")
            w.setModified(False); w.blockSignals(False)
        self._lbl_dn_max.setText(f"{dmax:+.4f}")
        self._lbl_dn_min.setText(f"{dmin:+.4f}")

    def _apply_dn_box(self, which):
        """Apply the value typed in Δn₀ or slope: offset from the current value
        → _apply_n_bulk. isModified guard (avoids spurious applies on focus loss)."""
        if self.state.current_ang == 8:
            return
        w = self._ed_dn if which == 'dn' else self._ed_slope
        if not w.isModified():
            return
        try:
            target = float(w.text().replace(',', '.'))
        except ValueError:
            self._refresh_dn_boxes(); return
        fit = self._dn_slope_fit()
        cur = (fit[0] if which == 'dn' else fit[1]) if fit else 0.0
        if which == 'dn':
            self._apply_n_bulk(shift=target - cur, slope=0.0)
        else:
            self._apply_n_bulk(shift=0.0, slope=target - cur)
        w.setModified(False)

    def _update_bulk_lock_labels(self):
        """Update the bulk-lock buttons' icon and color based on the current state.
        - 🔓 orange when all already locked → clicking unlocks them all
        - 🔒 dark gray otherwise            → clicking locks them all"""
        if not hasattr(self, '_btn_bulk_lock_n'):
            return
        s = self.state
        for btn_w, lock_list in [(self._btn_bulk_lock_n, s.lock_n),
                                  (self._btn_bulk_lock_k, s.lock_k)]:
            all_locked = all(lock_list[:14])
            btn_w.setText("🔓" if all_locked else "🔒")
            btn_w.setStyleSheet(self._BULK_STYLE_LOCKED if all_locked
                                else self._BULK_STYLE_UNLOCKED)

    def _sync_material_combos(self):
        """Show in the two menus the materials the state actually holds, without
        re-emitting the signals (the state is already the one we want)."""
        s = self.state
        for combo, idx in ((getattr(self, 'combo_substrate', None), s.substrate),
                           (getattr(self, 'combo_film', None), s.film_mat)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _update_sliders_from_state(self):
        """Update all sliders from the current AppState values."""
        s = self.state

        # Backward-compat with old 13-node sessions — extends to 14
        while len(s.n_slider_vals) < 14:
            s.n_slider_vals.append(ph.inv_n_map(2.0))
        while len(s.k_slider_vals) < 14:
            s.k_slider_vals.append(ph.inv_k_map(0.0))
        s.suppress_update = True
        s.suppress_override_clear = True

        # Left column: 12 parameters + Fr contrast + Fr ctr vtx + Fr ctr curv
        left_attrs = ['d','scatt','inhom','offs','km','ks','kb',
                      'w300','w335','w375','w400','w1100','pol','dn','pm']
        for sl, attr in zip(self._sliders_left, left_attrs):
            sl.set_value(getattr(s, attr))

        # n and k: 14 nodes each
        for i, sl in enumerate(self._sliders_n):
            sl.set_value(s.n_slider_vals[i])
        for i, sl in enumerate(self._sliders_k):
            sl.set_value(s.k_slider_vals[i])

        # Lock state — use min() for robustness with old sessions
        for i, sl in enumerate(self._sliders_left[:13]):
            if i < len(s.lock_left):
                sl.set_locked(s.lock_left[i])
        if len(self._sliders_left) > 13 and len(s.lock_extra) > 0:
            self._sliders_left[13].set_locked(s.lock_extra[0])  # dn
        if len(self._sliders_left) > 14 and len(s.lock_extra) > 1:
            self._sliders_left[14].set_locked(s.lock_extra[1])  # pm
        for i, sl in enumerate(self._sliders_n):
            if i < len(s.lock_n):
                sl.set_locked(s.lock_n[i])
        for i, sl in enumerate(self._sliders_k):
            if i < len(s.lock_k):
                sl.set_locked(s.lock_k[i])

        s.suppress_update = False
        s.suppress_override_clear = False
        self._update_bulk_lock_labels()

    def _btn_css(self, bg, fixed=True):
        """SINGLE stylesheet for ALL the bar buttons (consistency + fix).

        - SCOPED to `QPushButton {...}`: so it does NOT leak onto the QToolTip.
        - Text always BLACK, bold, 10pt (including the toggles).
        - Border + equal min=max width everywhere → uniform width even on macOS.
        """
        box = ("padding:2px 4px; min-width:97px; max-width:97px;" if fixed
               else "padding:2px 12px;")
        return (f"QPushButton {{ background-color:{bg}; color:black; "
                f"font-weight:bold; font-size:10pt; "
                f"border:1px solid #8a8a8a; border-radius:4px; {box} }}")

    def _style_button(self, b, bg, fixed=True):
        """Apply style + fixed width. To be used at creation AND on EVERY re-style
        (toggle): `setFixedWidth` — honored on macOS with the CSS border active —
        locks the geometry, so the width stays UNIFORM even when a toggle changes
        state/text. (Without it, a re-style would re-size the button to its text,
        giving activated toggles a different width.)"""
        b.setStyleSheet(self._btn_css(bg, fixed))
        if fixed:
            b.setFixedWidth(107)   # = 97 (CSS content) + 8 padding + 2 border

    def _build_button_bar(self):
        """3 rows of logically grouped buttons with visual state."""
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        def btn(label, slot, color="#E0E0E0", fixed=True, fw=None):
            b = QPushButton(label)
            self._style_button(b, color, fixed)
            b.setFixedHeight(28)
            b.clicked.connect(slot)
            return b

        # ── Row 1: FILE (load | save | export) ───────────────────────────────
        r1 = QHBoxLayout(); r1.setSpacing(4)
        _b_load = btn("LOAD", self._on_load, "skyblue")
        _b_load.setToolTip(
            "Load ALL angle spectra of one sample from a FOLDER: pick the folder,\n"
            "then tick the files to load in the list — 8/20/40/60 are recognized from\n"
            "the file names and a search box filters the list (a manifest is used if\n"
            "present). Loads R_exp only.\n"
            "Single angle → the per-angle 📂 buttons; parameters → LOAD PAR / LOAD SESS.")
        r1.addWidget(_b_load)
        r1.addWidget(btn("LOAD PAR",  self._on_load_params,  "gold"))
        r1.addWidget(btn("LOAD SESS", self._on_load_session, "#B0E0E6"))
        r1.addWidget(btn("SAVE",       self._on_save,         "plum"))
        r1.addWidget(btn("SAVE ALL",   self._on_save_all,     "#CE93D8"))
        r1.addWidget(btn("SAVE SESS",  self._on_save_session, "#B0E0E6"))
        _b_expfig = btn("EXPORT FIG", self._on_export_figure, "#FFE082")
        _b_expfig.setToolTip(
            "Export the current main figure (reflectance + n,k) as a vector PDF\n"
            "(default), a 600-DPI PNG or SVG — for the paper / figures.")
        r1.addWidget(_b_expfig)
        _b_camp = btn("CAMPAIGN", self._on_export_campaign, "#4DD0E1")
        _b_camp.setToolTip(
            "Append the current accepted fit to campaign_log.csv (one row).\n"
            "Auto-logs sample, angle, d, fringe-position residual, RMS residual\n"
            "and Δn (from fit and from maxima); asks for the judgement fields\n"
            "(reference label, m resolved, KK, locked params).")
        r1.addWidget(_b_camp)
        r1.addStretch()
        # ABOUT on the right (at the end of the FILE row, separated by a stretch)
        r1.addWidget(btn("ABOUT", self._on_about, "#E1BEE7", fixed=False))
        layout.addLayout(r1)

        # ── Row 2: FIT + OPTIONS ──────────────────────────────────────────────
        r2 = QHBoxLayout(); r2.setSpacing(4)
        self._btn_adatta    = btn("FIT",    self._on_fit,       "#90EE90")
        self._btn_fit_multi = btn("FIT MULTI", self._on_fit_multi, "#AED6F1")
        self._btn_fit_auto  = btn("FIT AUTO",  self._on_fit_auto,  "#A8D8A8")
        r2.addWidget(self._btn_adatta)
        r2.addWidget(self._btn_fit_multi)
        r2.addWidget(self._btn_fit_auto)
        _b_savefit = btn("SAVE AS FIT", self._on_save_as_fit, "#C5E1A5")
        _b_savefit.setToolTip(
            "Accept the current on-screen curve as THIS angle's fit, WITHOUT\n"
            "optimizing: use when the auto-computed per-angle curve is already\n"
            "good by eye. It stores the same per-angle snapshot a real fit would,\n"
            "so SAVE ALL saves this angle's parameters (no fallback to the current\n"
            "angle) and the 'missing fit' warning stops flagging it.")
        r2.addWidget(_b_savefit)
        r2.addWidget(btn("STOP",      self._on_stop,      "orange"))

        self._btn_mono_n  = btn("MONO n ON",  self._on_mono_n,  "#90ee90")
        self._btn_mono_k  = btn("MONO k ON",  self._on_mono_k,  "#90ee90")
        r2.addWidget(self._btn_mono_n)
        r2.addWidget(self._btn_mono_k)

        # K-BOUND: sibling of MONO k — a shape constraint on k. ON (default)
        # clamps k to ~0 above 550 nm (PSi transparent → k is a residue there);
        # OFF frees k to the graded physical cap. See AppState.k_bound.
        self._btn_k_bound = btn("K-BOUND ON", self._on_k_bound, "#90ee90")
        self._set_toggle_btn(self._btn_k_bound,
                             getattr(self.state, 'k_bound', True),
                             "K-BOUND ON", "K-BOUND OFF", "#4CAF50", "#BBBBBB")
        self._btn_k_bound.setToolTip(
            "k-bound in the transparency region (λ > 550 nm):\n"
            "ON (default) — k clamped to ~0 there, where PSi is transparent and a\n"
            "  finite k is a calculation residue, not physical information;\n"
            "OFF — k free up to the graded physical cap. PSi (free fit) only:\n"
            "  known materials keep their tabulated k regardless.")
        r2.addWidget(self._btn_k_bound)

        # "thin sample" toggle
        self._btn_thin_film = btn("THIN FILM",
                                  self._on_thin_film_toggle,
                                  "#BBBBBB")
        self._set_toggle_btn(self._btn_thin_film,
                             getattr(self.state, 'thin_film', False),
                             "THIN ON", "THIN OFF",
                             "#FF9800", "#BBBBBB")
        self._btn_thin_film.setToolTip(
            "Thin-sample mode (d ≲ 700 nm):\n"
            "- lowers the accepted λ threshold from 1000 to 600 nm\n"
            "- accepts even a single NIR peak (m=1 assumed)\n"
            "- switches to the single-angle formula (more stable on thin films)")
        r2.addWidget(self._btn_thin_film)

        # Low-contrast fringes aid: opt-in for the maxima-centering
        # penalty (_peak_penalty). Default OFF — see AppState.
        self._btn_low_contrast = btn("FRINGE AID",
                                     self._on_low_contrast_toggle,
                                     "#BBBBBB")
        self._set_toggle_btn(self._btn_low_contrast,
                             getattr(self.state, 'low_contrast_aid', False),
                             "AID ON", "AID OFF",
                             "#FF9800", "#BBBBBB")
        self._btn_low_contrast.setToolTip(
            "Low-contrast fringes aid (default OFF):\n"
            "- enable ONLY for samples with weak/few fringes where the fit\n"
            "  does not center the maxima on its own\n"
            "- adds a maxima-centering penalty for λ ≥ 500 nm (skips the deep\n"
            "  blue, where it can otherwise make the fit diverge)\n"
            "- keep OFF for normal, well-contrasted samples")
        r2.addWidget(self._btn_low_contrast)

        r2.addStretch()

        # Peak-source selector for the d calculation, pushed to the right.
        # (peak_window + fit_auto consistent.)
        lbl_src = QLabel("d src:")
        lbl_src.setStyleSheet("font-weight:bold; padding:2px;")
        r2.addWidget(lbl_src)
        self.combo_d_source = QComboBox()
        self.combo_d_source.addItems(["exp", "fit", "hybrid"])
        # init from the state (default 'exp' if not set)
        _src_init = getattr(self.state, 'd_source', 'exp')
        if _src_init not in ("exp", "fit", "hybrid"):
            _src_init = "exp"
        self.combo_d_source.setCurrentText(_src_init)
        self.combo_d_source.setStyleSheet(
            "QComboBox { background-color:#E8DAEF; color:black; "
            "font-size:10pt; font-weight:bold; padding:2px; }")
        self.combo_d_source.currentTextChanged.connect(self._on_d_source_changed)
        self.combo_d_source.setToolTip(
            "Peak source for the d calculation:\n"
            "exp = peaks from the experimental data (default, robust)\n"
            "fit = peaks from the model (useful at convergence)\n"
            "hybrid = fit peaks as guide, exp max within a ±30 nm window")
        self.combo_d_source.setFixedSize(90, 28)
        r2.addWidget(self.combo_d_source)

        layout.addLayout(r2)

        # ── Row 3: ANALYSIS ───────────────────────────────────────────────────
        r3 = QHBoxLayout(); r3.setSpacing(4)
        r3.addWidget(btn("PEAK ANAL.", self._on_peaks,  "gold"))
        self._btn_kk      = btn("KK OFF",      self._on_kk,    "#DDDDDD")
        self._style_button(self._btn_kk, "#DDDDDD")
        self._btn_kk.setToolTip(
            "Open the Kramers-Kronig consistency window: compares n(λ) from the\n"
            "fit with n_KK(λ) from k(λ). Consistent if RMS(n−n_KK) ≲ σ_KK.")
        r3.addWidget(self._btn_kk)
        _b_kkuv = btn("KK→UV",      self._on_kk_uv,    "#DDDDDD")
        _b_kkuv.setToolTip(
            "Apply the KK-consistent n(λ) to the free/UV nodes, nudging the fit\n"
            "toward KK self-consistency.")
        r3.addWidget(_b_kkuv)
        _b_jump = btn("MEAS JUMP",   self._on_measure_jump, "#FFCC80")
        _b_jump.setToolTip(
            "Measure the detector-jump amplitude (~890 nm) from the current\n"
            "angle's data, set it on the IR Jump slider and lock it. Data-anchored:\n"
            "prevents the fit from confusing the jump with n/k/scatt.")
        r3.addWidget(_b_jump)
        # COPY 8° + PRED 8° — operations on the 8° parameters.
        _b_copy = btn("COPY 8°",  self._on_copia8,    "#FFD700")
        _b_copy.setToolTip(
            "Copy the 8° fit parameters (n, k, d, ...) to the current angle.")
        r3.addWidget(_b_copy)
        self._btn_pred8   = btn("PRED 8° OFF", self._on_pred8, "#BBBBBB")
        self._style_button(self._btn_pred8, "#BBBBBB")
        self._btn_pred8.setToolTip(
            "Preview: overlay the reflectance the 8° parameters WOULD give at\n"
            "the current angle, WITHOUT copying them (COPY 8° applies them).")
        r3.addWidget(self._btn_pred8)
        _b_dscan = btn("d-SCAN", self._on_dscan_analysis, "#A5D6A7")
        _b_dscan.setToolTip(
            "Compare saved sessions at different d: plot χ²(d) and n(d) to pick\n"
            "the optimal thickness. Save one session per d first.")
        r3.addWidget(_b_dscan)
        # Spread label
        self._lbl_spread = QLabel("Spread: --")
        self._lbl_spread.setStyleSheet("font-size:9pt; font-weight:bold; color:dimgray; padding:2px 6px;")
        r3.addWidget(self._lbl_spread)
        r3.addStretch()
        r3.addWidget(btn("RESET", self._on_reset, "#FF6B6B"))
        r3.addWidget(btn("EXIT",  self.close,     "#FF3B30"))
        layout.addLayout(r3)

        return bar

    def _set_toggle_btn(self, btn: QPushButton, is_on: bool,
                        label_on: str, label_off: str,
                        color_on: str, color_off: str):
        """Update the text and color of a toggle button."""
        btn.setText(label_on if is_on else label_off)
        self._style_button(btn, color_on if is_on else color_off)

    def _update_all_toggles(self):
        """Bring all toggle buttons back to AppState's current state."""
        s = self.state
        if hasattr(self, '_btn_mono_n'):
            self._set_toggle_btn(self._btn_mono_n, s.mono_n,
                "MONO n ON","MONO n OFF","#4CAF50","#BBBBBB")
            self._set_toggle_btn(self._btn_mono_k, s.mono_k,
                "MONO k ON","MONO k OFF","#4CAF50","#BBBBBB")
            self._set_toggle_btn(self._btn_k_bound, getattr(s, 'k_bound', True),
                "K-BOUND ON","K-BOUND OFF","#4CAF50","#BBBBBB")
            self._set_toggle_btn(self._btn_kk, s.kk_on,
                "KK ON","KK OFF","#FF9800","#BBBBBB")
            self._set_toggle_btn(self._btn_pred8, s.use_pred_8,
                "PRED 8° ON","PRED 8° OFF","#FF9800","#BBBBBB")
        if hasattr(self, '_btn_thin_film'):
            self._set_toggle_btn(self._btn_thin_film,
                getattr(s, 'thin_film', False),
                "THIN ON", "THIN OFF", "#FF9800", "#BBBBBB")
        if hasattr(self, '_btn_low_contrast'):
            self._set_toggle_btn(self._btn_low_contrast,
                getattr(s, 'low_contrast_aid', False),
                "AID ON", "AID OFF", "#FF9800", "#BBBBBB")
        if hasattr(self, 'combo_d_source'):
            _ds = getattr(s, 'd_source', 'exp')
            if _ds in ("exp", "fit", "hybrid"):
                self.combo_d_source.setCurrentText(_ds)

    def _build_status_bar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Angle and material callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _select_ang(self, ang: int):
        """Select the current angle and refresh the UI."""
        self.state.current_ang = ang
        self._update_ang_buttons()
        self._refresh_view_mode_btn()
        self._update_plot_title()   # keeps the sample name always visible
        dd = self.state.data_angles[ang]
        if dd["exp"] is None:
            return
        wl_exp, r_exp = dd["exp"]
        self.l_exp.set_data(wl_exp, r_exp)
        self.ax1.set_xlim(200, 2600)
        self.ax_res.set_xlim(200, 2600)
        self.ax2.set_xlim(200, 2600)
        if dd.get("ext_override") is not None:
            self.state.ext_override = dd["ext_override"]
            if dd.get("sliders"):
                sl = dd["sliders"]
                self.state.suppress_update = True
                self.state.suppress_override_clear = True
                # Restore whatever _save_angle_state saved by iterating the dict
                # itself. Keys that are not state attributes ('da', the angular
                # spread) are skipped by hasattr.
                for attr, val in sl.items():
                    if hasattr(self.state, attr):
                        setattr(self.state, attr, val)
                self.state.suppress_update = False
                self.state.suppress_override_clear = False
                self._update_sliders_from_state()
        self._refresh_simulation()
        self._refresh_dn_boxes()          # Δn/slope boxes (vs 8°)

    def _load_ang(self, ang: int):
        """Open the file dialog and load the spectrum for the given angle."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {ang}° spectrum", "",
            "Spectra (*.txt *.csv *.asc *.dat);;All (*)")
        if not path:
            return
        result = self.engine.load_spectral_file(path, ang)
        if result is None:
            self.status.showMessage(f"Error loading {ang}°")
            return
        wl_exp, r_exp = result
        fname = os.path.basename(path)
        self.state.ang_filenames[ang] = fname
        self._ang_buttons[ang].setEnabled(True)
        self._ang_buttons[ang].setToolTip(fname)
        self.ax1.set_xlim(200, 2600)
        self.ax_res.set_xlim(200, 2600)
        self.ax2.set_xlim(200, 2600)
        self._update_plot_title()
        self.status.showMessage(f"Loaded {ang}° — {len(wl_exp)} pts — {fname}")
        self._select_ang(ang)
        self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        """Update the crosshair and coordinate label on mouse movement."""
        # ax2_k is the twin axis of ax2 — both must activate the crosshair
        active_axes = (self.ax1, self.ax_res, self.ax2, self.ax2_k)
        if event.inaxes not in active_axes:
            for ln in self._crosshair_lines:
                ln.set_visible(False)
            self.canvas.draw_idle()
            return

        x = event.xdata
        if x is None:
            return

        # Show the vertical line on all three plots
        for ln in self._crosshair_lines:
            ln.set_xdata([x, x])
            ln.set_visible(True)

        # Read the curve values at that λ point
        parts = [f"λ = {x:.1f} nm"]

        # ax1: R_exp and R_sim
        for line, name in [(self.l_exp, 'R_exp'), (self.l_sim, 'R_sim')]:
            xd, yd = line.get_xdata(), line.get_ydata()
            if len(xd) > 1:
                y = float(np.interp(x, xd, yd))
                parts.append(f"{name}={y:.2f}%")

        # ax_res: residual
        xd, yd = self.l_res.get_xdata(), self.l_res.get_ydata()
        if len(xd) > 1:
            y = float(np.interp(x, xd, yd))
            parts.append(f"Res={y:+.3f}%")

        # ax2: n (and n_KK if active)
        for line, name in [(self.l_n, 'n'), (self.l_n_kk, 'n_KK')]:
            xd, yd = line.get_xdata(), line.get_ydata()
            if len(xd) > 1 and float(np.max(np.abs(yd))) > 0:
                y = float(np.interp(x, xd, yd))
                parts.append(f"{name}={y:.4f}")

        # ax2_k: k (and k_KK if active)
        for line, name in [(self.l_k, 'k'), (self.l_k_kk, 'k_KK')]:
            xd, yd = line.get_xdata(), line.get_ydata()
            if len(xd) > 1 and float(np.max(np.abs(yd))) > 0:
                y = float(np.interp(x, xd, yd))
                parts.append(f"{name}={y:.5f}")

        self._cursor_label.setText("   |   ".join(parts))
        self.canvas.draw_idle()

    def _update_plot_title(self):
        """Update the ax1 title with the loaded file names per angle."""
        fnames = self.state.ang_filenames
        parts = [f"{a}°: {os.path.splitext(f)[0]}"
                 for a, f in sorted(fnames.items()) if f is not None]
        title = "  |  ".join(parts) if parts else "OPHIRA"
        self.ax1.set_title(title, fontsize=8, loc='left', pad=3)

    def _update_ang_buttons(self):
        """Update the enabled state/color of all angle buttons."""
        ang = self.state.current_ang
        for a, b in self._ang_buttons.items():
            has = self.state.has_data(a)
            b.setEnabled(has)
            if a == ang and has:
                b.setStyleSheet(
                    "QPushButton { background-color: #2E8B57; color: white; "
                    "font-size: 11pt; font-weight: bold; border-radius: 4px; }"
                    "QPushButton:checked { background-color: #2E8B57; color: white; }")
            elif has:
                b.setStyleSheet(
                    "QPushButton { background-color: #90EE90; color: black; "
                    "font-size: 11pt; font-weight: bold; border-radius: 4px; }"
                    "QPushButton:checked { background-color: #2E8B57; color: white; }")
            else:
                b.setStyleSheet(
                    "QPushButton { background-color: #D0D0D0; color: #888888; "
                    "font-size: 11pt; font-weight: bold; border-radius: 4px; }")
            b.setChecked(a == ang and has)

    def _on_substrate_changed(self, idx: int):
        self.state.substrate = idx
        import sys
        sys.modules['physics']._substrate[0] = idx
        self._refresh_simulation()

    def _on_film_changed(self, idx: int):
        # Ignore the disabled "Load user material..." entry
        if idx >= len(ph._FILM_MATS):
            self.combo_film.setCurrentIndex(self.state.film_mat)
            return
        self.state.film_mat = idx
        import sys
        sys.modules['physics']._film_mat[0] = idx
        s = self.state
        is_psi = (idx == 0)
        if not is_psi:
            # Load the tabulated values at the current 14 nodes and lock the sliders
            film_name = ph._FILM_MATS[idx]
            c_nodes = ph.build_c_nodes(s)
            nk_tab = ph._nk_film_known(c_nodes, film_name)
            if nk_tab is not None:
                n_tab = nk_tab.real
                k_tab = np.maximum(nk_tab.imag, 0.0)
                s.suppress_update = True
                s.suppress_override_clear = True
                for i in range(14):
                    s.n_slider_vals[i] = ph.inv_n_map(float(np.clip(n_tab[i],1.0,12.0)))
                    s.k_slider_vals[i] = ph.inv_k_map(float(np.clip(k_tab[i],1e-9,15.0)))
                    self._sliders_n[i].set_value(s.n_slider_vals[i])
                    self._sliders_k[i].set_value(s.k_slider_vals[i])
                    s.lock_n[i] = True; s.lock_k[i] = True
                    self._sliders_n[i].set_locked(True)
                    self._sliders_k[i].set_locked(True)
                s.suppress_update = False
                s.suppress_override_clear = False
                s.ext_override = None
        else:
            # Back to PSi: unlock the n/k sliders
            s.suppress_update = True
            for i in range(14):
                s.lock_n[i] = False; s.lock_k[i] = False
                self._sliders_n[i].set_locked(False)
                self._sliders_k[i].set_locked(False)
            s.suppress_update = False
            s.ext_override = None
        self._update_bulk_lock_labels()
        self._refresh_simulation()

    def _set_nk_sliders_enabled(self, enabled: bool):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Plot refresh
    # ─────────────────────────────────────────────────────────────────────────

    def _update_plots_from_state(self, ang: int):
        """Update the plots with the saved data for angle `ang`."""
        dd = self.state.data_angles[ang]
        if dd["exp"] is None:
            return
        wl_exp, r_exp = dd["exp"]
        self.l_exp.set_data(wl_exp, r_exp)
        self.ax1.set_xlim(wl_exp.min(), wl_exp.max())
        if dd["fit"] is not None:
            R = dd["fit"]
            self.l_sim.set_data(wl_exp, R)
            res = R - r_exp
            self.l_res.set_data(wl_exp, res)
            rms = float(np.sqrt(np.mean(res**2)))
            self.txt_chi.set_text(f'RMS resid: {rms:.3f} %R')
        ov = dd.get("ext_override")
        if ov is not None:
            ext_nd, n_arr, k_arr, _ = ov
            nf = np.interp(wl_exp, ext_nd, n_arr)
            kf = np.interp(wl_exp, ext_nd, k_arr)
            self.l_n.set_data(wl_exp, nf)
            self.l_k.set_data(wl_exp, kf)
            self.extra_dots_n.set_data(ext_nd, n_arr)
            self.extra_dots_k.set_data(ext_nd, k_arr)
        self.ax2.relim(); self.ax2.autoscale_view()
        self.ax2_k.relim(); self.ax2_k.autoscale_view()
        self.canvas.draw_idle()

    def _refresh_simulation(self):
        """Recompute R with the current parameters and refresh plots + sliders.
        Works both with and without loaded data and without ext_override."""
        from scipy.interpolate import PchipInterpolator
        try:
            ang = self.state.current_ang
            dd  = self.state.data_angles[ang]
            # Sync the physics global variables with the current state
            import sys as _sys
            _ph = _sys.modules['physics']
            _ph._substrate[0] = self.state.substrate
            _ph._film_mat[0]  = self.state.film_mat
            # Use the experimental grid if available, otherwise the default grid
            if dd["exp"] is not None:
                wl_exp, r_exp = dd["exp"]
            else:
                wl_exp = self._wl_default
                r_exp  = np.zeros(len(wl_exp))
            s = self.state

            # ── Current nodes ─────────────────────────────────────────────────
            c_nodes = ph.build_c_nodes(s)

            # Update the n and k slider labels
            for i, sl in enumerate(self._sliders_n[:14]):
                sl.set_label(f"{int(c_nodes[i])}nm")
            for i, sl in enumerate(self._sliders_k[:14]):
                sl.set_label(f"{int(c_nodes[i])}nm")

            # Update the node vlines
            if hasattr(self, 'vlines'):
                for line, node in zip(self.vlines, c_nodes):
                    line.set_xdata([node, node])

            # ── Pred 8° ───────────────────────────────────────────────────────
            ov = s.ext_override
            if s.use_pred_8 and ang != 8:
                ov8 = s.data_angles[8].get("ext_override")
                if ov8 is not None:
                    ov = ov8

            # ── Compute R, nf, kf ─────────────────────────────────────────────
            if ov is not None:
                ext_nd, n_arr, k_arr, extra = ov
                n_disp = np.interp(c_nodes, ext_nd, n_arr)
                k_disp = np.interp(c_nodes, ext_nd, k_arr)
                self.dots_n.set_data(c_nodes, n_disp)
                self.dots_k.set_data(c_nodes, k_disp)
                self.extra_dots_n.set_data(ext_nd, n_arr)
                self.extra_dots_k.set_data(ext_nd, k_arr)
                for i, sl in enumerate(self._sliders_n[:14]):
                    sl.set_display_value(n_disp[i])
                for i, sl in enumerate(self._sliders_k[:14]):
                    sl.set_display_value(k_disp[i])

                _refl_kw = ph.state_to_refl_kwargs(s)
                if ang in (40, 60) and isinstance(extra, dict) and 'wc' in extra:
                    wc, wl_e, wr, da = extra['wc'], extra['wl'], extra['wr'], extra['da']
                    ang_f = float(ang)
                    R, nf, kf = ph.calc_refl_3angle(wl_exp, s.d, ext_nd, n_arr, k_arr,
                        ang_f, da, wc, wl_e, wr, **_refl_kw)
                    if hasattr(self, '_lbl_spread'):
                        self._lbl_spread.setText(
                            f"Spread: C={wc:.2f} L={wl_e:.2f} R={wr:.2f} | da={da:.2f}°")
                else:
                    R, nf, kf = ph.calculate_refl_core(wl_exp, s.d, ext_nd, n_arr, k_arr,
                        theta_deg=ang, **_refl_kw)
                    if hasattr(self, '_lbl_spread'):
                        self._lbl_spread.setText("Spread: --")
            else:
                # No override: build n/k from the sliders
                curr_n = np.array([ph.n_map(v) for v in s.n_slider_vals])
                curr_k = np.array([ph.k_map(v) for v in s.k_slider_vals])
                ext_nd = ph._build_ext_nodes(c_nodes)
                n_ext  = np.clip(PchipInterpolator(c_nodes, curr_n)(ext_nd), 1.0, 12.0)
                k_ext  = np.maximum(PchipInterpolator(c_nodes, curr_k)(ext_nd), 0.0)
                self.dots_n.set_data(c_nodes, curr_n)
                self.dots_k.set_data(c_nodes, curr_k)
                self.extra_dots_n.set_data([], [])
                self.extra_dots_k.set_data([], [])
                for i, sl in enumerate(self._sliders_n[:14]):
                    sl.set_display_value(curr_n[i])
                for i, sl in enumerate(self._sliders_k[:14]):
                    sl.set_display_value(curr_k[i])
                if hasattr(self, '_lbl_spread'):
                    self._lbl_spread.setText("Spread: --")
                _refl_kw = ph.state_to_refl_kwargs(s)
                R, nf, kf = ph.calculate_refl_core(wl_exp, s.d, ext_nd, n_ext, k_ext,
                    theta_deg=ang, **_refl_kw)

            # ── Update the plots ───────────────────────────────────────────────
            self.l_exp.set_data(wl_exp, r_exp)   # always updated
            self.l_sim.set_data(wl_exp, R)
            self.l_n.set_data(wl_exp, nf)
            self.l_k.set_data(wl_exp, kf)
            self.ax2.relim(); self.ax2.autoscale_view()
            self.ax2_k.relim(); self.ax2_k.autoscale_view()
            if dd["exp"] is not None:
                res = R - r_exp
                self.l_res.set_data(wl_exp, res)
                res_max = max(float(np.abs(res).max()), 0.5)
                self.ax_res.set_ylim(-res_max * 1.2, res_max * 1.2)
                rms = float(np.sqrt(np.mean(res**2)))
                self.txt_chi.set_text(f'RMS resid: {rms:.3f} %R')
            else:
                # no data loaded \u2192 no meaningful residual (default sim only)
                self.l_res.set_data(wl_exp, np.zeros(len(wl_exp)))
                self.txt_chi.set_text('RMS resid: --')
            self._apply_view_mode(ang)
            self.canvas.draw_idle()
        except Exception as e:
            import traceback
            print(f"[_refresh_simulation] error: {e}")
            traceback.print_exc()
            self.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-angle Fit / Maxima view
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_view_mode(self, ang):
        """Display override for angle `ang` according to view_mode.

        'fit'    → standard behavior (fit curves visible).
        'maxima' → n = n_eff from the maxima (l_n_max), zeroed residual, k per
                   policy (60° hidden; others 'k from fit'), maxima markers on the
                   reflectance + faded fit R, disclaimer. It is a VISIBILITY/label
                   only layer: it doesn't touch the curve data → reversible.
        On-screen text ALWAYS in English.
        """
        from scipy.interpolate import PchipInterpolator
        dd = self.state.data_angles.get(ang, {})
        mode = dd.get('view_mode', 'fit')
        nmax = dd.get('n_maxima')
        show_max = (mode == 'maxima') and (nmax is not None) and (len(nmax) > 0)

        # ── n: maxima vs fit ──
        fit_n = [self.l_n, self.dots_n, self.extra_dots_n, self._band_n[0]]
        # Data of the n-from-maxima curve: ALWAYS set if available (even in fit
        # mode, where they stay hidden), so they enter the n-axis limit
        # computation → the axis frames the union fit∪maxima and is identical in
        # both modes (no jump, no clipping).
        if nmax is not None and len(nmax) > 0:
            arr = np.asarray(nmax, dtype=float); arr = arr[arr[:, 0].argsort()]
            try:
                xs = np.linspace(float(arr[:, 0].min()), float(arr[:, 0].max()), 200)
                ys = PchipInterpolator(arr[:, 0], arr[:, 1])(xs)
            except Exception:
                xs, ys = arr[:, 0], arr[:, 1]
            self.l_n_max.set_data(xs, ys)
            self.dots_n_max.set_data(arr[:, 0], arr[:, 1])
        else:
            self.l_n_max.set_data([], []); self.dots_n_max.set_data([], [])
        self.l_n_max.set_visible(show_max); self.dots_n_max.set_visible(show_max)
        for a in fit_n: a.set_visible(not show_max)
        self.l_n_kk.set_visible(not show_max)

        # n axis: ALWAYS frames the union fit-n ∪ n-from-maxima (only n; k stays
        # as is). Stable across the toggle (the union doesn't depend on the mode)
        # and without clipping n_eff.
        yvals = []
        for art in (self.l_n, self.l_n_max):
            yd = np.asarray(art.get_ydata(), dtype=float)
            if yd.size:
                yvals.append(yd[np.isfinite(yd)])
        yvals = [v for v in yvals if v.size]
        if yvals:
            allv = np.concatenate(yvals)
            lo, hi = float(allv.min()), float(allv.max())
            pad = max(0.03, 0.05 * (hi - lo))
            self.ax2.set_ylim(lo - pad, hi + pad)

        # ── residual: zeroed in maxima ──
        self.l_res.set_visible(not show_max)
        self._maxima_res_note.set_visible(show_max)

        # ── k: 60° hidden, others 'k from fit' ──
        fit_k = [self.l_k, self.dots_k, self.extra_dots_k, self._band_k[0]]
        if show_max and ang == 60:
            for a in fit_k: a.set_visible(False)
            self.l_k_kk.set_visible(False)
        else:
            for a in fit_k: a.set_visible(True)
            self.l_k.set_label('k from fit' if show_max else 'k')
            self.l_k_kk.set_visible(not show_max)

        # ── reflectance: maxima markers (no labels) + faded fit R ──
        if show_max:
            peaks = dd.get('peaks_wl'); exp = dd.get('exp')
            if peaks is not None and exp is not None and len(peaks):
                pk = np.asarray(peaks, dtype=float)
                self.dots_peaks.set_data(pk, np.interp(pk, exp[0], exp[1]))
                self.dots_peaks.set_visible(True)
            else:
                self.dots_peaks.set_visible(False)
            self.l_sim.set_alpha(0.25)
        else:
            self.dots_peaks.set_visible(False)
            self.l_sim.set_alpha(1.0)

        # ── disclaimer (only if we are really showing the maxima view) ──
        txt = self._DISCLAIMER.get(ang, '') if show_max else ''
        self._maxima_note.set_text(txt)
        self._maxima_note.set_visible(bool(txt))

        # ── n/k legend rebuilt from the visible artists only ──
        self._rebuild_nk_legend()

    def _rebuild_nk_legend(self):
        """Rebuild the combined n/k legend from the VISIBLE artists only with a
        useful label (handles the relabel 'k'→'k from fit', the 'n from maxima'
        entry, removal of k at 60° in maxima mode)."""
        pairs = []
        for art in (self.l_n, self.l_n_max, self.l_n_kk, self.l_k, self.l_k_kk):
            if art.get_visible():
                lbl = art.get_label()
                if lbl and not lbl.startswith('_'):
                    pairs.append((art, lbl))
        if pairs:
            self.ax2.legend([p[0] for p in pairs], [p[1] for p in pairs],
                            fontsize=7, loc='upper left',
                            bbox_to_anchor=(1.08, 1), borderaxespad=0)

    def _on_toggle_view_mode(self):
        """Toggle the current angle's view_mode fit↔maxima. When switching to
        'maxima' it computes n_maxima if missing (d = d_mem_global or state.d). If
        the peaks/m have not been computed yet (Peak window), it warns."""
        s = self.state; ang = s.current_ang
        dd = s.data_angles.get(ang)
        if dd is None or dd.get('exp') is None:
            self.status.showMessage("Load data for this angle first")
            return
        new_mode = 'maxima' if dd.get('view_mode', 'fit') == 'fit' else 'fit'
        if new_mode == 'maxima' and dd.get('n_maxima') is None:
            d = s.d_mem_global if getattr(s, 'd_mem_global', None) else s.d
            arr = self.engine.compute_n_from_maxima(ang, d)
            if arr is None or len(arr) == 0:
                self.status.showMessage(
                    f"{ang}°: compute maxima/d in the Peak window first")
                return
        dd['view_mode'] = new_mode
        self._refresh_view_mode_btn()
        self._refresh_simulation()

    def _refresh_view_mode_btn(self):
        """Update the toggle's text/color according to the current angle's
        view_mode."""
        if not hasattr(self, 'btn_view_mode'):
            return
        dd = self.state.data_angles.get(self.state.current_ang, {})
        is_max = dd.get('view_mode', 'fit') == 'maxima'
        self.btn_view_mode.setText("View: MAXIMA" if is_max else "View: FIT")
        self.btn_view_mode.setStyleSheet(
            "QPushButton { background-color: %s; color: black; font-weight: bold; "
            "border-radius: 4px; }" % ('#FFD27F' if is_max else '#D6EAF8'))

    # ─────────────────────────────────────────────────────────────────────────
    # UI updates during the fit (slots connected to the threads)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_fit_progress(self, msg: str):
        self.txt_chi.set_text(msg)
        self.status.showMessage(msg)
        self.canvas.draw_idle()

    def _on_fit_step(self, chi2: float, n_iter: int,
                     R, nf, kf, ext, en, ek, ang):
        """UI-update callback during L-BFGS-B, for the angle `ang` being fitted.

        Reused as the `on_fit_step` callback of `_on_fit_multi` and `_on_fit_auto`
        to give continuous feedback during the longest phase of the fit. L-BFGS-B
        calls on_fit_step on every iteration, throttled to 0.5s in the engine.

        For consistency with the `_on_fit` closure (single-angle), it also includes
        residuals + flush_events + processEvents for a real refresh even on the Agg
        backend (Spyder).

        `ang` comes from the engine, not from state.current_ang: the live curve is
        drawn only while its own angle is on screen; the other angles keep showing
        their own saved fit.
        """
        if ang != self.state.current_ang:
            return
        ang_data = self.state.data_angles[ang]["exp"]
        if ang_data is None:
            return
        wl_exp, r_exp = ang_data
        self.l_sim.set_data(wl_exp, R)
        self.l_n.set_data(wl_exp, nf)
        self.l_k.set_data(wl_exp, kf)
        self.extra_dots_n.set_data(ext, en)
        self.extra_dots_k.set_data(ext, ek)
        # Residuals (if available): helps to visually see convergence
        try:
            res = R - r_exp
            self.l_res.set_data(wl_exp, res)
            res_max = max(float(np.abs(res).max()), 0.5)
            self.ax_res.set_ylim(-res_max * 1.2, res_max * 1.2)
            rms = float(np.sqrt(np.mean(res**2)))
            self.txt_chi.set_text(
                f'RMS resid: {rms:.3f} %R\n'
                f'(fit obj: {chi2:.2e})')
        except Exception:
            # Fallback if l_res doesn't exist (defensive)
            self.txt_chi.set_text(f'fit obj: {chi2:.2e}\n(fitting…)')
        self.txt_iter.set_text(f'Iter: {n_iter}')
        # Real canvas update + Qt event loop. flush_events bypasses Qt and updates
        # the Agg backend (needed in Spyder).
        try:
            self.canvas.draw()
            self.canvas.flush_events()
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

    def _on_fit_done(self, result: dict, ang: int = None):
        self._fit_thread = None
        if not result:
            self.status.showMessage("Fit failed or stopped")
            return
        # The angle that was FITTED, passed in by the caller (not state.current_ang,
        # which the user is free to change while the fit runs).
        if ang is None:
            ang = self.state.current_ang
        self._update_sliders_from_state()

        # ── ±σ uncertainty bands ──
        try:
            sigma_n = result.get('sigma_n')
            sigma_k = result.get('sigma_k')
            en      = result.get('en')
            ek      = result.get('ek')
            ext     = result.get('ext')
            dd      = self.state.data_angles[ang]
            # Save sigma_k and ext in the state for the KK propagation
            if sigma_k is not None and ext is not None:
                self.state._last_sigma_k   = sigma_k
                self.state._last_sigma_ext = ext
            # Per-angle persistence: used by the SAVE/SAVE ALL buttons
            if (sigma_n is not None and sigma_k is not None and ext is not None):
                dd["sigma_nk"] = (np.asarray(ext, dtype=float),
                                  np.asarray(sigma_n, dtype=float),
                                  np.asarray(sigma_k, dtype=float))
            if (sigma_n is not None and sigma_k is not None and
                    en is not None and ek is not None and
                    ext is not None and dd["exp"] is not None):
                wl_exp = dd["exp"][0]
                n_mid = np.interp(wl_exp, ext, en)
                k_mid = np.interp(wl_exp, ext, ek)
                sn    = np.interp(wl_exp, ext, sigma_n)
                sk    = np.interp(wl_exp, ext, sigma_k)
                # Limit the bands to physically sensible values
                sn = np.minimum(sn, 0.05 * n_mid)
                sk = np.minimum(sk, 0.50 * np.maximum(k_mid, 1e-6))
                # Remove the old bands and redraw
                self._band_n[0].remove()
                self._band_k[0].remove()
                self._band_n[0] = self.ax2.fill_between(
                    wl_exp, n_mid - sn, n_mid + sn,
                    color='darkgreen', alpha=0.18)
                self._band_k[0] = self.ax2_k.fill_between(
                    wl_exp, np.maximum(k_mid - sk, 0), k_mid + sk,
                    color='purple', alpha=0.18)
        except Exception as e:
            print(f"[sigma bands] {e}")

        n_iter = result.get('n_iter', '--')
        self.txt_iter.set_text(f'Iter: {n_iter} — done')

        # Show the R curve directly from the result (avoids inconsistent recompute)
        # Show the RMS reflectance residual (pure data misfit), not the fit objective
        # which includes the regularization penalties. RMS = sqrt(mean(res^2)) in %R,
        # an unweighted least-squares misfit, NOT a sigma-normalized reduced chi2.
        R_final = result.get('R')
        dd = self.state.data_angles[ang]
        if R_final is not None and dd["exp"] is not None:
            wl_exp, r_exp = dd["exp"]
            res = R_final - r_exp
            rms = float(np.sqrt(np.mean(res**2)))
            self.l_sim.set_data(wl_exp, R_final)
            self.l_res.set_data(wl_exp, res)
            res_max = max(float(np.abs(res).max()), 0.5)
            self.ax_res.set_ylim(-res_max * 1.2, res_max * 1.2)
            self.txt_chi.set_text(f'RMS resid: {rms:.3f} %R')
            self.status.showMessage(f"Fit done — RMS residual: {rms:.3f} %R")
        else:
            self.status.showMessage("Fit done")

        # Save ext_override and sliders into data_angles BEFORE _select_ang
        ov_from_fit = result.get('ext_override') or self.state.ext_override
        sl_from_fit = result.get('sliders')
        if ov_from_fit is not None:
            self.state.data_angles[ang]['ext_override'] = ov_from_fit
            self.state.ext_override = ov_from_fit
        if sl_from_fit is not None:
            self.state.data_angles[ang]['sliders'] = sl_from_fit
        if R_final is not None:
            self.state.data_angles[ang]['fit'] = R_final

        # Update n/k in the plot without recomputing R
        nf = result.get('nf'); kf = result.get('kf')
        if nf is not None and dd["exp"] is not None:
            wl_exp = dd["exp"][0]
            self.l_n.set_data(wl_exp, nf)
            self.l_k.set_data(wl_exp, kf)
            self.ax2.relim(); self.ax2.autoscale_view()
            self.ax2_k.relim(); self.ax2_k.autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

        # Update sliders and buttons without recomputing the curve
        self._update_sliders_from_state()
        self._select_ang(ang)   # updates button colors, doesn't recompute R

    def _on_fit_err(self, msg: str):
        self._fit_thread = None
        self.status.showMessage(f"Fit error: {msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # Button callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _set_fitting_ui(self, fitting: bool, btn_ref=None):
        """Fit-button color during fitting (via _style_button → width, border and
        text consistent with the rest; #FF6600 = fit in progress), and lock the
        controls that must not be touched while the optimizer runs.

        The fit runs in the UI thread and the callbacks call processEvents(), so
        clicks ARE delivered mid-fit. The fit buttons are disabled mid-fit because
        a click would re-enter _on_fit and start a nested fit whose `finally` tears
        down the outer one's callbacks (STOP is the way out); the material combos
        are disabled because changing material swaps physics._film_mat[0] under
        L-BFGS-B, so the model would change while the bounds stay those of the old
        material.
        Deliberately left ENABLED: STOP (it is the way out), the angle buttons
        (looking at another angle during a fit is wanted), and MONO n / MONO k —
        intervening on the constraints mid-fit is a direction we want to keep open.
        """
        for b in (self._btn_adatta, self._btn_fit_multi, self._btn_fit_auto):
            b.setEnabled(not fitting)
        for c in ('combo_film', 'combo_substrate'):
            w = getattr(self, c, None)
            if w is not None:
                w.setEnabled(not fitting)
        if fitting:
            if btn_ref:
                self._style_button(btn_ref, "#FF6600")
        else:
            self._style_button(self._btn_adatta,    "#90EE90")
            self._style_button(self._btn_fit_multi, "#AED6F1")
            self._style_button(self._btn_fit_auto,  "#A8D8A8")

    def _on_fit(self):
        ang = self.state.current_ang
        dd  = self.state.data_angles[ang]
        if dd["exp"] is None:
            self.status.showMessage("No data for this angle")
            return
        wl_exp, r_exp = dd["exp"]
        self._set_fitting_ui(True, self._btn_adatta)
        self.status.showMessage(f"Fitting at {ang}°...")
        self.txt_chi.set_text("Fitting...")
        self.canvas.draw_idle()
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        def on_step(chi2, n_iter, R, nf, kf, ext, en, ek, ang_fit):
            # Only while its own angle is on screen: switching angle mid-fit is
            # allowed, and this curve belongs to `ang_fit` alone.
            if ang_fit != self.state.current_ang:
                return
            self.l_sim.set_data(wl_exp, R)
            self.l_n.set_data(wl_exp, nf)
            self.l_k.set_data(wl_exp, kf)
            self.extra_dots_n.set_data(ext, en)
            self.extra_dots_k.set_data(ext, ek)
            res = R - r_exp
            self.l_res.set_data(wl_exp, res)
            res_max = max(float(np.abs(res).max()), 0.5)
            self.ax_res.set_ylim(-res_max * 1.2, res_max * 1.2)
            rms = float(np.sqrt(np.mean(res**2)))
            self.txt_chi.set_text(f'RMS resid: {rms:.3f} %R')
            self.txt_iter.set_text(f'Iter: {n_iter}')
            # flush_events() bypasses Qt and directly updates the Agg backend (works in Spyder)
            self.canvas.draw()
            self.canvas.flush_events()
            QApplication.processEvents()

        self.engine.on_fit_step = on_step
        self.engine.on_progress = lambda m: (
            self.status.showMessage(m), QApplication.processEvents())
        # Reset iter_limit: a direct fit must run to completion
        # (phase 1 + phase 2 of IR fringe centering). Without this, an iter_limit
        # left over from a previous fit_auto would make phase 2 be skipped
        # (state.stop=True at the limit). run_fit doesn't reset it itself because
        # fit_rapido sets it deliberately.
        self.state.iter_limit = None
        try:
            result = self.engine.run_fit(ang, wl_exp, r_exp)
            self._on_fit_done(result, ang)      # the angle we fitted, not the one on screen
        except Exception as e:
            self.status.showMessage(f"Fit error: {e}")
            print(f"[_on_fit] error: {e}")
        finally:
            self.engine.on_fit_step = None
            self.engine.on_progress = None
            self._set_fitting_ui(False)

    def _on_fit_multi(self):
        ang_avail = [a for a in [8,20,40,60]
                     if self.state.data_angles[a]["exp"] is not None]
        if len(ang_avail) < 2:
            self.status.showMessage("FIT MULTI: need ≥2 angles")
            return
        self._set_fitting_ui(True, self._btn_fit_multi)
        self.status.showMessage("FIT MULTI running...")
        self.txt_chi.set_text("FIT MULTI in progress...")
        self.canvas.draw_idle()
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        # Register on_fit_step TOO (besides on_progress) to give continuous feedback
        # during L-BFGS-B in the various run_fit calls inside fit_multi. Without it
        # only on_progress fires → UI frozen for minutes.
        self.engine.on_fit_step = self._on_fit_step
        self.engine.on_progress = lambda m: (
            self.status.showMessage(m), self.txt_chi.set_text(m),
            self.canvas.draw_idle(), QApplication.processEvents())
        try:
            result = self.engine.run_fit_multi(ang_avail)
            self._on_fit_done(result or {})
        except Exception as e:
            self.status.showMessage(f"FIT MULTI error: {e}")
            print(f"[_on_fit_multi] error: {e}")
        finally:
            self.engine.on_fit_step = None
            self.engine.on_progress = None
            self._set_fitting_ui(False)

    def _on_fit_auto(self):
        ang_avail = [a for a in [8,20,40,60]
                     if self.state.data_angles[a]["exp"] is not None]
        self._set_fitting_ui(True, self._btn_fit_auto)
        self.status.showMessage("FIT AUTO running...")
        self.txt_chi.set_text("FIT AUTO in progress...")
        self.canvas.draw_idle()
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        # on_fit_step registered for continuous feedback during L-BFGS-B (pre-loop
        # full 20° fit, fit_rapido in cycles, final fit_multi). Without it the UI is
        # frozen between the on_progress messages.
        self.engine.on_fit_step = self._on_fit_step
        self.engine.on_progress = lambda m: (
            self.status.showMessage(m), self.txt_chi.set_text(m),
            self.canvas.draw_idle(), QApplication.processEvents())
        try:
            result = self.engine.run_fit_auto(ang_avail)
            self._on_fit_done(result or {})
        except Exception as e:
            self.status.showMessage(f"FIT AUTO error: {e}")
            print(f"[_on_fit_auto] error: {e}")
        finally:
            self.engine.on_fit_step = None
            self.engine.on_progress = None
            self._set_fitting_ui(False)

    def _on_stop(self):
        self.state.stop = True
        self.status.showMessage("STOP requested")

    def _on_mono_n(self):
        self.state.mono_n = not self.state.mono_n
        self._set_toggle_btn(self._btn_mono_n,self.state.mono_n,"MONO n ON","MONO n OFF","#4CAF50","#BBBBBB")

    def _on_mono_k(self):
        self.state.mono_k = not self.state.mono_k
        self._set_toggle_btn(self._btn_mono_k,self.state.mono_k,"MONO k ON","MONO k OFF","#4CAF50","#BBBBBB")

    def _on_k_bound(self):
        self.state.k_bound = not self.state.k_bound
        self._set_toggle_btn(self._btn_k_bound,self.state.k_bound,"K-BOUND ON","K-BOUND OFF","#4CAF50","#BBBBBB")

    def _on_d_source_changed(self, text):
        """Change the peak source for the d calculation.
        Immediate effect both in peak_window and in fit_auto (at the next run,
        because the cache is invalidated at the start of run_fit_auto)."""
        if text in ("exp", "fit", "hybrid"):
            self.state.d_source = text
            # Invalidate the d_peaks cache to reflect the change immediately
            self.engine._d_peaks_cache = None
            self.status.showMessage(f"d source: {text}")

    def _on_thin_film_toggle(self):
        """Toggle thin-sample mode.
        Lowers the λ threshold + activates the single-angle formula branch."""
        self.state.thin_film = not self.state.thin_film
        self._set_toggle_btn(self._btn_thin_film, self.state.thin_film,
                             "THIN ON", "THIN OFF", "#FF9800", "#BBBBBB")
        # Invalidate the d_peaks cache to reflect the change immediately
        self.engine._d_peaks_cache = None
        self.status.showMessage(
            f"thin_film: {'ON' if self.state.thin_film else 'OFF'}")

    def _on_low_contrast_toggle(self):
        """Low-contrast fringes aid: opt-in toggle for the maxima-centering penalty
        (_peak_penalty). Default OFF; when ON it acts only for λ≥500 nm (no deep
        blue). See AppState.low_contrast_aid."""
        self.state.low_contrast_aid = not getattr(self.state, 'low_contrast_aid', False)
        self._set_toggle_btn(self._btn_low_contrast, self.state.low_contrast_aid,
                             "AID ON", "AID OFF", "#FF9800", "#BBBBBB")
        self.status.showMessage(
            f"low-contrast fringe aid: {'ON' if self.state.low_contrast_aid else 'OFF'}")

    def _on_measure_jump(self):
        """Measure the detector jump (~890 nm) from the current angle's data, set
        it on the IR Jump slider (km) and lock it.

        Data-anchored: the estimate uses a separate quadratic fit on the two sides
        of 890 (captures the fringe slope) → km = R_above/R_below. Prevents the fit
        from confusing the instrumental jump with n/k/scatt. The jump varies per
        sample and per angle, so it is measured on the current angle.
        """
        s = self.state
        ang = s.current_ang
        ang_data = s.data_angles.get(ang, {}).get("exp")
        if ang_data is None:
            self.status.showMessage(f"MEAS JUMP: no data loaded at {ang}°")
            return
        wl, R = ang_data
        km = ph.measure_ir_jump(wl, R)
        if km is None:
            self.status.showMessage(
                f"MEAS JUMP: insufficient data around 890 nm ({ang}°)")
            return
        # set_value/set_locked do NOT emit signals → I update the state by hand
        s.km = km
        self._sliders_left[4].set_value(km)
        s.lock_left[4] = True
        self._sliders_left[4].set_locked(True)
        self._refresh_simulation()
        self.status.showMessage(
            f"MEAS JUMP {ang}°: IR Jump = {km:.3f} (measured from the data, "
            f"slider locked)")

    def _on_kk(self):
        s = self.state
        # KK physically valid only at 8° (near-normal incidence)
        if s.current_ang != 8:
            self.status.showMessage("KK: only valid at 8°")
            return
        # Open the dedicated KK window with plot and selective metric
        from windows.kk_window import KKWindow
        if not hasattr(self, '_kk_win') or self._kk_win is None:
            self._kk_win = KKWindow(self.state, self.engine, parent=self)
        else:
            self._kk_win._compute_and_refresh()
        self._kk_win.show()
        self._kk_win.raise_()
        s.kk_on = True
        self._set_toggle_btn(self._btn_kk, True, "KK ON", "KK OFF", "#FF9800", "#DDDDDD")
        # Show n_KK in the main plot if already computed
        if s.kk_cache is not None:
            wl_kk, n_kk, k_kk = s.kk_cache
            self.l_n_kk.set_data(wl_kk, n_kk)
            self.l_k_kk.set_data(wl_kk, k_kk)
            self.ax2.relim(); self.ax2.autoscale_view()
            self.canvas.draw_idle()
        self.status.showMessage("KK window opened — see metric in KK window")

    def _on_kk_uv(self):
        """Apply the KK n to ALL nodes of the current fit.

        The KK n is applied INSIDE ext_override on ALL ext-nodes covered by KK (the
        fitted n stays where KK doesn't reach), keeping k and the number of nodes.
        ext_override is NOT cleared — collapsing to the 14 base nodes and updating
        only n would lose the fit precision (~40 ext-nodes) and leave n/k
        inconsistent. So the step "KK→n + lock n + refit k" works without losing the
        work: n stays fixed at the KK-consistent values (ext-nodes), k is refined.
        """
        s = self.state
        if s.kk_cache is None:
            self.status.showMessage("KK: compute first"); return
        ov = s.ext_override
        if ov is None or len(ov) < 4:
            self.status.showMessage("KK→NODES: run a fit first"); return
        ext_nd, n_arr, k_arr, extra = ov
        ext_nd = np.asarray(ext_nd, dtype=float)
        wl_kk, n_kk, _ = s.kk_cache
        wl_kk = np.asarray(wl_kk, dtype=float)
        n_kk = np.asarray(n_kk, dtype=float)
        kk_lo, kk_hi = float(wl_kk.min()), float(wl_kk.max())
        # KK n on ALL ext-nodes in the KK range; fit unchanged outside; k unchanged
        n_new = np.asarray(n_arr, dtype=float).copy()
        in_kk = (ext_nd >= kk_lo) & (ext_nd <= kk_hi)
        if in_kk.any():
            n_new[in_kk] = np.clip(np.interp(ext_nd[in_kk], wl_kk, n_kk), 1.0, 6.5)
        s.ext_override = (ext_nd, n_new, k_arr, extra)
        # Align the base nodes (display), on ALL those in the KK range, WITHOUT clearing the override
        c_nodes = np.asarray(ph.build_c_nodes(s), dtype=float)
        s.suppress_update = True
        s.suppress_override_clear = True
        for i, cw in enumerate(c_nodes):
            if i < 14 and kk_lo <= cw <= kk_hi:
                internal = ph.inv_n_map(float(np.clip(np.interp(cw, wl_kk, n_kk), 1.0, 6.5)))
                s.n_slider_vals[i] = internal
                if i < len(self._sliders_n):
                    self._sliders_n[i].set_value(internal)
        s.suppress_update = False
        s.suppress_override_clear = False
        self._refresh_simulation()
        self.status.showMessage(
            f"KK→nodes: n updated on {int(in_kk.sum())} ext-nodes "
            f"(whole KK spectrum {kk_lo:.0f}-{kk_hi:.0f}nm); fit and k preserved")

    def _on_pred8(self):
        self.state.use_pred_8 = not self.state.use_pred_8
        self._set_toggle_btn(self._btn_pred8, self.state.use_pred_8,
                             "PRED 8° ON", "PRED 8° OFF", "#FF9800", "#BBBBBB")
        self._refresh_simulation()

    def _on_copia8(self):
        s = self.state; ang = s.current_ang
        if ang == 8: self.status.showMessage("Already at 8°"); return
        src = s.data_angles[8]
        if src["ext_override"] is None: self.status.showMessage("COPY 8°: fit 8° first"); return
        import copy as _copy, physics as ph
        ov8 = src["ext_override"]
        s.data_angles[ang]["ext_override"] = (ov8[0].copy(),ov8[1].copy(),ov8[2].copy(),_copy.copy(ov8[3]) if ov8[3] else {})
        s.ext_override = s.data_angles[ang]["ext_override"]
        sl8 = src.get("sliders") or {}
        s.suppress_update=True; s.suppress_override_clear=True
        for attr in ['d','scatt','inhom','offs','km','ks','kb','pol','w300','w335','w375','w400','w1100']:
            if attr in sl8: setattr(s,attr,sl8[attr])
        ext_nd,n_arr,k_arr,_ = ov8
        c_nd = ph.build_c_nodes(s)
        # Iterate over all base nodes (len(c_nd)=14) so every node — including the
        # last (2550 nm) — is synced from the copied ext_override; otherwise that
        # node's slider and the curve would disagree, and the stale value would
        # enter the curve as soon as a slider clears ext_override.
        for i in range(len(c_nd)):
            s.n_slider_vals[i]=ph.inv_n_map(float(np.interp(c_nd[i],ext_nd,n_arr)))
            s.k_slider_vals[i]=ph.inv_k_map(float(np.interp(c_nd[i],ext_nd,k_arr)))
        s.suppress_update=False; s.suppress_override_clear=False
        self._update_sliders_from_state(); self._refresh_simulation()
        self.status.showMessage(f"8° parameters copied to {ang}°")

    def _on_peaks(self):
        from windows.peak_window import PeakWindow
        if not hasattr(self,'_peak_win') or self._peak_win is None:
            self._peak_win = PeakWindow(self.state,self.engine,parent=self)
        self._peak_win.show(); self._peak_win.raise_()

    def _on_load(self):
        """LOAD = load ALL the angle spectra of one sample from a FOLDER.

        - Choose a folder; if it contains a `# PSi Reflectometer manifest`, that is
          used, otherwise the angles 8/20/40/60 are recognized from the file NAMES
          (see windows/filename_parse.py).
        - A window shows the detected files with CHECKBOXES (the multi-angle
          sample's files pre-checked) + a search box that filters the list live;
          confirm/choose which to load.
        - Loads R_exp only (no params/fit/nodes: for those use LOAD PAR / LOAD SESS;
          for a single angle the per-angle 📂 buttons).
        """
        from windows import filename_parse as fp
        import os as _os
        folder = QFileDialog.getExistingDirectory(
            self, "Folder with the sample's spectra")
        if not folder: return

        # Manifest in the folder? use it (explicit angle: path map)
        for f in sorted(_os.listdir(folder)):
            fp_path = _os.path.join(folder, f)
            if _os.path.isfile(fp_path) and self._is_manifest(fp_path):
                self._load_manifest(fp_path); return

        files = fp.list_angle_files(folder)
        if not files:
            QMessageBox.warning(self, "LOAD",
                f"No angle-spectrum recognized in:\n{folder}")
            return

        # Selection: file list with checkboxes + search filter (same window).
        # The multi-angle sample's files pre-checked → simple case = one OK; the
        # filter shrinks the list when the folder has many files.
        dflt = fp.default_sample(files)
        dlg = QDialog(self)
        dlg.setWindowTitle("LOAD — select the files to load")
        dlg.resize(580, 440)
        v = QVBoxLayout(dlg)
        ed_filter = QLineEdit()
        ed_filter.setPlaceholderText("filter: sample / angle / file name…")
        v.addWidget(ed_filter)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); vin = QVBoxLayout(inner)
        checks = []  # (QCheckBox, sample, angle, path)
        for smp, a, p in files:
            cb = QCheckBox(f"{a:>3}°    {_os.path.basename(p)}    [{smp}]")
            cb.setChecked(smp == dflt)
            vin.addWidget(cb)
            checks.append((cb, smp, a, p))
        vin.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        lbl_sel = QLabel(); lbl_sel.setStyleSheet("font-weight:bold; padding:2px;")
        v.addWidget(lbl_sel)

        def _update_count():
            angs = sorted(a for cb, smp, a, p in checks if cb.isChecked())
            lbl_sel.setText(f"Selected: {len(angs)}  —  angles {angs}")
        for _cb, *_r in checks:
            _cb.stateChanged.connect(lambda *_a: _update_count())
        _update_count()

        def _apply_filter(text):
            t = text.strip().lower()
            for cb, smp, a, p in checks:
                cb.setVisible(t in cb.text().lower())
        ed_filter.textChanged.connect(_apply_filter)

        bb = QDialogButtonBox(
            (QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            if PYQT == 6 else (QDialogButtonBox.Ok | QDialogButtonBox.Cancel))
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        accepted = (QDialog.DialogCode.Accepted if PYQT == 6 else QDialog.Accepted)
        if dlg.exec() != accepted:
            return

        sel = [(smp, a, p) for cb, smp, a, p in checks if cb.isChecked()]
        if not sel:
            self.status.showMessage("LOAD: no file selected"); return
        mapping = {}; conflicts = []
        for smp, a, p in sel:
            if a in mapping:
                conflicts.append(f"{a}°: {_os.path.basename(mapping[a])} / "
                                 f"{_os.path.basename(p)}")
            else:
                mapping[a] = p
        if conflicts:
            QMessageBox.warning(self, "LOAD",
                "Two files selected for the same angle:\n" + "\n".join(conflicts)
                + "\n\nDeselect one of the two.")
            return
        _samples = sorted(set(smp for smp, a, p in sel))
        sample = _samples[0] if len(_samples) == 1 else "selection"

        loaded = []
        for a in (8, 20, 40, 60):
            p = mapping.get(a)
            if not p: continue
            if self.engine.load_spectral_file(p, a) is None:
                continue
            self.state.ang_filenames[a] = _os.path.basename(p)
            loaded.append(a)
        if not loaded:
            self.status.showMessage("LOAD: no spectrum loaded"); return

        first = loaded[0]
        wl_exp, r_exp = self.state.data_angles[first]["exp"]
        self.l_exp.set_data(wl_exp, r_exp)
        self.ax1.set_xlim(200, 2600)
        self.ax_res.set_xlim(200, 2600)
        self._select_ang(first)
        self.state.current_ang = first
        self._refresh_simulation()
        self.canvas.draw_idle()
        self.status.showMessage(f"LOAD: {sample} → angles {loaded}")

    def _is_manifest(self, path):
        """True if the file's first line is the manifest marker."""
        try:
            with open(path, 'r', errors='ignore') as f:
                first = f.readline().strip()
            return first.startswith("# PSi Reflectometer manifest")
        except Exception:
            return False

    def _load_manifest(self, path):
        """Load only spectral data (R_exp) for multiple angles from a manifest.

        Consistent with the LOAD semantics = "I load curves" (no params/fit/nodes:
        for those use LOAD PAR / LOAD SESS).

        User confirmation only if there are already ≥2 angles with loaded data
        (risk of overwriting previous work).
        """
        angle_files = {}
        try:
            with open(path, 'r', errors='ignore') as f:
                for line in f:
                    l = line.strip()
                    if not l or l.startswith('#'): continue
                    if ':' not in l: continue
                    k, v = l.split(':', 1)
                    k = k.strip(); v = v.strip()
                    try:
                        ang = int(k)
                        if ang in (8, 20, 40, 60):
                            angle_files[ang] = v
                    except ValueError:
                        pass    # sample/angles/... ignored
        except Exception as e:
            self.status.showMessage(f"Manifest read error: {e}"); return
        if not angle_files:
            self.status.showMessage("Manifest empty or invalid"); return

        # Confirm if ≥2 angles are already loaded
        n_loaded = sum(1 for a in (8, 20, 40, 60) if self.state.has_data(a))
        if n_loaded >= 2:
            yes_btn = (QMessageBox.StandardButton.Yes if PYQT == 6 else QMessageBox.Yes)
            no_btn  = (QMessageBox.StandardButton.No  if PYQT == 6 else QMessageBox.No)
            reply = QMessageBox.question(
                self, "Manifest LOAD",
                f"There are already data loaded on {n_loaded} angles.\n"
                f"The manifest will overwrite the data for: "
                f"{sorted(angle_files.keys())}.\nProceed?",
                yes_btn | no_btn)
            if reply != yes_btn:
                self.status.showMessage("Manifest LOAD cancelled"); return

        base_dir = os.path.dirname(path)
        loaded = []; failed = []
        for ang in sorted(angle_files.keys()):
            rel = angle_files[ang]
            full = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
            if not os.path.exists(full):
                failed.append((ang, "missing")); continue
            result = self.engine.load_spectral_file(full, ang)
            if result is None:
                failed.append((ang, "parse")); continue
            loaded.append(ang)

        if not loaded:
            self.status.showMessage(
                f"Manifest: no angle loaded (failed={failed})"); return

        # Show the first loaded angle
        first = loaded[0]
        wl_exp, r_exp = self.state.data_angles[first]["exp"]
        self.l_exp.set_data(wl_exp, r_exp)
        self.ax1.set_xlim(200, 2600)
        self.ax_res.set_xlim(200, 2600)
        self._select_ang(first)
        self.state.current_ang = first
        self._refresh_simulation()
        self.canvas.draw_idle()

        msg = f"Manifest LOAD: {loaded}"
        if failed: msg += f"  (failed: {failed})"
        self.status.showMessage(msg)

    def _on_save(self):
        """Save data and parameters of the currently active angle.

        Produces two files:
          {nome}_{ang}deg.par       — all parameters + 14 base nodes (n,k,σ_n,σ_k)
          {nome}_{ang}deg_data.csv  — wl, R_exp, R_fit, continuous n, k [+ n_KK,k_KK] [+ σ_n,σ_k]
        """
        try:
            from PyQt6.QtWidgets import QInputDialog
        except ImportError:
            from PyQt5.QtWidgets import QInputDialog
        nome, ok = QInputDialog.getText(self, "Sample name", "Name:", text="Sample_01")
        if not ok or not nome.strip(): return
        nome = nome.strip()
        cartella = QFileDialog.getExistingDirectory(self, "Output folder")
        if not cartella: return

        ang = self.state.current_ang
        if not self.state.has_data(ang):
            self.status.showMessage("No data loaded"); return
        try:
            self._write_angle(nome, cartella, ang, suffix="")
            self.status.showMessage(f"Saved: {nome}_{ang}deg.par + _data.csv")
        except Exception as e:
            self.status.showMessage(f"Save failed: {e}")

    def _on_save_all(self):
        """Save data and parameters for each angle with loaded data.

        For each angle `ang` with data it produces:
          {nome}_all_{ang}deg.par       — per-angle parameters (taken from dd["sliders"]
                                          if the angle was fitted, otherwise from the
                                          current state)
          {nome}_all_{ang}deg_data.csv  — data + fit + continuous n,k

        The "_all_" suffix visually distinguishes the save-all files from the single saves.
        """
        try:
            from PyQt6.QtWidgets import QInputDialog
        except ImportError:
            from PyQt5.QtWidgets import QInputDialog
        nome, ok = QInputDialog.getText(self, "Sample name", "Name:", text="Sample_01")
        if not ok or not nome.strip(): return
        nome = nome.strip()
        cartella = QFileDialog.getExistingDirectory(self, "Output folder")
        if not cartella: return

        # A loaded angle without a per-angle fit snapshot would be saved with the
        # CURRENT angle's parameters (silent cross-angle contamination, see
        # _collect_angle_params). Detect those angles and give an explicit escape
        # hatch instead of writing a misleading .par silently.
        loaded   = [a for a in [8, 20, 40, 60] if self.state.has_data(a)]
        unfitted = [a for a in loaded
                    if a != self.state.current_ang
                    and not self.state.data_angles[a].get("sliders")]
        if unfitted:
            yes_btn = (QMessageBox.StandardButton.Yes if PYQT == 6 else QMessageBox.Yes)
            no_btn  = (QMessageBox.StandardButton.No  if PYQT == 6 else QMessageBox.No)
            reply = QMessageBox.question(
                self, "Missing fit",
                "A fit is not present for all loaded spectra "
                f"({', '.join(f'{a}°' for a in unfitted)}).\n"
                "Their parameters would be taken from the current angle "
                f"({self.state.current_ang}°).\n\nSave anyway?",
                yes_btn | no_btn, no_btn)
            if reply != yes_btn:
                self.status.showMessage("Save all cancelled (missing fit)")
                return

        saved = []; failed = []
        for ang in [8, 20, 40, 60]:
            if not self.state.has_data(ang): continue
            try:
                self._write_angle(nome, cartella, ang, suffix="_all")
                saved.append(ang)
            except Exception as e:
                failed.append(ang)
                print(f"[save_all] {ang}°: {e}")

        # ── Manifest: single entry point for multi-angle reloading ──────────
        if saved:
            import os, datetime
            manifest_path = os.path.join(cartella, f"{nome}_all.csv")
            today = datetime.date.today().isoformat()
            with open(manifest_path, 'w') as f:
                f.write("# PSi Reflectometer manifest\n")
                f.write(f"# Created: {today}\n")
                f.write(f"sample: {nome}\n")
                f.write(f"angles: {','.join(str(a) for a in saved)}\n")
                for ang in saved:
                    # Path relative to the manifest's own folder
                    f.write(f"{ang}: {nome}_all_{ang}deg_data.csv\n")

        msg = f"Saved all angles: {saved}"
        if failed: msg += f"  (failed: {failed})"
        if saved: msg += f"  + manifest {nome}_all.csv"
        self.status.showMessage(msg)

    def _on_save_as_fit(self):
        """Promote the current on-screen state of the active angle to a per-angle
        fit snapshot, WITHOUT running an optimization.

        Use when the auto-computed per-angle curve is already good enough by eye:
        it stores the same dd["sliders"]/ext_override/fit a real fit would, so
        SAVE ALL saves THIS angle's true parameters (no fallback to the current
        angle) and the 'missing fit' warning no longer flags it.
        """
        from scipy.interpolate import PchipInterpolator
        s   = self.state
        ang = s.current_ang
        if not s.has_data(ang):
            self.status.showMessage("No data loaded for this angle")
            return
        wl_exp, _ = s.data_angles[ang]["exp"]
        # da for this angle (0 for base angles, spread value for HOBL) — same
        # source _collect_angle_params reads from.
        da = 0.0
        ov = s.data_angles[ang].get("ext_override") or s.ext_override
        if ov is not None and isinstance(ov[3], dict):
            da = float(ov[3].get('da', 0.0))
        # ext_override: prefer the live one; if absent, freeze n,k from the
        # current sliders so this angle keeps ITS n,k (mirror of _write_angle).
        if s.ext_override is not None:
            ext_override = s.ext_override
        else:
            c_nodes  = ph.build_c_nodes(s)
            curr_n_s = np.array([ph.n_map(v) for v in s.n_slider_vals])
            curr_k_s = np.array([ph.k_map(v) for v in s.k_slider_vals])
            ext_nd   = ph._build_ext_nodes(c_nodes)
            n_ext    = np.clip(PchipInterpolator(c_nodes, curr_n_s)(ext_nd), 1.0, 12.0)
            k_ext    = np.maximum(PchipInterpolator(c_nodes, curr_k_s)(ext_nd), 0.0)
            ext_override = (ext_nd, n_ext, k_ext, {'da': da})
        # R_fit = the curve currently displayed (the one being judged good)
        try:
            R_fit = np.asarray(self.l_sim.get_ydata(), float)
            if R_fit.shape != wl_exp.shape:
                R_fit = np.zeros_like(wl_exp)
        except Exception:
            R_fit = np.zeros_like(wl_exp)
        ang_params = {'km': s.km, 'ks': s.ks, 'kb': s.kb,
                      'pol': s.pol, 'dn': s.dn, 'pm': s.pm}
        self.engine._save_angle_state(ang, R_fit, ext_override, ang_params, da)
        self.status.showMessage(
            f"{ang}°: current state saved as fit (d={s.d:.0f} nm, no optimization)")

    def _collect_angle_params(self, ang):
        """Return the dict of scalar parameters for saving angle `ang`.

        For the currently active angle it reads from `state.*` (reflects the latest
        manual slider changes). For other angles it uses the `dd["sliders"]`
        snapshot saved by fit_engine._save_angle_state, falling back to state.* if
        not available.
        """
        s = self.state
        keys = ['d','scatt','inhom','offs','km','ks','kb','pol','dn','pm','ps1',
                'lmin','w300','w335','w375','w400','w1100']
        sl = s.data_angles[ang].get("sliders") if ang != s.current_ang else None
        if sl:
            out = {k: float(sl.get(k, getattr(s, k))) for k in keys}
            out['da'] = float(sl.get('da', 0.0))
        else:
            out = {k: float(getattr(s, k)) for k in keys}
            # da: from the global ext_override if present for this angle
            da = 0.0
            ov = s.data_angles[ang].get("ext_override") or s.ext_override
            if ov is not None:
                _, _, _, extra = ov
                if isinstance(extra, dict):
                    da = float(extra.get('da', 0.0))
            out['da'] = da
        return out

    def _write_angle(self, nome, cartella, ang, suffix=""):
        """Helper: write .par + _data.csv for a single angle.

        For the HOBL angles (40°, 60°) it uses `calc_refl_3angle` with the angular
        spread parameters (wc, wl, wr, da) taken from the ext_override extra.
        """
        import os
        from scipy.interpolate import PchipInterpolator
        s  = self.state
        dd = s.data_angles[ang]
        if dd["exp"] is None: return
        wl_exp, r_exp = dd["exp"]
        par_dict = self._collect_angle_params(ang)
        da_val   = par_dict['da']
        # The node grid comes from THIS angle's W-nodes (par_dict), not from
        # state.*: build_c_nodes(s) would use the currently selected angle's
        # wavelengths, so a SAVE ALL would write a .par whose header w300..w1100
        # and node-table wavelengths disagree, which _on_load_params then reads
        # back positionally into wrong n/k.
        c_nodes = ph._apply_w_nodes(
            ph.nodes_wl_base, par_dict['w300'], par_dict['w335'],
            par_dict['w375'], par_dict['w400'], par_dict['w1100'])

        # ── R_fit + continuous n(λ), k(λ) + n,k at the base nodes ────────────
        # ext_override: prefer the per-angle one; fall back to the global one only if ang is active
        ov = dd.get("ext_override")
        if ov is None and ang == s.current_ang:
            ov = s.ext_override
        # Compute the saved R_fit from the PER-ANGLE parameters (par_dict), NOT from
        # the current state: state_to_refl_kwargs(s) + s.d would apply the
        # currently-selected angle's scalars to every angle during SAVE ALL,
        # cross-contaminating the contrast envelope (pol/dn/pm) and km/ks/kb between
        # angles so the saved R_fit would not match the on-screen curve, and would
        # disagree with the .par it is saved alongside.
        refl_kw = dict(
            scatt=par_dict['scatt'], offset=par_dict['offs'], inhom=par_dict['inhom'],
            kjump=par_dict['km'], alpha=par_dict['ks'], beta=par_dict['kb'],
            pol=par_dict['pol'], delta_n=par_dict['dn'], phi_mix=par_dict['pm'],
            phi_s1=par_dict['ps1'])
        d_ang = par_dict['d']
        if ov is not None:
            ext_nd, n_ext, k_ext, extra = ov
            curr_n_s = np.interp(c_nodes, ext_nd, n_ext)
            curr_k_s = np.interp(c_nodes, ext_nd, k_ext)
            if ang in (40, 60) and isinstance(extra, dict) and 'wc' in extra:
                R_fit, n_wl, k_wl = ph.calc_refl_3angle(
                    wl_exp, d_ang, ext_nd, n_ext, k_ext, float(ang),
                    da_val, extra['wc'], extra['wl'], extra['wr'], **refl_kw)
            else:
                R_fit, n_wl, k_wl = ph.calculate_refl_core(
                    wl_exp, d_ang, ext_nd, n_ext, k_ext,
                    theta_deg=float(ang), **refl_kw)
        else:
            curr_n_s = np.array([ph.n_map(v) for v in s.n_slider_vals])
            curr_k_s = np.array([ph.k_map(v) for v in s.k_slider_vals])
            ext_nd = ph._build_ext_nodes(c_nodes)
            n_ext  = np.clip(PchipInterpolator(c_nodes, curr_n_s)(ext_nd), 1.0, 12.0)
            k_ext  = np.maximum(PchipInterpolator(c_nodes, curr_k_s)(ext_nd), 0.0)
            R_fit, n_wl, k_wl = ph.calculate_refl_core(
                wl_exp, d_ang, ext_nd, n_ext, k_ext,
                theta_deg=float(ang), **refl_kw)

        rms = float(np.sqrt(np.mean((R_fit - r_exp)**2)))

        # ── σ at the base nodes (for .par) and at wl_exp (for _data.csv) ─────
        sigma_data = dd.get("sigma_nk")
        if sigma_data is not None:
            ext_sg, sn_ext, sk_ext = sigma_data
            sn_nodes = np.interp(c_nodes, ext_sg, sn_ext)
            sk_nodes = np.interp(c_nodes, ext_sg, sk_ext)
            sn_wl    = np.interp(wl_exp,  ext_sg, sn_ext)
            sk_wl    = np.interp(wl_exp,  ext_sg, sk_ext)
        else:
            sn_nodes = np.zeros(len(c_nodes))
            sk_nodes = np.zeros(len(c_nodes))
            sn_wl    = sk_wl = None

        # ── File paths ───────────────────────────────────────────────────────
        stem = f"{nome}{suffix}_{ang}deg"
        par_path  = os.path.join(cartella, f"{stem}.par")
        data_path = os.path.join(cartella, f"{stem}_data.csv")

        # ── Write the .par ───────────────────────────────────────────────────
        # KK only at near-normal incidence: the Kramers-Kronig relation between
        # n and k holds for the normal-incidence optical constants, so n_KK means
        # nothing beside an oblique fit. It is a property of the SAMPLE, obtained
        # at 8°.
        kk = getattr(s, 'kk_metric', None) if ang == 8 else None
        # "Real" material lists (populated from materials/): consistent with the combo boxes
        film_name = (ph._FILM_MATS[s.film_mat]
                     if 0 <= s.film_mat < len(ph._FILM_MATS) else "?")
        sub_name  = (ph._SUBSTRATI[s.substrate]
                     if 0 <= s.substrate < len(ph._SUBSTRATI) else "?")
        with open(par_path, 'w') as f:
            f.write("# PSi Reflectometer parameter file\n")
            f.write(f"# Sample: {nome}  Angle: {ang} deg\n")
            f.write(f"# RMS residual (%R): {rms:.4f}\n")
            if kk:
                f.write(f"# KK RMS(n-n_KK): {kk.get('rms', 0):.6f}"
                        f"  KK max|Dn|/n: {kk.get('max_rel', 0):.6f}\n")
            f.write("# Materials (symbolic names)\n")
            f.write(f"film_mat: {film_name}\n")
            f.write(f"substrate: {sub_name}\n")
            f.write("# Sample/instrument parameters\n")
            for k in ['d','scatt','inhom','offs']:
                label = 'offset' if k == 'offs' else k
                f.write(f"{label}: {par_dict[k]:.4f}\n")
            f.write("# IR correction\n")
            for k in ['km','ks','kb']:
                f.write(f"{k}: {par_dict[k]:.4f}\n")
            # Fringe-contrast envelope: the pol/dn/pm sliders are the contrast
            # amplitude / vertex / curvature (UI "Fr contrast / Fr ctr vtx /
            # Fr ctr curv"). Old files with polariz/dn/pm still load via the
            # aliases in _on_load_params.
            f.write("# Fringe contrast (envelope: amplitude, vertex, curvature)\n")
            f.write(f"fr_contrast: {par_dict['pol']:.4f}\n")
            f.write(f"fr_ctr_vtx: {par_dict['dn']:.4f}\n")
            f.write(f"fr_ctr_curv: {par_dict['pm']:.4f}\n")
            f.write("# UV phase + angular spread (HOBL)\n")
            f.write(f"ps1: {par_dict['ps1']:.4f}\n")
            f.write(f"da: {da_val:.4f}\n")
            f.write("# W-nodes (movable base-node wavelengths)\n")
            for k in ['w300','w335','w375','w400','w1100']:
                f.write(f"{k}: {par_dict[k]:.4f}\n")
            f.write(f"lmin: {par_dict['lmin']:.4f}\n")
            f.write("# Base nodes (14): wl_nm  n  k  sigma_n  sigma_k\n")
            for i in range(len(c_nodes)):
                f.write(f"{c_nodes[i]:.2f}\t{curr_n_s[i]:.8f}\t{curr_k_s[i]:.8f}"
                        f"\t{sn_nodes[i]:.6e}\t{sk_nodes[i]:.6e}\n")

        # ── Write the _data.csv ──────────────────────────────────────────────
        header_lines = [
            "PSi Reflectometer data file",
            f"Sample: {nome}  Angle: {ang} deg",
            f"d={par_dict['d']:.2f}nm  scatt={par_dict['scatt']:.3f}  "
            f"inhom={par_dict['inhom']:.3f}%  offset={par_dict['offs']:.4f}  "
            f"RMS residual(%R)={rms:.4f}",
            f"film_mat={film_name}  substrate={sub_name}",
        ]
        if kk:
            header_lines.append(
                f"KK RMS={kk.get('rms', 0):.6f}  KK max_rel={kk.get('max_rel', 0):.6f}")

        cols      = [wl_exp, r_exp, R_fit, n_wl, k_wl]
        col_names = ["lambda_nm","R_exp_pct","R_fit_pct","n","k"]
        fmt_list  = ['%.2f','%.6f','%.6f','%.6f','%.6f']
        if ang == 8 and s.kk_cache is not None:      # see the .par header above
            wl_kk, n_kk_arr, k_kk_arr = s.kk_cache
            cols      += [np.interp(wl_exp, wl_kk, n_kk_arr),
                          np.interp(wl_exp, wl_kk, k_kk_arr)]
            col_names += ["n_KK","k_KK"]
            fmt_list  += ['%.6f','%.6f']
        if sn_wl is not None:
            cols      += [sn_wl, sk_wl]
            col_names += ["sigma_n","sigma_k"]
            fmt_list  += ['%.4e','%.4e']
        header_lines.append(",".join(col_names))
        np.savetxt(data_path, np.column_stack(cols),
                   header="\n".join(header_lines),
                   delimiter=',', fmt=fmt_list, comments='# ')

    def _on_load_params(self):
        """Load a .par file. Backward-compatible with old-format files:
        - missing numeric keys → value left unchanged (doesn't reset)
        - missing materials → film_mat and substrate left unchanged
        - fewer than 14 nodes → only the present ones loaded (the rest unchanged)
        - nodes with 3 or 5 columns (the extra σ_n/σ_k are ignored)
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "Parameters (*.par *.txt);;All (*)")
        if not path: return
        s = self.state
        s.suppress_update = True
        s.suppress_override_clear = True
        s.ext_override = None
        try:
            param_dict = {}    # numeric (lower-case key)
            param_str  = {}    # non-numeric strings (materials, etc.)
            node_rows  = []    # [[wl, n, k], ...] — extra columns ignored

            with open(path, 'r') as f:
                for line in f:
                    l = line.strip()
                    if not l or l.startswith('#'): continue
                    if ':' in l and ',' not in l:
                        k, v = l.split(':', 1)
                        k = k.strip().lower(); v = v.strip()
                        try:
                            param_dict[k] = float(v)
                        except ValueError:
                            param_str[k] = v
                    else:
                        parts = l.replace(',', ' ').split()
                        if len(parts) == 2:
                            try: param_dict[parts[0].strip().lower()] = float(parts[1])
                            except ValueError: pass
                        elif len(parts) >= 3:
                            try: node_rows.append([float(p) for p in parts[:3]])
                            except ValueError: pass

            # ── Scalar parameters (aliases for backward compatibility) ───────
            key_map = {
                'd':     ('d', 'd_nm'),
                'scatt': ('scatt',),
                'inhom': ('inhom',),
                'offs':  ('offs', 'offset'),
                'km':    ('km', 'ir_jump'),
                'ks':    ('ks', 'ir_exp'),
                'kb':    ('kb', 'ir_dmp'),
                'pol':   ('pol', 'polariz', 'fr_contrast'),
                'dn':    ('dn', 'fr_ctr_vtx'),
                'pm':    ('pm', 'fr_ctr_curv'),
                'ps1':   ('ps1',),
                'lmin':  ('lmin',),
                'w300':  ('w300',),
                'w335':  ('w335',),
                'w375':  ('w375',),
                'w400':  ('w400',),
                'w1100': ('w1100',),
            }
            for attr, keys in key_map.items():
                for k in keys:
                    if k in param_dict:
                        setattr(s, attr, param_dict[k])
                        break

            # ── Symbolic materials (absent or unknown → unchanged) ───────────
            mat_changed = False
            film_str = param_str.get('film_mat')
            new_idx = ph.resolve_material('film', film_str) if film_str else None
            if new_idx is not None and new_idx != s.film_mat:
                s.film_mat = new_idx; mat_changed = True
            sub_str = param_str.get('substrate')
            new_idx = ph.resolve_material('sub', sub_str) if sub_str else None
            if new_idx is not None and new_idx != s.substrate:
                s.substrate = new_idx; mat_changed = True
            if mat_changed:
                import sys
                sys.modules['physics']._film_mat[0]  = s.film_mat
                sys.modules['physics']._substrate[0] = s.substrate

            # ── Nodes (n, k): up to 14, shorter files handled ───────────────
            n_load = min(len(node_rows), 14)
            for i in range(n_load):
                n_val = node_rows[i][1]
                k_val = node_rows[i][2]
                s.n_slider_vals[i] = ph.inv_n_map(np.clip(n_val, 1.0, 12.0))
                s.k_slider_vals[i] = ph.inv_k_map(np.clip(k_val, 1e-4, 15.0))

            s.suppress_update = False
            s.suppress_override_clear = False

            if mat_changed:
                self._sync_material_combos()

            self._update_sliders_from_state()
            self._refresh_simulation()
            self.status.showMessage(f"PAR loaded: {path.split('/')[-1]}  "
                                    f"({n_load} nodes)")
        except Exception as e:
            s.suppress_update = False
            s.suppress_override_clear = False
            self.status.showMessage(f".par error: {e}")

    def _on_save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "", "Session (*.psi_sess)")
        if not path: return
        # Reference label for the table/figures (e.g. A_15_200_9_1). Default = the
        # current label, otherwise the sample name inferred from the loaded files.
        from windows import filename_parse as fp
        try: from PyQt6.QtWidgets import QInputDialog
        except ImportError: from PyQt5.QtWidgets import QInputDialog
        default = getattr(self.state, 'sample_label', '') or ''
        if not default:
            for f in self.state.ang_filenames.values():
                if f:
                    default = fp.sample_name(f); break
        lbl, ok = QInputDialog.getText(
            self, "Reference label",
            "Sample label (shown in the table/figures):", text=default)
        if ok and lbl.strip():
            self.state.sample_label = lbl.strip()
        self.engine.save_session(path)
        self._last_session_path = path
        self.status.showMessage(f"Session saved: {path}")

    def _on_load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", "", "Session (*.psi_sess)")
        if not path: return
        ok = self.engine.load_session(path)
        if ok:
            self._last_session_path = path
            ang = self.state.current_ang
            # LOAD SESS is the opposite of SAVE SESS: the session has to come back
            # as the active one in every respect, controls included. The session
            # carries its own materials AND its own mode flags (MONO n/k, KK,
            # PRED 8°, THIN, AID, d source) — those live in the buttons' own label
            # and colour, so they must be told.
            self._sync_material_combos()
            self._update_all_toggles()
            self._update_sliders_from_state()
            self._select_ang(ang)
            self.status.showMessage(f"Session loaded: {path}")
        else:
            self.status.showMessage("Error loading session")

    def _on_export_campaign(self):
        """Append the current fit to the campaign log (one CSV row).

        Auto-collects the computable fields (sample, angle, d, fringe POSITION
        residual in the transparency window, RMS residual, Δn from the fit and from
        the maxima) and only asks for the JUDGEMENT fields (run, m_resolved,
        kk_consistent, locked_params + reason, notes). The misfit is logged as
        `rms_resid_pct`, an unweighted RMS in %R — not a chi², reduced or otherwise:
        the spectra carry no per-point sigma. How to use it is left to the operator
        (the judgement fields carry that decision). Engine untouched — data
        extraction only.

        A checkbox "append ALL angles with data" logs one row per angle (per-angle
        auto metrics) in a single click, with the judgement fields shared — for
        when an accepted multi-angle fit is being recorded."""
        from windows import campaign_export as ce
        try:
            auto = ce.gather_auto(self)
        except Exception as e:
            QMessageBox.warning(self, "→ Campaign",
                                f"Could not collect the fit data:\n{e}")
            return

        angs = [a for a in (8, 20, 40, 60) if self.state.has_data(a)]

        dlg = QDialog(self)
        dlg.setWindowTitle("→ Campaign — add fit")
        form = QFormLayout(dlg)
        info = QLabel(
            f"<b>sample</b> {auto['sample']}   "
            f"<b>angle</b> {auto['angle_deg']}°   <b>d</b> {auto['d_nm']} nm<br>"
            f"<b>fringe-pos residual</b> {auto['fringe_pos_residual_nm']} nm   "
            f"<b>scatter</b> {auto['fringe_pos_scatter_nm']} nm   "
            f"<b>RMS resid</b> {auto['rms_resid_pct']} %R<br>"
            f"<b>Δn_fit</b> off={auto['dn_fit_offset']} "
            f"min={auto['dn_fit_min']} max={auto['dn_fit_max']}   "
            f"<b>Δn_maxima</b> {auto['dn_maxima']}<br>"
            f"<b>KK</b> rms={auto['kk_rms']}  RMS/σ_KK={auto['kk_snr']}  "
            f"(λ≥{auto['kk_lmin_nm']} nm)")
        form.addRow(info)

        ed_label = QLineEdit(str(auto['sample']))
        ed_label.setPlaceholderText(
            "e.g. A_15_200_9_1 (ref / HF% / current / time / run)")
        ed_label.setToolTip(
            "User reference label shown in the table/figures (NOT the real file\n"
            "name). The real names stay in the saved session as provenance.")
        form.addRow("reference label", ed_label)

        sp_run = QSpinBox(); sp_run.setRange(1, 99); sp_run.setValue(1)
        cb_m = QComboBox(); cb_m.addItems(["yes", "no", "—"])
        cb_kk = QComboBox(); cb_kk.addItems(["yes", "no", "—"])
        cb_kk.setCurrentText("—")
        ed_lock = QLineEdit(); ed_lock.setPlaceholderText(
            "e.g.: n300,n335 locked to escape a local minimum")
        ed_note = QLineEdit()
        form.addRow("run #", sp_run)
        form.addRow("m resolved? (domain)", cb_m)
        form.addRow("KK consistent? (RMS≲σ_KK)", cb_kk)
        form.addRow("locked params + reason", ed_lock)
        form.addRow("notes", ed_note)

        cb_all = QCheckBox(
            f"append ALL angles with data  ({len(angs)}: {angs})  — one row each")
        cb_all.setToolTip(
            "If on, log one row per angle with data (per-angle auto metrics); the\n"
            "judgement fields above apply to every row. Use after a multi-angle fit.")
        if len(angs) >= 2:
            form.addRow(cb_all)

        bb = QDialogButtonBox(
            (QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            if PYQT == 6 else (QDialogButtonBox.Ok | QDialogButtonBox.Cancel))
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        accepted = (QDialog.DialogCode.Accepted if PYQT == 6 else QDialog.Accepted)
        if dlg.exec() != accepted:
            return

        # judgement fields (apply to EVERY row written)
        judge = {
            'run': sp_run.value(),
            'm_resolved': cb_m.currentText(),
            'kk_consistent': cb_kk.currentText(),
            'locked_params': ed_lock.text().strip(),
            'notes': ed_note.text().strip(),
        }
        _lbl = ed_label.text().strip()
        if _lbl:
            self.state.sample_label = _lbl   # persists (also in the session via to_dict)
        path = self._campaign_csv_path()
        try:
            if cb_all.isChecked() and len(angs) >= 2:
                # One row per angle with data: switch to each angle so the displayed
                # curves (and gather_auto) reflect it, then restore the original.
                orig = self.state.current_ang
                done = []
                for a in angs:
                    self._select_ang(a)
                    row = ce.gather_auto(self)
                    if _lbl:
                        row['sample'] = _lbl
                    row.update(judge)
                    ce.append_row(path, row)
                    done.append(a)
                self._select_ang(orig)
                self.status.showMessage(
                    f"Campaign: {_lbl or auto['sample']} angles {done} "
                    f"run{judge['run']} → {path}")
            else:
                row = dict(auto)
                if _lbl:
                    row['sample'] = _lbl
                row.update(judge)
                ce.append_row(path, row)
                self.status.showMessage(
                    f"Campaign: {row['sample']} {auto['angle_deg']}° "
                    f"run{judge['run']} → {path}")
        except Exception as e:
            QMessageBox.warning(self, "→ Campaign", f"CSV write error:\n{e}")
            return

    def _campaign_csv_path(self):
        """Campaign log CSV, in the project root (predictable path)."""
        import os as _os
        root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
        return _os.path.join(root, "campaign_log.csv")

    def _build_windows_menu(self):
        """'Windows' menu in the top bar: lists the open d-SCAN windows and brings
        them to the front (useful when many are open or with other applications
        active)."""
        # macOS/Spyder: the native menubar (at the top of the screen) may not appear
        # when the app runs inside another Qt app → force the menubar INSIDE the
        # OPHIRA window.
        self.menuBar().setNativeMenuBar(False)
        # "Windows" is the rightmost menu by convention → add Tools first.
        self._tools_menu = self.menuBar().addMenu("Tools")
        _a = self._tools_menu.addAction("Compare with Scout…")
        _a.triggered.connect(self._on_scout_compare)
        self._win_menu = self.menuBar().addMenu("Windows")
        self._win_menu.aboutToShow.connect(self._rebuild_windows_menu)

    def _rebuild_windows_menu(self):
        self._win_menu.clear()
        dlgs = [d for d in getattr(self, "_open_dialogs", []) if d.isVisible()]
        self._open_dialogs = dlgs   # discard the closed windows
        if not dlgs:
            a = self._win_menu.addAction("(no windows open)")
            a.setEnabled(False)
            return
        for d in dlgs:
            a = self._win_menu.addAction(d.windowTitle())
            a.triggered.connect(
                lambda _=False, dd=d: (dd.show(), dd.raise_(), dd.activateWindow()))
        self._win_menu.addSeparator()
        # "all windows", not "all d-scan windows": _open_dialogs also holds the
        # Scout comparison tool (see _on_scout_compare), and this closes those too.
        a = self._win_menu.addAction("Close all windows")
        a.triggered.connect(self._close_all_windows)

    def _close_all_windows(self):
        for d in list(getattr(self, "_open_dialogs", [])):
            try:
                d.close()
            except Exception:
                pass
        self._open_dialogs = []

    def _show_figure_dialog(self, fig, title, cascade_index=None, hover=False):
        """Show a matplotlib Figure in a non-modal window with a toolbar.

        hover=True adds a readout that, on mouse-over, names the NEAREST labelled
        curve (computed in display/pixel coords, so it works even where curves
        overlap) plus λ and the curve's value there — handy to tell which d-curve
        the cursor is on in the d-scan n(λ) overlay without doing the math.
        """
        try:
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        except ImportError:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        if PYQT == 6:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
        else:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(1000, 560)
        lay = QVBoxLayout(dlg)
        canvas = FigureCanvas(fig)
        _tb = NavigationToolbar2QT(canvas, dlg)
        from windows.widgets import style_nav_toolbar
        style_nav_toolbar(_tb)
        lay.addWidget(_tb)
        # explicit one-click Save PNG / Save PDF (the toolbar floppy can save
        # these too, but is easy to miss).
        if PYQT == 6:
            from PyQt6.QtWidgets import (QPushButton, QHBoxLayout, QFileDialog,
                                         QStyle, QLineEdit, QCheckBox)
        else:
            from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QFileDialog,
                                         QStyle, QLineEdit, QCheckBox)
        from windows.widgets import FigureAspectLock, parse_aspect
        _lock = FigureAspectLock(fig, canvas)
        dlg._aspect_lock = _lock                 # keep it alive with the dialog
        _safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in title).strip("_") or "figure"
        def _save_fig(fmt, _safe=_safe, _dlg=dlg):
            path, _ = QFileDialog.getSaveFileName(
                _dlg, f"Save {fmt.upper()}", f"{_safe}.{fmt}", f"{fmt.upper()} (*.{fmt})")
            if not path:
                return
            if not path.lower().endswith("." + fmt):
                path += "." + fmt
            _lock.savefig(path, dpi=200)         # crops to the aspect page when locked
            self.status.showMessage(f"Saved {path}")
        _sicon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogSaveButton if PYQT == 6 else QStyle.SP_DialogSaveButton)
        _srow = QHBoxLayout()
        for _fmt in ("png", "pdf"):
            _b = QPushButton(f"  {_fmt.upper()}")
            _b.setIcon(_sicon)
            _b.setToolTip(f"Save this figure as {_fmt.upper()}")
            _b.setAutoDefault(False); _b.setDefault(False)   # Enter must NOT trigger Save
            _b.clicked.connect(lambda _c=False, f=_fmt: _save_fig(f))
            _srow.addWidget(_b)
        # aspect ratio + lock-preview: ON → plot-only, letterboxed to the ratio (what you
        # see = what you save, kept on resize); OFF → full figure, free.
        _srow.addSpacing(18)
        _wh = fig.get_size_inches()
        _asp_edit = QLineEdit(f"{_wh[0] / _wh[1]:.2f}")
        _asp_edit.setFixedWidth(56)
        _asp_edit.setToolTip("Export aspect ratio — 'W:H' (e.g. 16:10) or a W/H number (e.g. 1.6).\n"
                             "Set the SAME value on every panel for a composite figure.\n"
                             "Press Enter or Apply to update the preview.")
        _cb_lock = QCheckBox("Lock aspect (preview)")
        _cb_lock.setToolTip("Preview the plot-only export at the chosen aspect: hides the table/"
                            "widgets and keeps the W:H on resize. Save exports exactly this.")
        def _toggle_lock(_checked):
            if _checked:
                _lock.enable(parse_aspect(_asp_edit.text()))
            else:
                _lock.disable()
        _cb_lock.toggled.connect(_toggle_lock)
        def _apply_aspect(*_):
            # Enter / Apply → lock ON at the typed ratio (or re-apply if already on).
            if _cb_lock.isChecked():
                _lock.enable(parse_aspect(_asp_edit.text()))
            else:
                _cb_lock.setChecked(True)          # triggers _toggle_lock → enable
        def _aspect_edited():                      # focus-loss: re-apply ONLY if already locked
            if _lock.active:
                _lock.enable(parse_aspect(_asp_edit.text()))
        _asp_edit.returnPressed.connect(_apply_aspect)
        _asp_edit.editingFinished.connect(_aspect_edited)
        _apply_btn = QPushButton("Apply")
        _apply_btn.setToolTip("Apply the aspect ratio / update the locked preview")
        _apply_btn.setAutoDefault(False); _apply_btn.setDefault(False)
        _apply_btn.clicked.connect(_apply_aspect)
        _srow.addWidget(QLabel("W:H")); _srow.addWidget(_asp_edit)
        _srow.addWidget(_apply_btn); _srow.addWidget(_cb_lock)
        _srow.addStretch()
        lay.addLayout(_srow)
        lay.addWidget(canvas)

        # (re)bind draggable legends to the LIVE canvas: set_draggable at figure-build
        # time binds to the pre-embed canvas, which FigureCanvas(fig) then replaces →
        # the drag is dead. Re-arm here so it works in every window.
        for _lax in fig.axes:
            _leg = _lax.get_legend()
            if _leg is not None:
                try:
                    _leg.set_draggable(False)
                    _leg.set_draggable(True)
                except Exception:
                    pass

        # interactive band sliders (created AFTER the canvas exists, else they get
        # no events) — figures opt in by exposing fig._slider_specs (d-discriminants).
        _specs = getattr(fig, "_slider_specs", None)
        if _specs:
            try:
                from matplotlib.widgets import RangeSlider
                dlg._sliders = []
                for _sax, _vmin, _vmax, _vinit, _label, _cb in _specs:
                    _rs = RangeSlider(_sax, _label, _vmin, _vmax, valinit=_vinit)
                    _rs.on_changed(_cb)
                    dlg._sliders.append(_rs)
            except Exception:
                for _sax, *_ in _specs:      # no RangeSlider (old mpl) → hide the axes
                    _sax.set_visible(False)

        # radio toggles (e.g. abs↔signed KK residual) — same deferred build as the
        # sliders (mpl widgets need the live canvas). Opt in via fig._radio_specs.
        _radios = getattr(fig, "_radio_specs", None)
        if _radios:
            try:
                from matplotlib.widgets import RadioButtons
                dlg._radios = []
                for _rax, _labels, _active, _rcb in _radios:
                    _rb = RadioButtons(_rax, _labels, active=_active)
                    _rb.on_clicked(_rcb)
                    dlg._radios.append(_rb)
            except Exception:
                for _rax, *_ in _radios:
                    try:
                        _rax.set_visible(False)
                    except Exception:
                        pass

        if hover:
            readout = QLabel("hover a curve to identify it")
            readout.setStyleSheet(
                "color:#1A237E; font-size:9pt; font-family:Courier; "
                "background:#F3F4FF; padding:2px 6px; border-radius:3px;")
            lay.addWidget(readout)

            def _on_move(event, _fig=fig, _ro=readout):
                if event.inaxes is None or event.x is None or event.y is None:
                    _ro.setText("hover a curve to identify it"); return
                cx, cy = float(event.x), float(event.y)
                best = None; bestd = 1e18
                for ax in _fig.axes:
                    for ln in ax.lines:
                        lbl = ln.get_label()
                        if not lbl or lbl.startswith('_'):
                            continue
                        xd = np.asarray(ln.get_xdata(), float)
                        yd = np.asarray(ln.get_ydata(), float)
                        if xd.size < 2:
                            continue
                        try:
                            pts = ax.transData.transform(np.column_stack([xd, yd]))
                        except Exception:
                            continue
                        k = int(np.argmin(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
                        d = float(np.hypot(pts[k, 0] - cx, pts[k, 1] - cy))
                        if d < bestd:
                            bestd = d; best = (lbl, xd, yd)
                lam = event.xdata
                if best is not None and bestd < 60.0:
                    lbl, xd, yd = best
                    nval = (float(np.interp(lam, xd, yd))
                            if xd.min() <= lam <= xd.max() else float('nan'))
                    _ro.setText(f"λ = {lam:.0f} nm    |    nearest curve:  {lbl}"
                                f"    n ≈ {nval:.4f}")
                elif lam is not None:
                    _ro.setText(f"λ = {lam:.0f} nm")
            # keep the cid on the dialog so the connection lives as long as it does
            dlg._hover_cid = canvas.mpl_connect('motion_notify_event', _on_move)

        # keep a reference, otherwise the dialog gets garbage-collected
        if not hasattr(self, "_open_dialogs"):
            self._open_dialogs = []
        # explicit cascade_index (reset on every d-SCAN call) → the windows do NOT
        # migrate off-screen on repeated calls.
        n = cascade_index if cascade_index is not None else len(self._open_dialogs)
        self._open_dialogs.append(dlg)
        dlg.show()
        # cascade: otherwise the windows open OVERLAPPED and only one is visible
        # (the other is behind). I offset them and bring them to the front.
        dlg.move(self.x() + 50 + 70 * n, self.y() + 80 + 70 * n)
        dlg.raise_(); dlg.activateWindow()

    def _on_scout_compare(self):
        """Open the read-only Ophira-vs-Scout comparison tool (Tools menu)."""
        from windows.scout_compare_dialog import ScoutCompareDialog
        dlg = ScoutCompareDialog(self)
        if not hasattr(self, "_open_dialogs"):
            self._open_dialogs = []
        self._open_dialogs.append(dlg)
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _on_dscan_analysis(self):
        """d-SCAN: select N sessions (curated fits at different d) → n(λ) overlay
        + χ²(d) with parabola/bracket + parameter table. READ ONLY (direct
        unpickle, doesn't touch the current state)."""
        from windows import dscan_analysis as dsa
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select d-scan sessions (≥2)", "", "Session (*.psi_sess)")
        if not paths:
            return
        if len(paths) < 2:
            self.status.showMessage("d-SCAN: select at least 2 sessions")
            return
        try:
            sessions = [dsa.read_session(p) for p in paths]
        except Exception as e:
            self.status.showMessage(f"d-SCAN: read error ({e})")
            return
        refl = dsa.read_reflectance_set(paths)
        try:
            fig_n = dsa.make_n_overlay_figure(sessions)
            fig_ct, dC0, dR0 = dsa.make_contrast_figure(sessions, refl)
            # DRAWER: legacy χ²(d)/KK-resid(d) figure (see dsa.LEGACY_DSCAN_DRAWER)
            fig_c, d_opt = (dsa.make_chi2_figure(sessions)
                            if dsa.LEGACY_DSCAN_DRAWER else (None, None))
        except Exception as e:
            self.status.showMessage(f"d-SCAN: figure error ({e})")
            return
        # AUTOMATIC save into _paper_figures/ (besides the toolbar floppy).
        import os as _os
        # common name WITHOUT the d suffix (e.g. sample_1300 → sample)
        names = [_os.path.splitext(_os.path.basename(p))[0].rsplit("_", 1)[0] for p in paths]
        prefix = _os.path.commonprefix(names).rstrip("_- ") or "dscan"
        self._dscan_count = getattr(self, "_dscan_count", 0) + 1
        cnt = self._dscan_count
        outdir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "_paper_figures")
        saved = ""
        try:
            _os.makedirs(outdir, exist_ok=True)
            _to_save = [(fig_n, f"{prefix}_dscan_n"),
                        (fig_ct, f"{prefix}_dscan_discriminants")]
            if fig_c is not None:                       # DRAWER (legacy χ²/KK)
                _to_save.append((fig_c, f"{prefix}_dscan_chi2"))
            for fg, b in _to_save:
                p = _os.path.join(outdir, b)
                fg.savefig(p + ".png", dpi=160); fg.savefig(p + ".pdf")
            saved = f"  — saved in _paper_figures/{prefix}_dscan_*.png/.pdf"
        except Exception:
            pass
        self._show_figure_dialog(fig_n, f"#{cnt} {prefix} — n(λ) overlay", 0, hover=True)
        self._show_figure_dialog(fig_ct, f"#{cnt} {prefix} — d-discriminants", 1)
        if fig_c is not None:                            # DRAWER (χ²/KK, gated by flag)
            self._show_figure_dialog(fig_c, f"#{cnt} {prefix} — RMS residual(d) / KK residual", 2)
        # Third window: exp vs sim reflectance overlay (interactive: toggle + zoom)
        if refl is not None:
            try:
                from windows.refl_overlay_dialog import ReflectanceOverlayDialog
                ang, wl, R, sims = refl
                rdlg = ReflectanceOverlayDialog(self, ang, wl, R, sims)
                rdlg.setWindowTitle(f"#{cnt} {prefix} — reflectance ({ang}°)")
                self._open_dialogs.append(rdlg)
                rdlg.show()
                rdlg.move(self.x() + 50 + 70 * 3, self.y() + 80 + 70 * 3)
                rdlg.raise_(); rdlg.activateWindow()
            except Exception as e:
                self.status.showMessage(f"d-SCAN: reflectance window not opened ({e})")
        msg = f"d-SCAN: {len(sessions)} sessions"
        xs = [x for x in (dC0, dR0) if x is not None]
        if xs:
            msg += f"  →  d ≈ {np.mean(xs):.0f} nm  (contrast/500nm discriminants)"
        elif d_opt:
            msg += f"  →  d ≈ {d_opt:.0f} nm (χ²)"
        self.status.showMessage(msg + saved)

    def _on_about(self):
        """About box: shows version, author, license, Zenodo DOI.
        To be kept aligned with `version.py` — single source of truth.
        """
        from version import about_html, version_string
        # QMessageBox.about() honors the clickable links (mailto:, http:) if the
        # text is RichText (auto-detect on <h2>, <p>, ...).
        QMessageBox.about(self, f"About {version_string()}", about_html())

    def _on_export_figure(self):
        """Export the main_window figure (R + n/k) for the paper.
        Formats: vector PDF (default), PNG 600 DPI, vector SVG.
        """
        from .export_fig import export_figure
        # Base name: use the file name loaded for the current angle if available,
        # otherwise 'sample'
        ang = self.state.current_ang
        fname = self.state.ang_filenames.get(ang)
        if fname:
            import os
            base = os.path.splitext(os.path.basename(fname))[0]
        else:
            base = "sample"
        default = f"{base}_main_{ang}deg"
        path = export_figure(
            self.fig, parent=self, default_name=default,
            title="Export main figure (R + n/k)",
            metadata={"Title": f"OPHIRA main — {base} {ang}°",
                      "Creator": "OPHIRA"})
        if path:
            self.status.showMessage(f"Figure exported: {path}")

    def _default_ext_override(self):
        """Default full fit curve as an ext_override tuple: n,k at the 40 ext nodes
        (14 base + 2 sub-nodes/gap); the wavelengths are rebuilt from the base nodes.
        Returns None on any size mismatch → falls back to the 14-node curve."""
        try:
            ext_nd = ph._build_ext_nodes(ph.build_c_nodes(self.state))
            n = np.asarray(EXT_N, float); k = np.asarray(EXT_K, float)
            if len(ext_nd) != len(n):
                return None
            return (ext_nd, n, k, {})
        except Exception:
            return None

    def _on_reset(self):
        reply = QMessageBox.question(
            self, "Reset", "Reset all parameters to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            if PYQT == 6 else
            QMessageBox.Yes | QMessageBox.No)
        yes = (QMessageBox.StandardButton.Yes if PYQT == 6 else QMessageBox.Yes)
        if reply == yes:
            self.engine.reset()
            # engine.reset() rebuilds a bare AppState with FLAT n/k defaults and
            # generic scalar defaults (d=1000, ...). Restore the realistic first-launch
            # curve — the scalar params AND the n/k nodes — so RESET matches the startup
            # curve, not the bare defaults.
            for _k, _v in PARAM_INIT.items():
                setattr(self.state, _k, _v)
            for i in range(14):
                self.state.n_slider_vals[i] = ph.inv_n_map(float(N_INIT[i]))
                self.state.k_slider_vals[i] = ph.inv_k_map(float(K_INIT[i]))
            self._update_sliders_from_state()
            # The controls have to follow the state back to the start: the toggles
            # (MONO n/k, KK, PRED 8°, THIN, AID) and the d-source combo carry their
            # value in their own label and colour, and the material combos are back
            # to Si/PSi. Without this they keep announcing the pre-reset state while
            # the engine already uses the new one.
            self._update_all_toggles()
            self._sync_material_combos()
            self.state.ext_override = self._default_ext_override()
            self._refresh_simulation()      # also syncs physics' material singletons
            self.status.showMessage("Reset done")
