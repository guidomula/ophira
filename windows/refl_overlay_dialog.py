"""
refl_overlay_dialog.py — d-scan reflectance overlay window: experimental vs
simulated reflectance, per-curve visibility toggles, and manual zoom controls
(x/y + reset). Used to inspect fit quality in the region where the fringes damp
out (absorption edge), where the thickness d is most discriminable.
"""
import numpy as np

try:
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
                                 QLineEdit, QPushButton, QLabel)
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT)
except ImportError:  # PyQt5
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
                                 QLineEdit, QPushButton, QLabel)
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT)
from matplotlib.figure import Figure
import matplotlib.cm as cm


class ReflectanceOverlayDialog(QDialog):
    """exp + N simulated curves (one per d) with toggles and manual zoom."""

    def __init__(self, parent, angle, wl_exp, R_exp, sims):
        super().__init__(parent)
        self.setWindowTitle(f"d-scan — reflectance exp vs sim ({angle}°)")
        self.resize(1020, 640)
        self.wl = np.asarray(wl_exp, float)
        self.R = np.asarray(R_exp, float)
        self.sims = sims
        self._cols = cm.coolwarm(np.linspace(0, 1, max(len(sims), 1)))

        lay = QVBoxLayout(self)

        # ── visibility toggle row ──
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Show:"))
        self.cb_exp = QCheckBox("exp"); self.cb_exp.setChecked(True)
        self.cb_exp.toggled.connect(self._redraw); trow.addWidget(self.cb_exp)
        self.cbs = []
        for (d, _), c in zip(sims, self._cols):
            cb = QCheckBox(f"d={d:.0f}"); cb.setChecked(True)
            cb.setStyleSheet(f"color: rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)});")
            cb.toggled.connect(self._redraw)
            trow.addWidget(cb); self.cbs.append(cb)
        trow.addStretch()
        lay.addLayout(trow)

        # ── canvas + toolbar ──
        self.fig = Figure(figsize=(9, 4.6))
        self.ax = self.fig.add_subplot(111)
        self.fig._plot_axes = [self.ax]          # for the aspect-lock export/preview
        self.canvas = FigureCanvas(self.fig)
        _tb = NavigationToolbar2QT(self.canvas, self)
        try:
            from windows.widgets import style_nav_toolbar
            style_nav_toolbar(_tb)
        except Exception:
            pass
        lay.addWidget(_tb)

        # ── save row: PNG/PDF + aspect-lock preview ──
        from windows.widgets import FigureAspectLock, parse_aspect
        self._aspect_lock = FigureAspectLock(self.fig, self.canvas)
        _srow = QHBoxLayout()
        def _save(fmt):
            try:
                from PyQt6.QtWidgets import QFileDialog
            except ImportError:
                from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self, f"Save {fmt.upper()}", f"reflectance_{angle}deg.{fmt}",
                f"{fmt.upper()} (*.{fmt})")
            if not path:
                return
            if not path.lower().endswith("." + fmt):
                path += "." + fmt
            self._aspect_lock.savefig(path, dpi=200)
        for _fmt in ("png", "pdf"):
            _b = QPushButton(_fmt.upper()); _b.setToolTip(f"Save as {_fmt.upper()}")
            _b.setAutoDefault(False); _b.setDefault(False)   # Enter must NOT trigger Save
            _b.clicked.connect(lambda _c=False, f=_fmt: _save(f))
            _srow.addWidget(_b)
        _srow.addSpacing(18)
        _wh = self.fig.get_size_inches()
        self._asp_edit = QLineEdit(f"{_wh[0] / _wh[1]:.2f}"); self._asp_edit.setFixedWidth(56)
        self._asp_edit.setToolTip("Export aspect W:H (e.g. 16:10) or a W/H number (e.g. 1.6).\n"
                                  "Press Enter or Apply to update the preview.")
        _cb = QCheckBox("Lock aspect (preview)")
        def _toggle_lock(chk):
            if chk:
                self._aspect_lock.enable(parse_aspect(self._asp_edit.text()))
            else:
                self._aspect_lock.disable()
        _cb.toggled.connect(_toggle_lock)
        def _apply_aspect(*_):                     # Enter / Apply → lock ON at the typed ratio
            if _cb.isChecked():
                self._aspect_lock.enable(parse_aspect(self._asp_edit.text()))
            else:
                _cb.setChecked(True)
        def _aspect_edited():                      # focus-loss: re-apply only if already locked
            if self._aspect_lock.active:
                self._aspect_lock.enable(parse_aspect(self._asp_edit.text()))
        self._asp_edit.returnPressed.connect(_apply_aspect)
        self._asp_edit.editingFinished.connect(_aspect_edited)
        _apply_btn = QPushButton("Apply"); _apply_btn.setToolTip("Apply the aspect / update preview")
        _apply_btn.setAutoDefault(False); _apply_btn.setDefault(False)
        _apply_btn.clicked.connect(_apply_aspect)
        _srow.addWidget(QLabel("W:H")); _srow.addWidget(self._asp_edit)
        _srow.addWidget(_apply_btn); _srow.addWidget(_cb)
        _srow.addStretch()
        lay.addLayout(_srow)
        lay.addWidget(self.canvas)

        # ── manual zoom row ──
        zrow = QHBoxLayout()
        self.e_xmin = QLineEdit(); self.e_xmax = QLineEdit()
        self.e_ymin = QLineEdit(); self.e_ymax = QLineEdit()
        for lab, e in (("λ min", self.e_xmin), ("λ max", self.e_xmax),
                       ("R min", self.e_ymin), ("R max", self.e_ymax)):
            zrow.addWidget(QLabel(lab)); e.setFixedWidth(64)
            e.returnPressed.connect(self._apply_limits); zrow.addWidget(e)
        b_apply = QPushButton("Apply zoom"); b_apply.clicked.connect(self._apply_limits)
        b_reset = QPushButton("Reset (full)"); b_reset.clicked.connect(self._reset)
        zrow.addWidget(b_apply); zrow.addWidget(b_reset); zrow.addStretch()
        lay.addLayout(zrow)

        self._redraw()
        self._reset()

    def _redraw(self):
        # preserve the current limits (toggling must not reset the zoom)
        xl = self.ax.get_xlim() if self.ax.lines else None
        yl = self.ax.get_ylim() if self.ax.lines else None
        self.ax.clear()
        if self.cb_exp.isChecked():
            self.ax.plot(self.wl, self.R, "k-", lw=2.2, label="exp", zorder=10)
        for (d, fit), cb, c in zip(self.sims, self.cbs, self._cols):
            if cb.isChecked() and fit.shape == self.wl.shape:
                self.ax.plot(self.wl, fit, color=c, lw=1.3, label=f"d={d:.0f}")
        self.ax.set_xlabel("λ (nm)"); self.ax.set_ylabel("Reflectance (%)")
        self.ax.grid(alpha=.25)
        _leg = self.ax.legend(fontsize=8, loc="upper left")
        try:
            _leg.set_draggable(True)     # draggable off overlapping curves (live canvas)
        except Exception:
            pass
        if xl is not None:
            self.ax.set_xlim(xl); self.ax.set_ylim(yl)
        self.canvas.draw_idle()

    def _apply_limits(self, *args):
        def val(e):
            try:
                return float(e.text())
            except ValueError:
                return None
        xmin, xmax = val(self.e_xmin), val(self.e_xmax)
        ymin, ymax = val(self.e_ymin), val(self.e_ymax)
        if xmin is not None and xmax is not None and xmax > xmin:
            self.ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None and ymax > ymin:
            self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()

    def _reset(self, *args):
        self.ax.relim(); self.ax.autoscale_view()
        self.ax.set_xlim(float(self.wl.min()), float(self.wl.max()))
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self.e_xmin.setText(f"{xl[0]:.0f}"); self.e_xmax.setText(f"{xl[1]:.0f}")
        self.e_ymin.setText(f"{yl[0]:.1f}"); self.e_ymax.setText(f"{yl[1]:.1f}")
        self.canvas.draw_idle()
