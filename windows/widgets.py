"""
widgets.py — Reusable OPHIRA widgets

MappedSlider: slider with non-linear mapping + lock button.
SliderGroup:  group of MappedSlider with colored background and title.
"""

try:
    from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                                  QLabel, QSlider, QPushButton, QGroupBox,
                                  QSizePolicy, QScrollArea, QLineEdit)
    from PyQt6.QtCore import Qt, pyqtSignal, QSize
    from PyQt6.QtGui import QFont, QColor, QPixmap, QPainter, QIcon
    PYQT = 6
except ImportError:
    from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                                  QLabel, QSlider, QPushButton, QGroupBox,
                                  QSizePolicy, QScrollArea, QLineEdit)
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QIcon
    PYQT = 5

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib toolbar icon contrast
# ─────────────────────────────────────────────────────────────────────────────
def style_nav_toolbar(toolbar, icon_color=None):
    """Recolor a matplotlib NavigationToolbar's icons for high contrast.

    matplotlib's toolbar glyphs are dark grey and can be nearly invisible
    depending on the Qt/OS theme. This tints every icon to a colour chosen for
    contrast with the toolbar's own background — white on a dark theme, near-black
    on a light one (pass icon_color to force a colour). Cosmetic and best-effort:
    it never raises, so it cannot break toolbar creation.
    """
    try:
        bg = toolbar.palette().color(toolbar.backgroundRole())
        auto = "#f5f5f5" if bg.lightness() < 128 else "#202020"
    except Exception:
        auto = "#202020"
    col = QColor(icon_color or auto)
    try:                                    # PyQt6 scoped enum
        srcin = QPainter.CompositionMode.CompositionMode_SourceIn
    except AttributeError:                  # PyQt5
        srcin = QPainter.CompositionMode_SourceIn
    for act in toolbar.actions():
        try:
            ic = act.icon()
            if ic is None or ic.isNull():
                continue
            sz = ic.actualSize(QSize(24, 24))
            if sz.width() <= 0 or sz.height() <= 0:
                sz = QSize(24, 24)
            pm = ic.pixmap(sz)
            if pm.isNull():
                continue
            tinted = QPixmap(pm.size())
            tinted.fill(QColor(0, 0, 0, 0))     # transparent, version-independent
            p = QPainter(tinted)
            p.drawPixmap(0, 0, pm)
            p.setCompositionMode(srcin)
            p.fillRect(tinted.rect(), col)
            p.end()
            act.setIcon(QIcon(tinted))
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Aspect-lock preview / plot-only export for matplotlib figures
# ─────────────────────────────────────────────────────────────────────────────
def parse_aspect(text, default=1.6):
    """Parse an aspect ratio from 'W:H' (e.g. '16:10') or a plain W/H number
    (e.g. '1.6'). Returns a positive float, or `default` on any problem."""
    try:
        t = str(text).strip().replace(",", ".")
        if ":" in t:
            w, h = t.split(":", 1)
            a = float(w) / float(h)
        else:
            a = float(t)
        return a if a > 0.05 else default
    except Exception:
        return default


class FigureAspectLock:
    """Constrain a matplotlib figure to a target width/height on screen so the
    preview matches a plot-only export.

    When enabled, the figure's PLOT axes (declared as ``fig._plot_axes``; falls
    back to all axes) are letterboxed into a centred 'page' of the target aspect
    within whatever the canvas size is, and the DECORATION axes (parameter tables,
    slider/radio widget axes — everything not in ``fig._plot_axes``) are hidden.
    A dashed rectangle marks the page. On resize the page is recomputed so the
    aspect is kept (true WYSIWYG). ``savefig`` crops to the page → the file is
    exactly the previewed plot at the target aspect, with crisp (un-stretched)
    text. Everything is restored on ``disable``. matplotlib-only (no Qt)."""

    # plot-axes rect WITHIN the page (margins for y-labels left/right, x-label, title)
    PLOT_RECT = (0.145, 0.14, 0.70, 0.76)

    def __init__(self, fig, canvas):
        self.fig = fig
        self.canvas = canvas
        self.aspect = 1.6
        self.active = False
        self._cid = None
        self._page_patch = None
        self._saved = None          # [(axes, position.bounds, visible), ...]

    def _plot_axes(self):
        pa = getattr(self.fig, "_plot_axes", None)
        return list(pa) if pa else list(self.fig.axes)

    def _page(self):
        """Centred rectangle of the target aspect within the canvas (figure fractions)."""
        W, H = self.canvas.get_width_height()
        if W <= 0 or H <= 0:
            return (0.03, 0.03, 0.94, 0.94)
        r = self.aspect / (W / H)       # page_w_frac / page_h_frac
        if r >= 1.0:                    # page relatively wider → width-limited
            pw = 0.94; ph = pw / r
        else:                           # taller → height-limited
            ph = 0.94; pw = ph * r
        return ((1 - pw) / 2, (1 - ph) / 2, pw, ph)

    def _layout(self, *args):
        px, py, pw, ph = self._page()
        mx, my, mw, mh = self.PLOT_RECT
        rect = (px + mx * pw, py + my * ph, mw * pw, mh * ph)
        for a in self._plot_axes():
            a.set_position(rect)
        if self._page_patch is None:
            from matplotlib.patches import Rectangle
            self._page_patch = Rectangle((px, py), pw, ph, fill=False, ec="#9e9e9e",
                                         lw=0.8, ls="--", transform=self.fig.transFigure,
                                         zorder=0)
            self.fig.add_artist(self._page_patch)
        else:
            self._page_patch.set_bounds(px, py, pw, ph)
        self.canvas.draw_idle()

    def enable(self, aspect):
        self.aspect = float(aspect)
        if not self.active:
            self._saved = [(a, a.get_position().bounds, a.get_visible())
                           for a in self.fig.axes]
            keep = {id(a) for a in self._plot_axes()}
            for a in self.fig.axes:
                if id(a) not in keep:
                    a.set_visible(False)
            self._cid = self.canvas.mpl_connect("resize_event", self._layout)
            self.active = True
        self._layout()

    def disable(self):
        if not self.active:
            return
        if self._cid is not None:
            self.canvas.mpl_disconnect(self._cid); self._cid = None
        if self._page_patch is not None:
            try:
                self._page_patch.remove()
            except Exception:
                pass
            self._page_patch = None
        for a, bounds, vis in (self._saved or []):
            a.set_position(bounds); a.set_visible(vis)
        self._saved = None
        self.active = False
        self.canvas.draw_idle()

    def savefig(self, path, dpi=200):
        """Save the figure. When locked, crop to the page (aspect, plot-only)."""
        if not self.active:
            self.fig.savefig(path, dpi=dpi)
            return
        from matplotlib.transforms import Bbox
        px, py, pw, ph = self._page()
        w_in, h_in = self.fig.get_size_inches()
        bb = Bbox([[px * w_in, py * h_in], [(px + pw) * w_in, (py + ph) * h_in]])
        patch_vis = self._page_patch.get_visible() if self._page_patch is not None else None
        if self._page_patch is not None:
            self._page_patch.set_visible(False)
        try:
            self.fig.savefig(path, dpi=dpi, bbox_inches=bb)
        finally:
            if self._page_patch is not None and patch_vis is not None:
                self._page_patch.set_visible(patch_vis)
                self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# MappedSlider
# ─────────────────────────────────────────────────────────────────────────────

class MappedSlider(QWidget):
    """Horizontal slider with:
      - non-linear mapping (to_display for the shown value)
      - name label + value label
      - lock button (green=free, red=locked)
      - value_changed(float) signal emits the INTERNAL value (pre-mapping)

    The INTERNAL value matches what AppState stores:
      - n_slider_vals[i] ∈ [0.0, 6.5]  →  n_physical = 1 + x³/80
      - k_slider_vals[i] ∈ [-4.0, 1.0]  →  k_physical = 10^x
      - all others: internal = physical (identity mapping)
    """

    value_changed = pyqtSignal(float)   # internal value
    lock_changed  = pyqtSignal(bool)    # True = locked

    STEPS = 10000  # internal QSlider resolution

    def __init__(self, label: str,
                 vmin: float, vmax: float,
                 initial: float,
                 to_display=None,   # internal → value to display (default: identity)
                 fmt: str = "{:.3f}",
                 label_width: int = 80,
                 slider_style: str = None,
                 parent=None):
        super().__init__(parent)
        self._vmin = vmin
        self._vmax = vmax
        self._to_display = to_display or (lambda x: x)
        self._fmt = fmt
        self._locked = False
        self._suppress = False   # avoid recursion during set_value

        # Horizontal layout: [lock] [label] [slider] [value]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(1, 0, 1, 0)
        layout.setSpacing(2)

        # Lock button
        self._btn_lock = QPushButton("●")
        self._btn_lock.setFixedSize(14, 14)
        self._btn_lock.setCheckable(True)
        self._btn_lock.setToolTip("Lock/unlock parameter")
        self._btn_lock.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; "
            "border: none; border-radius: 7px; font-size: 7px; }"
            "QPushButton:checked { background: #F44336; }")
        self._btn_lock.toggled.connect(self._on_lock_toggled)
        layout.addWidget(self._btn_lock)

        # Name label
        self._lbl = QLabel(label)
        self._lbl.setFixedWidth(label_width)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignRight |
                          Qt.AlignmentFlag.AlignVCenter
                          if PYQT == 6 else
                          Qt.AlignRight | Qt.AlignVCenter)
        f = QFont(); f.setPointSize(10); f.setBold(False)
        self._lbl.setFont(f)
        self._lbl.setStyleSheet("color: black; background: transparent;")
        layout.addWidget(self._lbl)

        # QSlider
        self._slider = QSlider(Qt.Orientation.Horizontal
                               if PYQT == 6 else Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self.STEPS)
        self._slider.setValue(self._to_steps(initial))
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding
                                   if PYQT == 6 else QSizePolicy.Expanding,
                                   QSizePolicy.Policy.Fixed
                                   if PYQT == 6 else QSizePolicy.Fixed)
        self._slider.setFixedHeight(16)
        if slider_style:
            self._slider.setStyleSheet(slider_style)
        self._slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self._slider)

        # Editable value box (input + display)
        self._val_label = QLineEdit(self._fmt.format(self._to_display(initial)))
        self._val_label.setFixedWidth(68)
        f2 = QFont(); f2.setPointSize(10); f2.setFamily("Courier")
        self._val_label.setFont(f2)
        self._val_label.setStyleSheet(
            "color: black; background: #FFFDE7; border: 1px solid #CCCCCC; "
            "border-radius: 2px; padding: 0px 2px;")
        self._val_label.editingFinished.connect(self._on_value_typed)
        layout.addWidget(self._val_label)

        self.setFixedHeight(21)   # compact slider rows

    # ── Conversions ─────────────────────────────────────────────────────────────

    def _to_steps(self, internal: float) -> int:
        """Internal value → QSlider integer."""
        if self._vmax == self._vmin: return 0
        t = (internal - self._vmin) / (self._vmax - self._vmin)
        return int(np.clip(t, 0.0, 1.0) * self.STEPS)

    def _to_internal(self, steps: int) -> float:
        """QSlider integer → internal value."""
        return self._vmin + (steps / self.STEPS) * (self._vmax - self._vmin)

    # ── Current value ───────────────────────────────────────────────────────────

    def get_value(self) -> float:
        """Current internal value (what AppState stores)."""
        return self._to_internal(self._slider.value())

    def set_value(self, internal: float):
        """Set the value externally (e.g. after a fit) without emitting a signal."""
        self._suppress = True
        self._slider.setValue(self._to_steps(internal))
        self._val_label.setText(self._fmt.format(
            self._to_display(internal)))
        self._suppress = False

    def set_label(self, text: str):
        """Update the name-label text (e.g. nodes W300/W335/W375)."""
        self._lbl.setText(text)

    def set_display_value(self, display_val: float):
        """Update only the value box without touching the slider."""
        self._val_label.setText(self._fmt.format(display_val))

    def _on_value_typed(self):
        """A value was typed in the box — convert to internal and move the slider."""
        # Act only on a genuine edit. editingFinished also fires on focus loss
        # (e.g. opening a dialog): without this guard a spurious recommit would
        # emit value_changed → clear ext_override → degrade the curve as if a
        # slider had moved. Programmatic setText() resets isModified to False → no-op.
        if not self._val_label.isModified():
            return
        try:
            # The text is in DISPLAY scale — to_display must be inverted
            text = self._val_label.text().strip()
            display_val = float(text)
            # Approximate inversion: find x ∈ [vmin, vmax] such that to_display(x) ≈ display_val.
            # For a monotonic mapping (n_map, k_map, identity) a linear search suffices.
            xs = np.linspace(self._vmin, self._vmax, 5000)
            ys = np.array([self._to_display(x) for x in xs])
            idx = int(np.argmin(np.abs(ys - display_val)))
            internal = float(xs[idx])
            self._suppress = True
            self._slider.setValue(self._to_steps(internal))
            self._val_label.setText(self._fmt.format(self._to_display(internal)))
            self._suppress = False
            self.value_changed.emit(internal)
        except (ValueError, Exception):
            # Invalid value — restore the current value
            self._val_label.setText(self._fmt.format(
                self._to_display(self.get_value())))

    # ── Lock ──────────────────────────────────────────────────────────────────

    def is_locked(self) -> bool:
        return self._locked

    def set_locked(self, locked: bool):
        self._suppress = True
        self._btn_lock.setChecked(locked)
        self._locked = locked
        # The slider always stays movable; the lock only acts on the fit bounds
        # (handled by AppState).
        self._slider.setStyleSheet(
            (self._slider.styleSheet() or "") )
        self._suppress = False

    def _on_lock_toggled(self, checked: bool):
        self._locked = checked
        # Does NOT disable the slider — it stays movable at all times
        if not self._suppress:
            self.lock_changed.emit(checked)

    # ── Slider slot ─────────────────────────────────────────────────────────────

    def _on_slider_moved(self, steps: int):
        internal = self._to_internal(steps)
        self._val_label.setText(self._fmt.format(
            self._to_display(internal)))
        if not self._suppress:
            self.value_changed.emit(internal)


# ─────────────────────────────────────────────────────────────────────────────
# SliderGroup — group with colored background
# ─────────────────────────────────────────────────────────────────────────────

class SliderGroup(QGroupBox):
    """QGroupBox with colored background that holds MappedSlider widgets in a column."""

    def __init__(self, title: str, bg_color: str,
                 border_color: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(
            f"QGroupBox {{ background-color: {bg_color}; "
            f"border: 1px solid {border_color}; border-radius: 3px; "
            f"margin-top: 10px; padding: 0px; font-size: 9pt; font-weight: bold; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 4px; "
            f"padding: 0 2px; color: {border_color}; }}")
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(2, 10, 2, 2)
        self._sliders: list[MappedSlider] = []

    def add_slider(self, slider: MappedSlider) -> MappedSlider:
        self._layout.addWidget(slider)
        self._sliders.append(slider)
        return slider

    def sliders(self) -> list:
        return self._sliders
