"""
peak_window.py — OPHIRA Peak Analysis window
"""
try:
    from PyQt6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
        QPushButton,QLabel,QLineEdit,QGroupBox,QGridLayout,QFileDialog,
        QSizePolicy,QStatusBar,QScrollArea,QFrame,QRadioButton,QButtonGroup)
    from PyQt6.QtCore import Qt; PYQT=6
except ImportError:
    from PyQt5.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
        QPushButton,QLabel,QLineEdit,QGroupBox,QGridLayout,QFileDialog,
        QSizePolicy,QStatusBar,QScrollArea,QFrame,QRadioButton,QButtonGroup)
    from PyQt5.QtCore import Qt; PYQT=5

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigureCanvas,
                                                   NavigationToolbar2QT as NavToolbar)
except ImportError:
    from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                    NavigationToolbar2QT as NavToolbar)
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from app_state import AppState
from fit_engine import FitEngine
import physics as ph


class PeakWindow(QMainWindow):
    """Peak Analysis window."""

    COLORS = {8:'green',20:'red',40:'purple',60:'orange'}

    def __init__(self,state:AppState,engine:FitEngine,parent=None):
        super().__init__(parent)
        self.state=state; self.engine=engine
        self.setWindowTitle("OPHIRA — Peak Analysis")
        # Initial size compatible with a MacBook 14"/13" (1440×900 → ~800px usable).
        # Moderate width; the right panel has a ScrollArea so it doesn't overflow.
        self.resize(1200, 720)
        self.setMinimumSize(900, 500)

        # Internal window state
        self.vis_exp={8:True,20:True,40:True,60:True}
        self.vis_fit={8:True,20:True,40:True,60:True}
        self.vis_n  ={8:True,20:True,40:True,60:True}
        # n_fit: fitted n(λ) curve (from the per-angle ext_override)
        # overlaid on the n_maxima markers, for the direct comparison of the two methods.
        self.vis_n_fit={8:True,20:True,40:True,60:True}
        self.vis_lbl={8:True,20:True,40:True,60:True}   # per-angle maxima labels (the "tag" column)
        self.local_n={8:None,20:None,40:None,60:None}
        self.use_fit=[True]
        self._d_mem=None   # thickness computed from peaks
        self._maxima_src=None   # 'fit'/'exp' source of the extracted peaks = the n_maxima source (shown)

        self._build_ui()
        self._refresh()

    def _build_ui(self):
        # NB: no QScrollArea here: an auto-resizing scroll would make the
        # plots jump as the canvas redraws on hover.
        central=QWidget(); self.setCentralWidget(central)
        main=QHBoxLayout(central); main.setSpacing(6)

        # ── Plot ─────────────────────────────────────────────────────────────
        self.fig=Figure(figsize=(9,6))
        self.canvas=FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding if PYQT==6 else QSizePolicy.Expanding,
                                  QSizePolicy.Policy.Expanding if PYQT==6 else QSizePolicy.Expanding)
        self.ax=self.fig.add_subplot(111)
        self.ax_n=self.ax.twinx()
        self.ax.set_xlabel("Wavelength (nm)", fontsize=12); self.ax.set_ylabel("Reflectance (%)", color='blue', fontsize=12)
        self.ax_n.set_ylabel("Refractive index, n", color='darkgreen', fontsize=12)
        # twinx puts the n TICKS on the right but the label landed on the left → force it right.
        self.ax_n.yaxis.set_label_position("right")
        self.ax.grid(True,alpha=0.2); self.ax.set_xlim(450,2600); self.ax.set_ylim(0,65)
        # matplotlib toolbar (zoom / pan / Customize) above the canvas, for homogeneity with
        # the other windows. Same trim as the main window: drop Subplots/Save, keep Customize.
        plot_col=QWidget(); plot_v=QVBoxLayout(plot_col)
        plot_v.setContentsMargins(0,0,0,0); plot_v.setSpacing(2)
        self.toolbar=NavToolbar(self.canvas, plot_col)
        self.toolbar.setMaximumHeight(28)
        try:
            from windows.widgets import style_nav_toolbar
            style_nav_toolbar(self.toolbar)
        except Exception:
            pass
        for _action in self.toolbar.actions():
            if _action.text() in ('Subplots','Save'):
                self.toolbar.removeAction(_action)
        plot_v.addWidget(self.toolbar); plot_v.addWidget(self.canvas)
        main.addWidget(plot_col,stretch=4)

        # ── Right panel (in a QScrollArea for small monitors)
        # The panel can exceed the available height on a 13"/14" laptop, so it
        # uses a ScrollArea with fixed width and vertically scrollable content.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame if PYQT == 6 else QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff if PYQT == 6 else Qt.ScrollBarAlwaysOff)
        # Right panel width: wide enough to keep the Calculation buttons from overlapping.
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(400)
        main.addWidget(scroll, stretch=1)
        right_widget = QWidget()
        scroll.setWidget(right_widget)
        right = QVBoxLayout(right_widget)
        right.setSpacing(4)
        right.setContentsMargins(4, 4, 4, 4)

        # Compact style shared by the QGroupBox to reduce vertical overhead
        _GB_STYLE = ("QGroupBox{margin-top:8px;padding:6px;font-size:9pt;font-weight:bold;}"
                     "QGroupBox::title{subcontrol-origin:margin;left:6px;padding:0 3px;}")

        # Toggle matrix (Visibility) — smaller buttons, less padding
        grp = QGroupBox("Visibility"); grp.setStyleSheet(_GB_STYLE)
        grid = QGridLayout(grp); grid.setSpacing(2); grid.setContentsMargins(4, 4, 4, 4)
        grid.addWidget(QLabel(""), 0, 0)
        for j, col in enumerate(["exp", "fit", "n", "n_fit", "tag"]):
            lbl = QLabel(col)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT == 6 else Qt.AlignCenter)
            grid.addWidget(lbl, 0, j+1)
        self._toggle_btns = {}
        for i, ang in enumerate([8, 20, 40, 60]):
            lbl = QLabel(f"{ang}°"); grid.addWidget(lbl, i+1, 0)
            for j, (diz, key) in enumerate([(self.vis_exp, "exp"),
                                            (self.vis_fit, "fit"),
                                            (self.vis_n, "n"),
                                            (self.vis_n_fit, "n_fit"),
                                            (self.vis_lbl, "lbl")]):
                b = QPushButton("●"); b.setFixedSize(26, 20)
                b.setCheckable(True); b.setChecked(True)
                b.setStyleSheet("QPushButton{background:#90ee90;border:none;border-radius:3px}"
                                "QPushButton:!checked{background:#d0d0d0}")
                b.toggled.connect(self._make_toggle(diz, ang, key, b))
                grid.addWidget(b, i+1, j+1); self._toggle_btns[(ang, key)] = b
        right.addWidget(grp)

        # Mouse mode toggle: switch between
        # HAIRLINE (cross-hair that follows the mouse to read R/n) and
        # LABELS (draggable m+λ labels). The hairline's default `motion_notify`
        # can interfere with the label-drag motion handler — this toggle disables one
        # system while the other is active.
        g_mouse = QGroupBox("Mouse mode"); g_mouse.setStyleSheet(_GB_STYLE)
        v_mouse = QHBoxLayout(g_mouse); v_mouse.setSpacing(4)
        v_mouse.setContentsMargins(4, 4, 4, 4)
        self.mouse_mode = 'hairline'   # 'hairline' | 'labels'
        self.btn_mouse_mode = QPushButton("HAIRLINE")
        self.btn_mouse_mode.setFixedHeight(24)
        self.btn_mouse_mode.setMinimumWidth(110)
        self.btn_mouse_mode.setStyleSheet(
            "QPushButton{background-color:#90CAF9;color:#000;font-weight:bold;}")
        self.btn_mouse_mode.clicked.connect(self._toggle_mouse_mode)
        v_mouse.addWidget(self.btn_mouse_mode)
        right.addWidget(g_mouse)

        # Compact row: Selection (ALL/NONE) + Source (FIT/EXP) side-by-side
        row_sel = QHBoxLayout(); row_sel.setSpacing(4)
        g2 = QGroupBox("Selection"); g2.setStyleSheet(_GB_STYLE)
        v2 = QHBoxLayout(g2); v2.setSpacing(4); v2.setContentsMargins(4, 4, 4, 4)
        b_tutti = QPushButton("ALL"); b_tutti.setFixedHeight(24)
        b_tutti.clicked.connect(self._tutti)
        b_nessuno = QPushButton("NONE"); b_nessuno.setFixedHeight(24)
        b_nessuno.clicked.connect(self._nessuno)
        v2.addWidget(b_tutti); v2.addWidget(b_nessuno)
        row_sel.addWidget(g2)

        g3 = QGroupBox("Source"); g3.setStyleSheet(_GB_STYLE)
        v3 = QHBoxLayout(g3); v3.setSpacing(2); v3.setContentsMargins(4, 4, 4, 4)
        self.btn_source = QPushButton("FIT"); self.btn_source.setFixedHeight(24)
        self.btn_source.setMinimumWidth(70)
        self.btn_source.setToolTip("Source of the interference maxima used to COMPUTE d and n(λ):\n"
                                   "FIT = peaks of the simulated spectrum, EXP = peaks of the measured\n"
                                   "spectrum. The choice is shown on the plot title (n from FIT/EXP maxima).")
        self.btn_source.setCheckable(True); self.btn_source.setChecked(True)
        self.btn_source.setStyleSheet(
            "QPushButton{font-weight:bold;}"
            "QPushButton:checked{background:#FFF9C4}"
            "QPushButton:!checked{background:#FFCCBC}")
        self.btn_source.toggled.connect(self._on_source_toggled)
        v3.addWidget(self.btn_source)
        row_sel.addWidget(g3)
        right.addLayout(row_sel)

        # Load per angle (more compact, 1 row × 4 columns)
        g4 = QGroupBox("Load spectrum"); g4.setStyleSheet(_GB_STYLE)
        v4 = QGridLayout(g4); v4.setSpacing(2); v4.setContentsMargins(4, 4, 4, 4)
        self._led_labels = {}
        for i, ang in enumerate([8, 20, 40, 60]):
            b = QPushButton(f"{ang}°"); b.setFixedHeight(22); b.setFixedWidth(54)
            b.clicked.connect(self._make_loader(ang))
            self._toggle_btns[(ang, 'load')] = b
            v4.addWidget(b, 0, i*2)
            led = QLabel("●")
            has = self.state.data_angles[ang]["exp"] is not None
            led.setStyleSheet(f"color:{'green' if has else 'red'}; font-size:11pt;")
            self._led_labels[ang] = led
            v4.addWidget(led, 0, i*2+1)
        right.addWidget(g4)

        # Parameters (Sample + 4 input boxes in 1 compact row)
        g5 = QGroupBox("Parameters"); g5.setStyleSheet(_GB_STYLE)
        v5 = QVBoxLayout(g5); v5.setSpacing(2); v5.setContentsMargins(4, 4, 4, 4)
        hr = QHBoxLayout(); hr.setSpacing(2)
        hr.addWidget(QLabel("Sample:"))
        self.txt_name = QLineEdit("Sample_01"); self.txt_name.setFixedHeight(22)
        hr.addWidget(self.txt_name); v5.addLayout(hr)

        # 4 input boxes in a 2x4 grid (R min/max, n min/max).
        g_lim = QGridLayout(); g_lim.setSpacing(3)
        def _mk_input(default, callback):
            le = QLineEdit(default); le.setFixedWidth(60); le.setFixedHeight(24)
            le.editingFinished.connect(callback)
            return le
        g_lim.addWidget(QLabel("R:"), 0, 0)
        self.txt_ymin_sx = _mk_input("0",  self._on_axes_limits)
        g_lim.addWidget(self.txt_ymin_sx, 0, 1)
        g_lim.addWidget(QLabel("–"), 0, 2)
        self.txt_ymax    = _mk_input("65", self._on_axes_limits)
        g_lim.addWidget(self.txt_ymax,    0, 3)
        g_lim.addWidget(QLabel("n:"), 1, 0)
        self.txt_ymin_dx = _mk_input("1.0", self._on_axes_limits)
        g_lim.addWidget(self.txt_ymin_dx, 1, 1)
        g_lim.addWidget(QLabel("–"), 1, 2)
        self.txt_ymax_dx = _mk_input("3.0", self._on_axes_limits)
        g_lim.addWidget(self.txt_ymax_dx, 1, 3)
        v5.addLayout(g_lim)
        right.addWidget(g5)

        # Actions (4 buttons in a 2×2 grid). Row spacing and top margin are
        # sized so the groupbox title does not overlap the first row.
        g6 = QGroupBox("Actions"); g6.setStyleSheet(_GB_STYLE)
        v6 = QGridLayout(g6); v6.setHorizontalSpacing(4); v6.setVerticalSpacing(8)
        v6.setContentsMargins(6, 10, 6, 6)
        b_agg = QPushButton("UPDATE"); b_agg.setFixedHeight(26)
        b_agg.clicked.connect(self._refresh)
        b_rst = QPushButton("RESET"); b_rst.setFixedHeight(26)
        b_rst.clicked.connect(self._reset)
        b_csv = QPushButton("SAVE CSV"); b_csv.setFixedHeight(26)
        b_csv.clicked.connect(self._salva_csv)
        b_chiu = QPushButton("CLOSE"); b_chiu.setFixedHeight(26)
        b_chiu.clicked.connect(self.close)
        v6.addWidget(b_agg, 0, 0); v6.addWidget(b_rst, 0, 1)
        v6.addWidget(b_csv, 1, 0); v6.addWidget(b_chiu, 1, 1)
        # EXPORT FIG: saves the figure as HD PDF/PNG/SVG
        b_exp = QPushButton("EXPORT FIG"); b_exp.setFixedHeight(26)
        b_exp.setStyleSheet("QPushButton{background-color:#FFE082;color:#000;font-weight:bold;}")
        b_exp.clicked.connect(self._on_export_figure)
        v6.addWidget(b_exp, 2, 0, 1, 2)
        right.addWidget(g6)

        # Calculation (n-d, n(λ), d result)
        g7 = QGroupBox("Calculation"); g7.setStyleSheet(_GB_STYLE)
        v7 = QVBoxLayout(g7); v7.setSpacing(8); v7.setContentsMargins(6, 12, 6, 8)
        b_nd = QPushButton("COMPUTE d (8/20)"); b_nd.setFixedHeight(24)
        b_nd.setStyleSheet(
            "QPushButton{background-color:#D1C4E9;color:#000;font-weight:bold;}")
        b_nd.clicked.connect(self._on_compute_d_clicked); v7.addWidget(b_nd)
        # Two mutually-exclusive radios choose WHICH d feeds n(λ) (and the FIT/EXP toggle):
        # the two-angle COMPUTE d, or a user-typed d (e.g. the d-scan d_eff). So switching
        # the maxima source does not overwrite the chosen d.
        self._d_group = QButtonGroup(self)
        row_dc = QHBoxLayout(); row_dc.setSpacing(4)
        self.rb_d_computed = QRadioButton(); self.rb_d_computed.setChecked(True)
        self.rb_d_computed.setToolTip("Use the computed (8/20) d for n(λ)")
        self._d_group.addButton(self.rb_d_computed); row_dc.addWidget(self.rb_d_computed)
        self.lbl_d_res = QLabel("d (nm):  --")
        self.lbl_d_res.setStyleSheet("font-size:10pt; font-weight:bold; padding:2px;")
        row_dc.addWidget(self.lbl_d_res); row_dc.addStretch()
        v7.addLayout(row_dc)
        row_du = QHBoxLayout(); row_du.setSpacing(4)
        self.rb_d_custom = QRadioButton()
        self.rb_d_custom.setToolTip("Use the d typed here for n(λ) (e.g. the d-scan d_eff)")
        self._d_group.addButton(self.rb_d_custom); row_du.addWidget(self.rb_d_custom)
        row_du.addWidget(QLabel("d (nm):"))
        self._custom_d_edit = QLineEdit(); self._custom_d_edit.setFixedHeight(24)
        self._custom_d_edit.setPlaceholderText("e.g. 2050")
        self._custom_d_edit.textEdited.connect(lambda *_: self.rb_d_custom.setChecked(True))
        row_du.addWidget(self._custom_d_edit)
        v7.addLayout(row_du)
        b_nwl = QPushButton("CALC n(λ)"); b_nwl.setFixedHeight(24)
        b_nwl.setStyleSheet(
            "QPushButton{background-color:#B3E5FC;color:#000;font-weight:bold;}")
        b_nwl.clicked.connect(self._calcola_n_lambda); v7.addWidget(b_nwl)
        right.addWidget(g7)

        # Hairline + value table
        g8 = QGroupBox("Hairline"); g8.setStyleSheet(_GB_STYLE)
        v8 = QVBoxLayout(g8); v8.setSpacing(2); v8.setContentsMargins(4, 4, 4, 4)
        self.lbl_hairline = QLabel("λ: —")
        self.lbl_hairline.setStyleSheet(
            "font-family:Menlo,Courier,monospace; font-size:9pt; "
            "background:#FAFAFA; color:#000; padding:3px; "
            "border:1px solid #DDD;")
        self.lbl_hairline.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
                                       if PYQT == 6 else
                                       Qt.AlignTop | Qt.AlignLeft)
        # Dedicated scroll: with all angles the hairline lists ~20 rows (R + n per curve),
        # which overflow the fixed-height QLabel and get clipped. Scroll to reach them all.
        self._hair_scroll = QScrollArea()
        self._hair_scroll.setWidgetResizable(True)
        self._hair_scroll.setWidget(self.lbl_hairline)
        self._hair_scroll.setMinimumHeight(90)
        # Grow to fill the spare vertical space of the right panel: when there is room
        # all rows show without scrolling; scroll only when the window is too short.
        # (min 90 keeps it usable when cramped.)
        self._hair_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred if PYQT == 6 else QSizePolicy.Preferred,
            QSizePolicy.Policy.Expanding if PYQT == 6 else QSizePolicy.Expanding)
        self._hair_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff if PYQT == 6 else Qt.ScrollBarAlwaysOff)
        v8.addWidget(self._hair_scroll)
        g8.setSizePolicy(
            QSizePolicy.Policy.Preferred if PYQT == 6 else QSizePolicy.Preferred,
            QSizePolicy.Policy.Expanding if PYQT == 6 else QSizePolicy.Expanding)
        right.addWidget(g8, 1)   # stretch=1: the Hairline box absorbs the leftover space
        # (no trailing addStretch: the spare space goes to the Hairline box above)

        # Hairline: vertical line that follows the mouse + callback
        # `_hairline_obj` recreated in _refresh after every axes clear.
        self._hairline_obj = None
        self._hairline_visible = False
        # Pinned readout: a click latches the value table at that x so it stays
        # on screen when the mouse leaves the plot, until the next click.
        self._hairline_pinned_text = None
        self._hairline_pinned_x = None
        # Manual drag of the m+λ labels (matplotlib's native Annotation.draggable()
        # is unreliable with textcoords='offset points' + a canvas embedded in Qt).
        # Approach: a list of annotations, on click identify the one under the
        # mouse, on motion update xytext.
        self._draggable_anns = []
        self._drag_target = None
        self._drag_start_px = (0, 0)
        self._drag_start_off = (0, 0)
        try:
            self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
            self.canvas.mpl_connect('axes_leave_event',   self._on_canvas_leave)
            self.canvas.mpl_connect('button_press_event',   self._on_canvas_press)
            self.canvas.mpl_connect('button_press_event',   self._on_label_press)
            self.canvas.mpl_connect('motion_notify_event',  self._on_label_motion)
            self.canvas.mpl_connect('button_release_event', self._on_label_release)
        except Exception:
            pass

        self.status=QStatusBar(); self.setStatusBar(self.status)

    def _make_toggle(self,diz,ang,key,btn):
        def cb(checked):
            diz[ang]=checked; self._refresh()
        return cb

    def _make_loader(self,ang):
        def cb():
            path,_=QFileDialog.getOpenFileName(self,f"Load {ang}°","",
                "Text (*.txt *.csv *.asc *.dat);;All (*)")
            if not path: return
            result=self.engine.load_spectral_file(path,ang)
            if result:
                # Update the LED
                if ang in self._led_labels:
                    self._led_labels[ang].setStyleSheet("color:green; font-size:12pt;")
                self._refresh(); self.status.showMessage(f"Loaded {ang}°")
        return cb

    def _on_source_toggled(self,checked):
        self.use_fit[0]=checked
        self.btn_source.setText("FIT" if checked else "EXP")
        # Re-extract the maxima for the NEW source (so the n markers + the computed-d
        # readout track it), but KEEP the active-d choice (the radios): n(λ) is recomputed
        # at the ACTIVE d, so switching the source does not overwrite a custom d you set.
        had_n = any(v is not None for v in self.local_n.values())
        self._calcola_nd()                 # re-extract peaks + refresh the computed-d readout (radio unchanged)
        if had_n and self._active_d() is not None:
            self._calcola_n_lambda()       # n(λ) at the ACTIVE d (computed or custom)

    def _tutti(self):
        for a in [8,20,40,60]:
            self.vis_exp[a]=self.vis_fit[a]=self.vis_n[a]=self.vis_n_fit[a]=self.vis_lbl[a]=True
            for key in ['exp','fit','n','n_fit','lbl']:
                if (a,key) in self._toggle_btns:
                    self._toggle_btns[(a,key)].setChecked(True)
        self._refresh()

    def _nessuno(self):
        for a in [8,20,40,60]:
            self.vis_exp[a]=self.vis_fit[a]=self.vis_n[a]=self.vis_n_fit[a]=self.vis_lbl[a]=False
            for key in ['exp','fit','n','n_fit','lbl']:
                if (a,key) in self._toggle_btns:
                    self._toggle_btns[(a,key)].setChecked(False)
        self._refresh()

    def _on_canvas_motion(self, event):
        """Update the hairline + value table under the cursor.

        The matplotlib lines all have a `label` (assigned in _refresh), so we
        iterate over self.ax.lines / self.ax_n.lines and do
        np.interp(x, xdata, ydata) to get the value at the cursor's x.

        Early return if `mouse_mode == 'labels'`: in that mode the m+λ labels are
        being dragged and the continuous hairline update would interfere with the
        drag (motion events + canvas.draw).
        """
        if getattr(self, 'mouse_mode', 'hairline') != 'hairline':
            return
        if event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        # Update the hairline x-position
        if self._hairline_obj is not None:
            self._hairline_obj.set_xdata([x, x])
            if not self._hairline_visible:
                self._hairline_obj.set_visible(True)
                self._hairline_visible = True
        self.lbl_hairline.setText("\n".join(self._hairline_rows_at(x)))
        # Throttled draw: idle only, no flush_events (frequent mouse_moves)
        self.canvas.draw_idle()

    def _hairline_rows_at(self, x):
        """Build the compact monospace value table (R and n per visible curve) at
        wavelength `x`, by interpolating every labelled line. Shared by the live
        hairline and the pinned (click-latched) readout."""
        lines_r = []
        lines_n = []
        for ln in self.ax.lines:
            lbl = ln.get_label()
            if not lbl or lbl.startswith('_'):
                continue
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            if len(xd) < 2:
                continue
            try:
                if x < float(np.min(xd)) or x > float(np.max(xd)):
                    continue
                y = float(np.interp(x, xd, yd))
                lines_r.append((lbl, y))
            except Exception:
                continue
        for ln in self.ax_n.lines:
            lbl = ln.get_label()
            if not lbl or lbl.startswith('_'):
                continue
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            if len(xd) < 2:
                continue
            try:
                if x < float(np.min(xd)) or x > float(np.max(xd)):
                    continue
                y = float(np.interp(x, xd, yd))
                lines_n.append((lbl, y))
            except Exception:
                continue
        rows = [f"λ: {x:7.1f} nm", ""]
        if lines_r:
            rows.append("── R (%) ──")
            for lbl, y in lines_r:
                rows.append(f"{lbl[:14]:14s} {y:6.2f}")
        if lines_n:
            rows.append("")
            rows.append("──   n   ──")
            for lbl, y in lines_n:
                rows.append(f"{lbl[:14]:14s} {y:6.3f}")
        return rows

    def _on_canvas_press(self, event):
        """Pin the hairline readout on left-click (hairline mode): latch the value
        table at that x so it stays on screen when the mouse leaves the plot, until
        the next click."""
        if getattr(self, 'mouse_mode', 'hairline') != 'hairline':
            return
        # ignore clicks while a toolbar tool (pan/zoom) is active
        if getattr(getattr(self, 'toolbar', None), 'mode', '') != '':
            return
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        rows = self._hairline_rows_at(x)
        rows[0] = f"λ: {x:7.1f} nm  [PIN]"
        self._hairline_pinned_text = "\n".join(rows)
        self._hairline_pinned_x = x
        if self._hairline_obj is not None:
            self._hairline_obj.set_xdata([x, x])
            self._hairline_obj.set_visible(True)
            self._hairline_visible = True
        self.lbl_hairline.setText(self._hairline_pinned_text)
        self.canvas.draw_idle()

    def _on_canvas_leave(self, event):
        """Mouse left the axes: hide the hairline and clear the table — UNLESS a
        click pinned the readout, in which case keep the pinned table + hairline."""
        if self._hairline_pinned_text is not None:
            # keep the pinned snapshot visible (data persists off-plot)
            if self._hairline_obj is not None and self._hairline_pinned_x is not None:
                self._hairline_obj.set_xdata([self._hairline_pinned_x,
                                              self._hairline_pinned_x])
                self._hairline_obj.set_visible(True)
                self._hairline_visible = True
                self.canvas.draw_idle()
            self.lbl_hairline.setText(self._hairline_pinned_text)
            return
        if self._hairline_obj is not None and self._hairline_visible:
            self._hairline_obj.set_visible(False)
            self._hairline_visible = False
            self.canvas.draw_idle()
        self.lbl_hairline.setText("λ: —")

    def _on_label_press(self, event):
        """Start dragging a label (only in mouse_mode='labels')."""
        if getattr(self, 'mouse_mode', 'hairline') != 'labels':
            return
        if event.button != 1 or event.x is None or event.y is None:
            return
        # Find the first annotation whose text bbox contains the click
        renderer = self.canvas.get_renderer()
        for ann in self._draggable_anns:
            try:
                bbox = ann.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if bbox.contains(event.x, event.y):
                self._drag_target = ann
                self._drag_start_px = (float(event.x), float(event.y))
                # xytext in 'offset points' is returned by get_position()
                pos = ann.get_position()
                self._drag_start_off = (float(pos[0]), float(pos[1]))
                break

    def _on_label_motion(self, event):
        """Update the label position while dragging."""
        if self._drag_target is None or event.x is None or event.y is None:
            return
        # Δpixel → Δpoints: 1 point = 1/72 inch, 1 inch = dpi pixels
        dpi = float(self.canvas.figure.dpi)
        dx_pt = (event.x - self._drag_start_px[0]) * 72.0 / dpi
        dy_pt = (event.y - self._drag_start_px[1]) * 72.0 / dpi
        new_x = self._drag_start_off[0] + dx_pt
        new_y = self._drag_start_off[1] + dy_pt
        self._drag_target.set_position((new_x, new_y))
        self.canvas.draw_idle()

    def _on_label_release(self, event):
        """End of label drag."""
        self._drag_target = None

    def _toggle_mouse_mode(self):
        """Toggle between HAIRLINE (cross-hair to read R/n at the cursor) and
        LABELS (m+λ labels draggable with the mouse). The two systems are
        mutually exclusive: the first intercepts all motion events, the second
        requires the drag events to reach the manual label-drag handlers.
        """
        if self.mouse_mode == 'hairline':
            self.mouse_mode = 'labels'
            self.btn_mouse_mode.setText("LABELS")
            self.btn_mouse_mode.setStyleSheet(
                "QPushButton{background-color:#FFCC80;color:#000;font-weight:bold;}")
            # Hide the hairline and clear the value table (+ drop any pin)
            if self._hairline_obj is not None and self._hairline_visible:
                self._hairline_obj.set_visible(False)
                self._hairline_visible = False
            self._hairline_pinned_text = None
            self._hairline_pinned_x = None
            self.lbl_hairline.setText("λ: — (LABELS mode)")
        else:
            self.mouse_mode = 'hairline'
            self.btn_mouse_mode.setText("HAIRLINE")
            self.btn_mouse_mode.setStyleSheet(
                "QPushButton{background-color:#90CAF9;color:#000;font-weight:bold;}")
            self.lbl_hairline.setText("λ: —")
        # Recreate the plot to enable/disable draggable on the annotations
        self._refresh()

    def _update_layer_colors(self):
        """Color the toggle buttons (4 angles x exp/fit/n/n_fit) according to state.

        - GRAY: data not available for that (angle, key)
        - GREEN: data available AND visible (button checked)
        - RED: data available BUT hidden (button !checked)

        Called after _refresh, _make_toggle, _tutti, _nessuno and after
        load_spectral_file / _calcola_n_lambda (when new data exist).
        """
        if not hasattr(self, '_toggle_btns'):
            return
        s = self.state
        for (ang, key), btn in self._toggle_btns.items():
            if key == 'load':
                continue  # handled separately (separate LEDs)
            # Check data existence
            if key == "exp":
                has_data = s.data_angles[ang]["exp"] is not None
            elif key == "fit":
                has_data = s.data_angles[ang]["fit"] is not None
            elif key == "n":
                has_data = (hasattr(self, 'local_n') and
                            self.local_n.get(ang) is not None)
            elif key == "n_fit":
                has_data = s.data_angles[ang].get("ext_override") is not None
            elif key == "lbl":
                # the m/λ labels live on the SOURCE curve (per the FIT/EXP toggle):
                # active only if that curve exists AND is currently shown for this angle
                if self.use_fit[0]:
                    has_data = (s.data_angles[ang]["fit"] is not None
                                and self.vis_fit[ang])
                else:
                    has_data = (s.data_angles[ang]["exp"] is not None
                                and self.vis_exp[ang])
            else:
                has_data = False
            checked = btn.isChecked()
            if not has_data:
                # GRAY: no data
                bg_on = bg_off = "#d0d0d0"
            elif checked:
                # GREEN: data present and visible
                bg_on = "#90ee90"; bg_off = "#d0d0d0"
            else:
                # RED: data present but hidden
                bg_on = "#90ee90"; bg_off = "#ff8a80"
            btn.setStyleSheet(
                f"QPushButton{{background:{bg_on};border:none;border-radius:3px}}"
                f"QPushButton:!checked{{background:{bg_off}}}")

    def _on_axes_limits(self):
        """Update the 4 axis limits.

        - txt_ymin_sx / txt_ymax  → left axis (Reflectance %)
        - txt_ymin_dx / txt_ymax_dx → right axis (n)
        Invalid limits are ignored (keeps the previous one).
        """
        def _read(widget, default):
            try:
                return float(widget.text())
            except ValueError:
                return default
        rmin = _read(self.txt_ymin_sx, 0.0)
        rmax = _read(self.txt_ymax, 65.0)
        nmin = _read(self.txt_ymin_dx, 1.0)
        nmax = _read(self.txt_ymax_dx, 3.0)
        if rmax > rmin:
            self.ax.set_ylim(rmin, rmax)
        if nmax > nmin:
            self.ax_n.set_ylim(nmin, nmax)
        self.canvas.draw_idle()

    def _on_ymax(self):
        self._on_axes_limits()

    def _calcola_nd(self):
        """Compute thickness d from the 8° and 20° peaks via the unified method.

        Calls `engine.calcola_d_da_picchi`, the SAME method used by fit_auto, so
        the same source/aggregation yields the same value of d in both places.

        The local FIT/EXP toggle (`self.use_fit[0]`) maps to the method's
        `source` parameter:
        - usa_fit=True  → source='fit'
        - usa_fit=False → source='exp'
        ('hybrid' is also supported by the engine.)

        Aggregation: 'weighted_median' by default (more precise). The computation
        includes the thin_film branch with the single-angle formula when
        appropriate.

        Also saves peaks_wl/m_values in state.data_angles for marker display.
        """
        s = self.state
        engine = self.engine
        usa_fit = self.use_fit[0]
        source = 'fit' if usa_fit else 'exp'
        self._maxima_src = source          # remember which peaks feed d and n(λ) (shown on the plot)

        # Reset previous data
        for ang in self.local_n:
            self.local_n[ang] = None
        for ang in [8, 20, 40, 60]:
            s.data_angles[ang].pop("peaks_wl", None)
            s.data_angles[ang].pop("m_values", None)

        # Peak extraction for n(λ) and for the window markers. Bragg-guided:
        # position from the data (find_peaks + parabolic-in-E), order+filter from
        # the Bragg relation (λ>500 nm, spurious dropped, n-smoothness auto-discard).
        # The d value is unaffected: it comes from calcola_d_da_picchi, which keeps
        # its own (blind, 8/20) extraction — so d stays an independent measurement.
        for ang in [8, 20, 40, 60]:
            if s.data_angles[ang]["exp"] is None:
                continue
            peaks_wl, m_values = engine.extract_maxima_bragg(ang, source, s.d)
            if peaks_wl is not None:
                s.data_angles[ang]["peaks_wl"] = peaks_wl
                s.data_angles[ang]["m_values"] = m_values

        # Unified d computation (NO cache, on-demand)
        # NB: we save and restore the engine cache so as not to interfere with
        # any fit_auto in progress (the cache is valid only within a fit_auto
        # run; here we are standalone)
        saved_cache = engine._d_peaks_cache
        engine._d_peaks_cache = None
        try:
            d_result = engine.calcola_d_da_picchi(
                source=source, aggregation='weighted_median')
        finally:
            engine._d_peaks_cache = saved_cache

        if d_result is not None:
            self._d_mem = d_result
            s.d_mem_global = d_result
            src_tag = source.upper()
            agg_tag = "wmedian"
            self.lbl_d_res.setText(f"d (nm):  {d_result:.1f}  [{src_tag}]")
            self.status.showMessage(
                f"d = {d_result:.1f} nm  (source: {src_tag}, agg: {agg_tag})")
        else:
            self.lbl_d_res.setText("d (nm):  --")
            self.status.showMessage(
                f"d not computable (source: {source.upper()})")
        self._refresh()

    def _on_compute_d_clicked(self):
        """COMPUTE d button: compute the two-angle (8/20) d and select it as the active d."""
        self._calcola_nd()
        self.rb_d_computed.setChecked(True)   # explicit compute → use this d for n(λ)

    def _active_d(self):
        """The d that feeds n(λ), per the two mutually-exclusive radios: the computed
        (8/20) d, or the user-typed d. None if unavailable/invalid."""
        if self.rb_d_custom.isChecked():
            try:
                v = float(self._custom_d_edit.text())
            except (ValueError, TypeError):
                return None
            return v if 100.0 < v < 5000.0 else None
        return getattr(self, '_d_mem', None)

    def _calcola_n_lambda(self):
        """Compute n(λ) from the maxima at the ACTIVE d (the computed 8/20 d OR the
        user-typed d, per the two radios). The maxima SOURCE is the FIT/EXP toggle.
        Delegates to `engine.compute_n_from_maxima` (SHARED with the main window
        Fit/Maxima view; also writes state.data_angles[ang]['n_maxima']). n_fit is
        overlaid for the direct comparison of the two methods."""
        d = self._active_d()
        if d is None:
            self.status.showMessage("Enter a valid custom d (100–5000 nm)"
                                    if self.rb_d_custom.isChecked()
                                    else "Compute d first (8/20)")
            return
        if self._maxima_src is None:
            self.status.showMessage("Run COMPUTE d first (it extracts the maxima)"); return
        for ang in self.local_n:
            self.local_n[ang] = self.engine.compute_n_from_maxima(ang, d)
        src = (self._maxima_src or "?").upper()
        dtag = "user d" if self.rb_d_custom.isChecked() else "8/20 d"
        self._refresh()
        self.status.showMessage(f"n(λ) from {src} maxima at {dtag} = {d:.0f} nm")

    def _reset(self):
        for a in [8,20,40,60]:
            self.local_n[a]=None
            self.state.data_angles[a]["m_values"]=None
        self._refresh()

    def _on_export_figure(self):
        """Export the peak_window figure (R + peaks + n).
        Formats: vector PDF (default), PNG 600 DPI, vector SVG.
        """
        from .export_fig import export_figure
        name = self.txt_name.text().strip() or "sample"
        default = f"{name}_peaks"
        path = export_figure(
            self.fig, parent=self, default_name=default,
            title="Export peak analysis figure",
            metadata={"Title": f"OPHIRA peak analysis — {name}",
                      "Creator": "OPHIRA"})
        if path:
            self.status.showMessage(f"Figure exported: {path}")

    def _salva_csv(self):
        import csv
        path,_=QFileDialog.getSaveFileName(self,"Save analysis","",
            "CSV (*.csv);;All (*)")
        if not path: return
        righe=[["Angle","Order (m)","Wavelength (nm)","n"]]
        for ang in [8,20,40,60]:
            nd=self.local_n[ang]
            mv=self.state.data_angles[ang].get("m_values")
            if nd is not None and mv is not None:
                for i in range(len(nd)):
                    m=mv[i] if i<len(mv) else "N/A"
                    righe.append([ang,m,f"{nd[i,0]:.2f}",f"{nd[i,1]:.4f}"])
        if len(righe)>1:
            with open(path,'w',newline='') as f:
                csv.writer(f).writerows(righe)
            self.status.showMessage(f"Saved: {path}")

    def _refresh(self,_=None):
        # Reset the draggable-annotations list: the old objects point to
        # Annotations already destroyed by ax.clear(). _drag_target must also be
        # cleared to avoid pointing to a dead object.
        self._draggable_anns = []
        self._drag_target = None
        """Refresh the plot."""
        self.ax.clear(); self.ax_n.clear()
        # UPDATE re-syncs the Load-spectrum LEDs with the actual data presence
        for _a, _led in getattr(self, "_led_labels", {}).items():
            _has = self.state.data_angles[_a]["exp"] is not None
            _led.setStyleSheet(f"color:{'green' if _has else 'red'}; font-size:11pt;")
        # Sample name intentionally NOT in the title: keeps an exported figure paper-clean.
        # The name still lives in the Sample field and the CSV export.
        _src = getattr(self, "_maxima_src", None)
        _src_note = f"n from {_src.upper()} maxima" if _src else ""
        self.ax.set_title(f"IR Peak Analysis{'   ·   ' + _src_note if _src_note else ''}")
        self.ax.set_xlabel("Wavelength (nm)", fontsize=12); self.ax.set_ylabel("Reflectance (%)", color='blue', fontsize=12)
        self.ax_n.set_ylabel("Refractive index, n", color='darkgreen', fontsize=12)
        self.ax_n.yaxis.set_label_position("right")   # ax_n.clear() resets it to the left
        # Read 4 limits from the widgets; defaults on error.
        def _read(widget, default):
            try: return float(widget.text())
            except ValueError: return default
        rmin = _read(self.txt_ymin_sx, 0.0)
        rmax = _read(self.txt_ymax, 65.0)
        nmin = _read(self.txt_ymin_dx, 1.0)
        nmax = _read(self.txt_ymax_dx, 3.0)
        if rmax <= rmin: rmin, rmax = 0.0, 65.0
        if nmax <= nmin: nmin, nmax = 1.0, 3.0
        self.ax.set_ylim(rmin, rmax); self.ax.set_xlim(450, 2600)
        self.ax.grid(True, alpha=0.2); self.ax_n.set_ylim(nmin, nmax)
        usa_fit=self.use_fit[0]
        s=self.state

        for ang in [8,20,40,60]:
            dd=s.data_angles[ang]
            if dd["exp"] is None: continue
            col=self.COLORS[ang]
            wl,refl=dd["exp"][0],dd["exp"][1]
            mask=wl>=450; wf,rf=wl[mask],refl[mask]

            # IR correction — sliders → state priority (consistent with
            # `_extract_peaks`). Use an explicit None check, not `or`: the `or`
            # operator treats 0.0 as falsy and would fall back to the state,
            # causing inconsistency between the plotted curve and the curve the
            # peaks are extracted from (marker on a "flank" because the display
            # curve was corrected with a kb different from the sliders' real one).
            sl = dd.get("sliders") or {}
            km = sl.get("km") if sl.get("km") is not None else s.km
            ks = sl.get("ks") if sl.get("ks") is not None else s.ks
            kb = sl.get("kb") if sl.get("kb") is not None else s.kb
            rf_clean=rf.copy()
            mir=wf>890
            if np.any(mir) and km>0:
                rf_clean[mir]/=ph.ir_instr_factor(wf[mir],km,ks,kb)

            sorgente=None
            if self.vis_exp[ang]:
                self.ax.plot(wf,rf_clean,color=col,ls='--',alpha=0.5,label=f'{ang}° exp',lw=1.2)
                if not usa_fit: sorgente=rf_clean

            if dd["fit"] is not None and self.vis_fit[ang]:
                rfit=dd["fit"][mask].copy()
                if np.any(mir) and km>0:
                    rfit[mir]/=ph.ir_instr_factor(wf[mir],km,ks,kb)
                self.ax.plot(wf,rfit,color=col,ls='-',label=f'{ang}° fit',lw=2.2)
                if usa_fit: sorgente=rfit

            if sorgente is not None:
                # display↔computation CONSISTENCY: use the same `_extract_peaks`
                # as the `calcola_d_da_picchi` computation. Plain find_peaks
                # (height=1.0, no prominence) shows dozens of spurious peaks from
                # noise on raw exp data; the computation discards the spurious
                # ones with prominence=2.0 (for source 'exp_only'). Using the same
                # extractor keeps the displayed peaks 1:1 with the ones used.
                src_maxima = 'fit' if usa_fit else 'exp'
                try:
                    pks_refined, m_vals = self.engine.extract_maxima_bragg(
                        ang, src_maxima, self.state.d)
                except Exception:
                    pks_refined, m_vals = None, None
                if pks_refined is not None and len(pks_refined) > 0:
                    # The refined position is subpixel → interpolate y on the
                    # plotted source curve.
                    y_pks = np.interp(pks_refined, wf, sorgente)
                    self.ax.plot(pks_refined, y_pks, 'o',
                                 color=col, markersize=5)
                    # Position AND order come together from the Bragg-guided
                    # extractor (position data-driven; order from the Bragg
                    # relation; λ>500; n-smoothness guard). It is the SAME source
                    # that feeds n(λ) in _calcola_nd, so the markers and the n(λ)
                    # curve stay 1:1. m and d are not recomputed here.
                    has_m = m_vals is not None and len(m_vals) == len(pks_refined)
                    live_m = ({float(p): float(mm)
                               for p, mm in zip(pks_refined, m_vals)}
                              if has_m else None)
                    # Labels spaced with a tier offset + leader line + draggable.
                    # Y tier per angle: 8°=18, 20°=42, 40°=66, 60°=90 px above
                    # the marker; each peak adds +14 px alternating (i%2) to
                    # separate adjacent peaks.
                    # Semi-transparent white background for readability over the
                    # curves. Thin connecting line of the same color as the peak.
                    # Registering the annotation in _draggable_anns enables moving it with the mouse (manual drag) — note:
                    # UPDATE/RESET recreate the annotations and the positions are
                    # lost → move the labels as the last step before EXPORT FIG.
                    base_y = {8: 18, 20: 42, 40: 66, 60: 90}.get(int(ang), 18)
                    for i, (px, py) in (enumerate(zip(pks_refined, y_pks)) if self.vis_lbl[ang] else []):  # tag column gate
                        if has_m and live_m is not None:
                            # Direct lookup in live_m via float key (exact
                            # because pks_refined is the same list used to
                            # build live_m).
                            m_val = live_m.get(float(px))
                            if m_val is not None:
                                m_int = int(round(m_val))
                                label = f"m={m_int}\nλ={px:.0f}"
                            else:
                                label = f"{px:.0f}"
                        else:
                            label = f"{px:.0f}"
                        y_off = base_y + 14 * (i % 2)
                        if py > rmin + 0.72 * (rmax - rmin):   # peak near the top: flip the label below to keep it inside the axes
                            y_off = -y_off
                        ann = self.ax.annotate(
                            label, xy=(px, py),
                            xytext=(0, y_off), textcoords="offset points",
                            ha='center', fontsize=7, color=col,
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='-', lw=0.5,
                                            color=col, alpha=0.7,
                                            shrinkA=0, shrinkB=2),
                            bbox=dict(boxstyle='round,pad=0.25',
                                      fc='white', ec=col, lw=0.5,
                                      alpha=0.85))
                        # Register the annotation for the manual drag
                        # (_on_label_press/_motion/_release). The drag is active
                        # only if mouse_mode == 'labels' (checked inside
                        # _on_label_press).
                        self._draggable_anns.append(ann)

            if self.vis_n[ang] and self.local_n[ang] is not None:
                nd=self.local_n[ang]; nd=nd[nd[:,0].argsort()]
                xn,yn=nd[:,0],nd[:,1]
                try:
                    pp=PchipInterpolator(xn,yn)
                    xnew=np.linspace(xn.min(),xn.max(),300)
                    self.ax_n.plot(xnew,pp(xnew),color=col,ls=':',label=f'{ang}° n',lw=1.5)
                    # Anchor points (n from the maxima): they are few and must
                    # be shown explicitly. Label '_' → excluded from the legend
                    # and the hairline table.
                    self.ax_n.plot(xn,yn,color=col,ls='',marker='o',ms=5,
                                   mfc=col,mec='white',mew=0.7,zorder=6,
                                   label='_nolegend_')
                except Exception:
                    self.ax_n.plot(xn,yn,color=col,ls=':',lw=1.5,marker='o',
                                   ms=5,mfc=col,mec='white',mew=0.7,label=f'{ang}° n')

            # n_fit: n(λ) curve fitted from the per-angle ext_override,
            # overlaid on n_maxima for the direct comparison of the two methods.
            # Thin solid line to distinguish it from the dotted n_maxima.
            ov = dd.get("ext_override")
            if self.vis_n_fit[ang] and ov is not None:
                try:
                    ext_nd = np.asarray(ov[0], dtype=float)
                    n_arr  = np.asarray(ov[1], dtype=float)
                    order  = ext_nd.argsort()
                    xd, yd = ext_nd[order], n_arr[order]
                    # Smooth PCHIP curve on a dense grid — the SAME n(λ) the main window
                    # draws (physics.calculate_refl_core uses PchipInterpolator on the
                    # nodes, clipped to [1,12]), not the raw node polyline, which
                    # would look piecewise-linear in the sparse IR.
                    try:
                        xden = np.linspace(xd[0], xd[-1], 500)
                        yden = np.clip(PchipInterpolator(xd, yd)(xden), 1.0, 12.0)
                    except Exception:
                        xden, yden = xd, yd
                    self.ax_n.plot(xden, yden, color=col,
                                   ls='-', lw=1.3, alpha=0.85, label=f'{ang}° n_fit')
                except Exception:
                    pass

        # If only one category of curves is shown, mirror its axis on the empty
        # side so an exported figure has matching axes (no stray unused axis to
        # edit out): only reflectance -> two reflectance axes; only n -> two n
        # axes. Both present (or neither) -> left as is.
        drew_R = len(self.ax.lines) > 0
        drew_n = len(self.ax_n.lines) > 0
        if drew_R and not drew_n:
            self.ax_n.set_ylim(self.ax.get_ylim())
            self.ax_n.set_ylabel("Reflectance (%)", color='blue', fontsize=12)
        elif drew_n and not drew_R:
            self.ax.set_ylim(self.ax_n.get_ylim())
            self.ax.set_ylabel("Refractive index, n", color='darkgreen', fontsize=12)

        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc='upper left',fontsize=7)
        # Separate legend for the n curves (right axis), at the top RIGHT, so it
        # doesn't lengthen the data/fit one (top left, already long).
        h_n, l_n = self.ax_n.get_legend_handles_labels()
        if h_n:
            self.ax_n.legend(h_n, l_n, loc='upper right', fontsize=7,
                             title='n(λ)', title_fontsize=7, framealpha=0.85)
        # Recreate the hairline after the axes clear()
        try:
            self._hairline_obj = self.ax.axvline(
                np.nan, color='gray', lw=0.8, alpha=0.7, zorder=10,
                visible=False, label='_hairline')
            self._hairline_visible = False
        except Exception:
            self._hairline_obj = None
        self.canvas.draw_idle()
        # Update the layer-button colors according to data state and visibility.
        self._update_layer_colors()
