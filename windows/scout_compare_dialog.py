"""scout_compare_dialog.py — read-only comparison of an Ophira fit vs a Scout fit.

From a chosen folder it lists the Ophira per-angle ``*_data.csv`` files and
the Scout exports (``.xls/.xlsx`` or ``.csv``); given one of each it
overlays R (exp / Ophira-sim / Scout-sim), n(λ) and k(λ), and reports, over a
chosen band, the RMS deviation of R (sim vs exp) for each tool and the RMS
difference of n and k between the two tools.

Static / read-only: it never touches the current fit state.
"""
import os, glob
import numpy as np

try:
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QComboBox, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog,
        QButtonGroup, QRadioButton)
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT)
    PYQT = 6
except ImportError:
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
        QComboBox, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog,
        QButtonGroup, QRadioButton)
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT)
    PYQT = 5

from matplotlib.figure import Figure


# ── loaders ─────────────────────────────────────────────────────────────────
def load_ophira(path):
    """Ophira _data.csv → dict(wl, R_exp, R_sim, n, k)."""
    a = np.genfromtxt(path, delimiter=',', comments='#')
    if a.ndim != 2 or a.shape[1] < 5:
        raise ValueError("unexpected Ophira CSV layout (need ≥5 columns)")
    a = a[np.isfinite(a[:, 0])]
    return dict(wl=a[:, 0], R_exp=a[:, 1], R_sim=a[:, 2], n=a[:, 3], k=a[:, 4])

def load_scout(path):
    """Scout export → dict(wl, R_exp, R_sim, n, k).

    Accepts .xls/.xlsx (via pandas + xlrd/openpyxl) OR .csv (no Excel library
    needed — just save-as CSV from Excel). Finds the spectral block and reads
    (wl, R_meas, R_sim) at cols 0-2 and (wl, n, k) at cols 4-6."""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        # No Excel library needed. Sniff delimiter + decimal (Italian Excel
        # often exports ';'-separated with comma decimals).
        with open(path, 'r', errors='ignore') as f:
            sample = f.read(4096)
        if ';' in sample:                    # Excel EU export: ';' delim, ',' decimal
            sep, dec = ';', ','
        elif '\t' in sample:
            sep, dec = '\t', '.'
        else:
            sep, dec = ',', '.'
        candidates = [pd.read_csv(path, sep=sep, header=None, decimal=dec, engine='python')]
    else:
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            if 'xlrd' in str(e).lower():
                raise RuntimeError(
                    "Reading .xls needs 'xlrd' (or openpyxl for .xlsx). Simplest: in "
                    "Excel do Save As -> CSV and load that file — no library needed.") from e
            raise
        candidates = [xl.parse(sh, header=None) for sh in xl.sheet_names]
    best = None
    for df in candidates:
        if len(df) == 0 or df.shape[1] < 3:
            continue
        valid = int(pd.to_numeric(df.iloc[:, 0], errors='coerce').notna().sum())
        if valid > 50 and (best is None or valid > best[1]):
            best = (df, valid)
    if best is None:
        raise ValueError("no spectral block found in the Scout file")
    df = best[0]
    # Data columns only (drops blank separator columns + the header row). Layout is
    # WL, R_exp, R_fit, [blank,] [nm,] n, k  →  R = first 3; n,k = last two.
    datacols = [c for c in df.columns
                if int(pd.to_numeric(df[c], errors='coerce').notna().sum()) > 50]
    if len(datacols) < 5:
        raise ValueError(f"Scout data: found {len(datacols)} numeric columns, "
                         "need >=5 (WL, R_exp, R_fit, n, k)")
    def block(cols):
        return df[list(cols)].apply(pd.to_numeric, errors='coerce').dropna().values
    R  = block(datacols[:3])           # WL, R_exp, R_fit
    wl = R[:, 0]
    if len(datacols) >= 6:             # ...(nm,) n, k — n,k carry their own wl column
        NK = block(datacols[-3:])
        n = np.interp(wl, NK[:, 0], NK[:, 1]); k = np.interp(wl, NK[:, 0], NK[:, 2])
    else:                              # 5 cols: WL, R_exp, R_fit, n, k (shared wl)
        A = block(datacols)
        n = np.interp(wl, A[:, 0], A[:, 3]); k = np.interp(wl, A[:, 0], A[:, 4])
    return dict(wl=wl, R_exp=R[:, 1], R_sim=R[:, 2], n=n, k=k)


# ── dialog ──────────────────────────────────────────────────────────────────
class ScoutCompareDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._O = self._S = self._wl = None
        self.setWindowTitle("Compare  Ophira ⇄ Scout")
        self.resize(1000, 760)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        b = QPushButton("Load folder…"); b.clicked.connect(self._load_folder)
        self.cmb_o = QComboBox(); self.cmb_s = QComboBox()
        self.cmb_o.currentIndexChanged.connect(self._update)
        self.cmb_s.currentIndexChanged.connect(self._update)
        top.addWidget(b)
        top.addWidget(QLabel("Ophira:")); top.addWidget(self.cmb_o, 1)
        top.addWidget(QLabel("Scout:"));  top.addWidget(self.cmb_s, 1)
        lay.addLayout(top)

        row2 = QHBoxLayout()
        self.rg = QButtonGroup(self)
        for i, name in enumerate(("R", "n", "k")):
            rb = QRadioButton(name); self.rg.addButton(rb, i)
            if i == 0:
                rb.setChecked(True)
            rb.toggled.connect(lambda on: on and self._draw())
            row2.addWidget(rb)
        row2.addSpacing(24)
        row2.addWidget(QLabel("band λ:"))
        self.ed_lo = QLineEdit("450"); self.ed_hi = QLineEdit("2500")
        for e in (self.ed_lo, self.ed_hi):
            e.setFixedWidth(58); e.editingFinished.connect(self._update)
        row2.addWidget(self.ed_lo); row2.addWidget(QLabel("–"))
        row2.addWidget(self.ed_hi); row2.addWidget(QLabel("nm")); row2.addStretch()
        lay.addLayout(row2)

        self.fig = Figure(figsize=(8, 5))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.07)
        self.ax  = self.fig.add_subplot(gs[0])
        self.axr = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.canvas = FigureCanvas(self.fig)
        tb = NavigationToolbar2QT(self.canvas, self)
        try:
            from windows.widgets import style_nav_toolbar
            style_nav_toolbar(tb)
        except Exception:
            pass
        lay.addWidget(tb)
        lay.addWidget(self.canvas, 1)

        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Metric", "Ophira", "Scout"])
        self.tbl.setMaximumHeight(170)
        lay.addWidget(self.tbl)
        self.lbl_warn = QLabel(""); self.lbl_warn.setStyleSheet("color:#C0392B;")
        lay.addWidget(self.lbl_warn)

    # ── data flow ────────────────────────────────────────────────────────────
    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder with Ophira + Scout files")
        if not folder:
            return
        ofiles = sorted(glob.glob(os.path.join(folder, "*_data.csv")))
        sfiles = sorted(glob.glob(os.path.join(folder, "*.xls")) +
                        glob.glob(os.path.join(folder, "*.xlsx")) +
                        [f for f in glob.glob(os.path.join(folder, "*.csv"))
                         if not os.path.basename(f).endswith("_data.csv")])
        for cmb, files in ((self.cmb_o, ofiles), (self.cmb_s, sfiles)):
            cmb.blockSignals(True); cmb.clear()
            for f in files:
                cmb.addItem(os.path.basename(f), f)
            cmb.blockSignals(False)
        self._update()

    def _band(self):
        try:
            return float(self.ed_lo.text()), float(self.ed_hi.text())
        except ValueError:
            return 450.0, 2500.0

    def _update(self):
        op = self.cmb_o.currentData(); sp = self.cmb_s.currentData()
        if not op or not sp:
            return
        try:
            self._O = load_ophira(op); self._S = load_scout(sp)
        except Exception as e:
            self.lbl_warn.setText(f"Load error: {e}"); return
        self._compute(); self._draw()

    def _compute(self):
        O, S = self._O, self._S
        lo, hi = self._band()
        lo = max(lo, O['wl'].min(), S['wl'].min())
        hi = min(hi, O['wl'].max(), S['wl'].max())
        wl = O['wl'][(O['wl'] >= lo) & (O['wl'] <= hi)]
        self._wl = wl
        I = lambda d, key: np.interp(wl, d['wl'], d[key])
        rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))
        dev_O = rms(I(O, 'R_sim') - I(O, 'R_exp'))
        dev_S = rms(I(S, 'R_sim') - I(S, 'R_exp'))
        dn = rms(I(O, 'n') - I(S, 'n')); dk = rms(I(O, 'k') - I(S, 'k'))
        expdiff = rms(I(O, 'R_exp') - I(S, 'R_exp'))
        self.lbl_warn.setText("" if expdiff < 0.5 else
            f"⚠ Ophira and Scout experimental curves differ (RMS {expdiff:.2f} %R "
            f"over {lo:.0f}–{hi:.0f} nm) — not the same raw spectrum?")
        rows = [
            (f"RMS deviation R (%R), {lo:.0f}–{hi:.0f} nm", dev_O, dev_S),
            ("RMS Δn  (Ophira − Scout)",                    dn,    None),
            ("RMS Δk  (Ophira − Scout)",                    dk,    None),
        ]
        self.tbl.setRowCount(len(rows))
        for r, (name, a, bb) in enumerate(rows):
            self.tbl.setItem(r, 0, QTableWidgetItem(name))
            self.tbl.setItem(r, 1, QTableWidgetItem("—" if a is None else f"{a:.4g}"))
            self.tbl.setItem(r, 2, QTableWidgetItem("—" if bb is None else f"{bb:.4g}"))
        self.tbl.resizeColumnsToContents()

    def _draw(self):
        if self._O is None or self._wl is None:
            return
        wl = self._wl; O, S = self._O, self._S
        I = lambda d, key: np.interp(wl, d['wl'], d[key])
        which = ("R", "n", "k")[self.rg.checkedId()]
        self.ax.clear(); self.axr.clear()
        if which == "R":
            self.ax.plot(wl, I(O, 'R_exp'), color='k', lw=1.2, label='R$_{exp}$')
            self.ax.plot(wl, I(O, 'R_sim'), color='#1f4e79', lw=1.1, label='R Ophira')
            self.ax.plot(wl, I(S, 'R_sim'), color='#c0392b', lw=1.1, ls='--', label='R Scout')
            self.ax.set_ylabel("R (%)")
            self.axr.plot(wl, I(O, 'R_sim') - I(O, 'R_exp'), color='#1f4e79', lw=0.8, label='Ophira')
            self.axr.plot(wl, I(S, 'R_sim') - I(S, 'R_exp'), color='#c0392b', lw=0.8, label='Scout')
            self.axr.set_ylabel("sim − exp")
        else:
            self.ax.plot(wl, I(O, which), color='#1f4e79', lw=1.4, label=f'{which} Ophira')
            self.ax.plot(wl, I(S, which), color='#c0392b', lw=1.4, ls='--', label=f'{which} Scout')
            self.ax.set_ylabel(which)
            self.axr.plot(wl, I(O, which) - I(S, which), color='#555', lw=0.9)
            self.axr.set_ylabel("O − S")
        self.axr.axhline(0, color='#999', lw=0.5)
        self.ax.tick_params(labelbottom=False)
        self.ax.legend(fontsize=8, loc='best')
        self.axr.set_xlabel("λ (nm)")
        self.ax.grid(True, ls=':', lw=0.4); self.axr.grid(True, ls=':', lw=0.4)
        self.ax.set_title("Ophira vs Scout")
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.11)
        self.canvas.draw()
