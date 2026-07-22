"""
export_fig.py — High-resolution figure export.

Shared helper used by main_window, peak_window, kk_window.
Supported output:
  - Vector PDF (default, crisp line-art at any zoom)
  - PNG 600 DPI    (for standard print raster)
  - Vector SVG (for post-editing in Inkscape/Illustrator)

File-name convention: {sample}_{window}_{angle?}_{YYYYMMDD}.{ext}
"""
from datetime import datetime

try:
    from PyQt6.QtWidgets import QFileDialog
except ImportError:
    from PyQt5.QtWidgets import QFileDialog

import matplotlib


_FILTER = ("Vector PDF (*.pdf);;"
           "PNG 600 DPI (*.png);;"
           "Vector SVG (*.svg)")


def _default_pdf_metadata(extra=None):
    """Base metadata for exported PDFs: producer = OPHIRA vX.Y.Z.
    Any extra metadata (Title, Subject, etc.) is merged in."""
    try:
        from version import version_string, AUTHOR
        base = {
            "Producer": version_string(),
            "Author":   AUTHOR,
        }
    except Exception:
        base = {}
    if extra:
        base.update(extra)
    return base


def export_figure(fig, parent=None, default_name="figure",
                  title="Export figure",
                  metadata=None):
    """Show a QFileDialog and save the matplotlib figure with the correct
    preset for the chosen format.

    Args:
        fig:          matplotlib.figure.Figure to export
        parent:       parent QWidget for the dialog (may be None)
        default_name: base file name (without extension); the date is
                      appended automatically
        title:        dialog window title
        metadata:     optional dict with PDF metadata (Author, Subject, ...)

    Returns:
        saved path (str), or None if the user cancels.
    """
    today = datetime.now().strftime("%Y%m%d")
    suggested = f"{default_name}_{today}.pdf"
    path, sel_filter = QFileDialog.getSaveFileName(
        parent, title, suggested, _FILTER)
    if not path:
        return None

    # Force the extension consistent with the selected filter if the user
    # didn't add it (common case on Linux/macOS QFileDialog).
    ext_for_filter = {
        "Vector PDF (*.pdf)": ".pdf",
        "PNG 600 DPI (*.png)":    ".png",
        "Vector SVG (*.svg)": ".svg",
    }
    forced_ext = ext_for_filter.get(sel_filter, "")
    if forced_ext and not path.lower().endswith(forced_ext):
        path = path + forced_ext

    # Build the savefig kwargs based on the format
    is_png = path.lower().endswith(".png")
    is_pdf = path.lower().endswith(".pdf")

    save_kwargs = dict(
        bbox_inches='tight',
        facecolor='white',     # force white background (the Qt canvas is dark)
        edgecolor='none',
    )
    if is_png:
        save_kwargs['dpi'] = 600
    else:
        # PDF / SVG: vector; dpi only matters for embedded raster images
        # (none in these plots).
        save_kwargs['dpi'] = 300

    if is_pdf:
        # Always embed Producer=OPHIRA vX.Y.Z + Author in PDFs (useful for
        # provenance and to trace which version produced the figure). Any
        # extra metadata (e.g. Title, Subject passed by the caller) is kept.
        save_kwargs['metadata'] = _default_pdf_metadata(metadata)

    # Save while preserving the original facecolor on return (the canvas
    # may have a dark/transparent background we don't want in the PDF).
    orig_face = fig.get_facecolor()
    try:
        fig.set_facecolor('white')
        fig.savefig(path, **save_kwargs)
    finally:
        fig.set_facecolor(orig_face)

    return path
