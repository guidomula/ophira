"""
version.py — single source of truth for OPHIRA metadata.

Imported by main_window for the window title, by the About box, by
export_fig for the PDF metadata, and by the manual build scripts.

Semantic version X.Y.Z:
  X (major) — backward-incompatible changes (e.g. a session format
              that older OPHIRA cannot read)
  Y (minor) — backward-compatible additions (new angle, new material,
              new button)
  Z (patch) — bugfixes, refactors, UX improvements without new features
"""

VERSION    = "1.0.0"
BUILD_DATE = "2026-06-18"

AUTHOR     = "Guido Mula"
AFFILIATION = "Dipartimento di Fisica, Università degli Studi di Cagliari"
EMAIL      = "guido.mula@unica.it"

# Rights holder (copyright) vs author, per the University's decision (2026-07):
# the University holds the rights, Guido Mula is credited as the author.
COPYRIGHT_YEAR   = "2026"
COPYRIGHT_HOLDER = "Università degli Studi di Cagliari"
COPYRIGHT        = f"Copyright {COPYRIGHT_YEAR} {COPYRIGHT_HOLDER}"

LICENSE    = "PolyForm Noncommercial License 1.0.0"
LICENSE_URL = "https://polyformproject.org/licenses/noncommercial/1.0.0"

# Zenodo concept DOI (all versions — the one to cite). The specific
# v1.0.0 archive is 10.5281/zenodo.21495780.
ZENODO_DOI = "10.5281/zenodo.21495779"

# Public GitHub repository URL.
GITHUB_URL = "https://github.com/guidomula/ophira"

APP_NAME      = "OPHIRA"
APP_LONG_NAME = "Optical Thin Film Reflectance Analyzer"


def version_string():
    """Short string 'OPHIRA v1.0.0' for windowTitle and similar uses."""
    return f"{APP_NAME} v{VERSION}"


def full_title():
    """Extended string for the main title: 'OPHIRA v1.0.0 — …'."""
    return f"{version_string()} — {APP_LONG_NAME}"


def about_html():
    """HTML content for the 'About OPHIRA' dialog.
    Intended for QMessageBox.about() / QDialog.setText() with RichText.
    """
    doi_line = (f"<p><b>DOI:</b> "
                f'<a href="https://doi.org/{ZENODO_DOI}">{ZENODO_DOI}</a></p>'
                if ZENODO_DOI else
                "<p><b>DOI:</b> <i>pending — sarà assegnato al rilascio Zenodo</i></p>")
    gh_line = (f'<p><b>Source:</b> <a href="{GITHUB_URL}">{GITHUB_URL}</a></p>'
               if GITHUB_URL else
               "<p><b>Source:</b> <i>repository sarà reso pubblico con la submission del paper</i></p>")
    return (
        f"<h2>{APP_NAME}</h2>"
        f"<p><b>{APP_LONG_NAME}</b></p>"
        f"<p>Version <b>{VERSION}</b> &nbsp;·&nbsp; Build {BUILD_DATE}</p>"
        f"<hr>"
        f"<p><b>Author:</b> {AUTHOR}<br>"
        f"{AFFILIATION}<br>"
        f'<a href="mailto:{EMAIL}">{EMAIL}</a></p>'
        f'<p><b>License:</b> <a href="{LICENSE_URL}">{LICENSE}</a><br>'
        f"<span style='font-size:9pt;'>{COPYRIGHT}.<br>"
        f"Free for noncommercial use; commercial use reserved to the rights holder."
        f"</span></p>"
        f"{doi_line}"
        f"{gh_line}"
        f"<hr>"
        f"<p style='font-size:9pt;color:#666;'>"
        f"Riflettometria multi-angolo per la caratterizzazione di "
        f"film sottili (silicio poroso e simili). Estrae spessore, "
        f"n(λ), k(λ) e birifrangenza Δn(λ) da spettri non polarizzati."
        f"</p>"
    )
