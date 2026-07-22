# OPHIRA — Optical Thin Film Reflectance Analyzer

OPHIRA is a graphical tool for fitting multi-angle optical reflectance spectra of
thin films (porous silicon and similar). From **unpolarized** reflectance
measurements it extracts the film thickness *d* and the dispersion of the refractive
index *n(λ)*, the extinction coefficient *k(λ)*, and the uniaxial birefringence
*Δn(λ)*, using a Fresnel/Airy thin-film model with Kramers–Kronig consistency checks.

## Features

- Fresnel/Airy reflectance model with a node-based, PCHIP-interpolated *n(λ)*, *k(λ)*.
- Multi-angle fitting (8°, 20°, 40°, 60°) with angular-spread averaging and
  birefringence for the oblique angles.
- Thickness from interference maxima and from the full model fit.
- Kramers–Kronig consistency analysis.
- Tabulated optical constants for common substrates and films (see `materials/`;
  each CSV carries the attribution of its source data in the header).

## Requirements

- Python 3.10+
- PyQt6 (PyQt5 also works — the code falls back automatically)
- numpy, scipy, matplotlib

Install the dependencies with:

    pip install -r requirements.txt

## Running

    python main.py

## Documentation

- `docs/D1_user_manual.pdf` — user manual
- `docs/D2_physics_math_reference.pdf` — physics & mathematical reference
- `docs/D3_procedures.pdf` — operating procedures

## License

Source-available under the **PolyForm Noncommercial License 1.0.0** (see `LICENSE`):
free to use, study, modify, and share for any **noncommercial** purpose — academic
research and education included — while **commercial use is reserved** to the rights
holder. This is *not* an OSI-approved open-source license.

Copyright 2026 **Università degli Studi di Cagliari**.
Author: **Guido Mula** (Dipartimento di Fisica, Università degli Studi di Cagliari).

## Citation

If you use OPHIRA in your work, please cite it — see `CITATION.cff` (GitHub and
Zenodo render it automatically), and cite the Zenodo DOI once the release is archived.
