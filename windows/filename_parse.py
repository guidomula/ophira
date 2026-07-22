"""filename_parse.py — sample/angle recognition from file names (LOAD all).

Expected file-naming convention:
  - the SAMPLE NAME is always `letters + 3-4 digits` at the start (sometimes with a
    stray space between letters and digits, e.g. "Na 1178");
  - the ANGLE is a number 8/20/40/60 (sometimes with °, deg or nothing) and is
    usually AT THE END, with possible extra info in between (ox, DHN, analysis no., ...).

Robustness:
  - the sample-name region is discarded → no collisions with its own digits
    (e.g. the 8 or 20 embedded in the sample-name digits);
  - the angle number is matched with DIGIT BOUNDARIES (not inside a longer number)
    and the candidate with a °/deg suffix is preferred, then the rightmost one.
"""
import os
import re
from collections import defaultdict

_RE_ANGLE  = re.compile(r"(?<!\d)(8|20|40|60)\s*(°|deg|d)?(?!\w)", re.IGNORECASE)
_RE_SAMPLE = re.compile(r"^\s*([A-Za-z]+)\s*(\d{3,4})")

DATA_EXT = ('.asc', '.csv', '.txt', '.dat')


def sample_name(fname):
    """Sample name = letters + 3-4 leading digits (stray space removed)."""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = _RE_SAMPLE.match(base)
    if m:
        return m.group(1) + m.group(2)
    return re.split(r"[ _\-.]+", base)[0]


def detect_angle(fname):
    """Angle (8/20/40/60) from the file name, or None if unrecognizable."""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = _RE_SAMPLE.match(base)
    rest = base[m.end():] if m else base           # drop the sample-name region
    cands = [(int(x.group(1)), bool(x.group(2)), x.start())
             for x in _RE_ANGLE.finditer(rest)]
    if not cands:
        return None
    suff = [c for c in cands if c[1]]              # prefer °/deg
    pool = suff if suff else cands
    pool.sort(key=lambda c: c[2])                  # angle at the end → rightmost
    return pool[-1][0]


def list_angle_files(folder):
    """All data files with a recognizable angle in `folder`, as a list of
    (sample, angle, path) sorted by sample and angle. No dedup: each file is one
    entry → the checkbox UI leaves control to the user.
    """
    try:
        files = sorted(f for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in DATA_EXT)
    except OSError:
        return []
    out = []
    for f in files:
        a = detect_angle(f)
        if a is None:
            continue
        out.append((sample_name(f), a, os.path.join(folder, f)))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def default_sample(files):
    """Sample with the most angles (to pre-check in the list), '' if empty.
    `files` = output of list_angle_files."""
    by = defaultdict(set)
    for smp, a, _p in files:
        by[smp].add(a)
    return max(by, key=lambda s: len(by[s])) if by else ""
