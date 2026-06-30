#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPM (Orientations of Proteins in Membranes) data access.

Sources:
    PDB files:  https://storage.googleapis.com/opm-assets/pdb/<pdbid>.pdb
    Metadata:   https://opm-back.cc.lehigh.edu/opm-backend/

The metadata API is a Rails JSON API (the same backend that powers
opm.phar.umich.edu).  We use it to fetch per-chain transmembrane-segment
annotations and write them as text files that ``parse_opm_tm_text``
(in preprocess.py) already understands.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import requests

OPM_API = "https://opm-back.cc.lehigh.edu/opm-backend"
OPM_PDB_URL = "https://storage.googleapis.com/opm-assets/pdb/{pdb_id}.pdb"

# Be polite — small delay between API calls.
_API_DELAY = 0.15


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, out_path: str, *, overwrite: bool = False) -> str:
    """Download the OPM-oriented PDB for *pdb_id* and write it to *out_path*.

    Returns *out_path*.  Existing files are skipped unless *overwrite* is
    ``True``.
    """
    if os.path.exists(out_path) and not overwrite:
        return out_path

    url = OPM_PDB_URL.format(pdb_id=pdb_id.lower())
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path


def fetch_tm_annotation(pdb_id: str) -> str:
    """Return the transmembrane-segment text for *pdb_id* (the same format
    that the parser in ``preprocess.parse_opm_tm_text`` expects).

    One line per chain::

        A - Tilt: 22 - TM segments: 1( 581- 599), 2( 610- 629), ...

    If the protein has no TM annotation, an empty string is returned.
    """
    protein = _get_protein_detail(pdb_id)
    if protein is None:
        raise ValueError(f"Protein {pdb_id!r} not found in OPM database")

    subunits = protein.get("subunits") or []
    if not subunits:
        return ""

    lines: List[str] = []
    for sub in subunits:
        chain = sub.get("protein_letter", "")
        segments = sub.get("segment", "")
        tilt = sub.get("tilt", "")
        if not chain or not segments.strip():
            continue
        tilt_part = f"Tilt: {tilt} - " if tilt and tilt.strip() else ""
        lines.append(f"{chain} - {tilt_part}TM segments: {segments.strip()}")

    return "\n".join(lines) + ("\n" if lines else "")


def ensure_raw_inputs(
    pdb_ids: List[str], raw_dir: str = "data/raw", *, overwrite: bool = False
) -> Dict[str, Dict[str, str]]:
    """Make sure every PDB in *pdb_ids* has a ``.pdb`` and ``_tm.txt`` file
    inside *raw_dir*.  Existing files are skipped unless *overwrite* is set.

    Returns a mapping::

        {pdb_id: {"pdb": "/abs/path/to/1bl8.pdb",
                  "tm":  "/abs/path/to/1bl8_tm.txt"}}
    """
    raw_dir = os.path.abspath(raw_dir)
    result: Dict[str, Dict[str, str]] = {}

    for pid in pdb_ids:
        pid_l = pid.lower()
        pdb_path = os.path.join(raw_dir, f"{pid_l}.pdb")
        tm_path = os.path.join(raw_dir, f"{pid_l}_tm.txt")

        # PDB
        download_pdb(pid_l, pdb_path, overwrite=overwrite)
        print(f"[{pid}] PDB -> {pdb_path}")

        # TM annotation
        if not os.path.exists(tm_path) or overwrite:
            text = fetch_tm_annotation(pid_l)
            if text:
                with open(tm_path, "w") as fh:
                    fh.write(text)
                print(f"[{pid}] TM  -> {tm_path}")
            else:
                print(f"[{pid}] (no TM segments — skipping tm.txt)")

        result[pid] = {"pdb": pdb_path, "tm": tm_path}
        if pid != pdb_ids[-1]:
            time.sleep(_API_DELAY)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_protein_detail(pdb_id: str) -> dict | None:
    """Retrieve the full protein detail from the OPM API (includes embedded
    ``subunits`` with ``segment`` strings)."""
    numeric_id = _search_protein(pdb_id)
    if numeric_id is None:
        return None

    url = f"{OPM_API}/primary_structures/{numeric_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _search_protein(pdb_id: str) -> int | None:
    """Look up the numeric OPM id for a PDB identifier.

    The list endpoint does **not** embed TM segments; only the detail
    endpoint (by numeric id) does, so this two-step dance is necessary.
    """
    url = f"{OPM_API}/primary_structures"
    resp = requests.get(url, params={"search": pdb_id.lower(), "page_size": 1},
                        timeout=30)
    resp.raise_for_status()
    data = resp.json()
    objects: List[dict] = data.get("objects", [])
    if not objects:
        return None
    return int(objects[0]["id"])
