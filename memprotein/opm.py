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
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Set

import requests

OPM_API = "https://opm-back.cc.lehigh.edu/opm-backend"
OPM_PDB_URL = "https://storage.googleapis.com/opm-assets/pdb/{pdb_id}.pdb"

# Be polite — small delay between API calls.
_API_DELAY = 0.15


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    pdb_id: str
    status: str = "ok"                  # "ok", "warn", "skip"
    reason: str = ""
    pdb_path: str = ""
    tm_path: str = ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, out_path: str, *, overwrite: bool = False) -> str | None:
    """Download the OPM-oriented PDB for *pdb_id* and write it to *out_path*.

    Returns *out_path* on success, ``None`` on failure.
    Existing files are skipped unless *overwrite* is ``True``.
    """
    if os.path.exists(out_path) and not overwrite:
        return out_path

    url = OPM_PDB_URL.format(pdb_id=pdb_id.lower())
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path


def fetch_tm_annotation(pdb_id: str) -> str | None:
    """Return the transmembrane-segment text for *pdb_id*, or ``None`` if the
    protein is not in OPM or has no TM segments.

    Output format (compatible with ``preprocess.parse_opm_tm_text``)::

        A - Tilt: 22 - TM segments: 1( 581- 599), 2( 610- 629), ...
    """
    protein = _get_protein_detail(pdb_id)
    if protein is None:
        return None

    subunits = protein.get("subunits") or []
    if not subunits:
        return None

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
) -> List[FetchResult]:
    """Download PDB + TM annotation for every *pdb_id* into *raw_dir*.

    Existing files are skipped unless *overwrite* is set.  Each protein gets
    a health check after download (chain consistency, membrane coverage).

    Returns a list of ``FetchResult`` — one per input *pdb_id*:

    * ``ok``   — ready to simulate
    * ``warn`` — usable but with a warning (e.g. chain-id mismatch);
                 simulation will try but results may be incomplete
    * ``skip`` — excluded; the reason is in ``.reason``
    """
    raw_dir = os.path.abspath(raw_dir)
    results: List[FetchResult] = []

    for i, pid in enumerate(pdb_ids):
        pid_l = pid.lower()
        pdb_path = os.path.join(raw_dir, f"{pid_l}.pdb")
        tm_path = os.path.join(raw_dir, f"{pid_l}_tm.txt")

        # ----   PDB  ----
        dl = download_pdb(pid_l, pdb_path, overwrite=overwrite)
        if dl is None:
            results.append(FetchResult(pid, "skip", "PDB download failed"))
            print(f"[{pid}] SKIP — PDB download failed")
            continue
        print(f"[{pid}] PDB  -> {pdb_path}")

        # ----   TM annotation  ----
        if not os.path.exists(tm_path) or overwrite:
            try:
                text = fetch_tm_annotation(pid_l)
            except requests.RequestException:
                results.append(FetchResult(pid, "skip", "API unreachable"))
                print(f"[{pid}] SKIP — OPM API unreachable")
                _maybe_sleep(i, pdb_ids)
                continue

            if text is None:
                results.append(FetchResult(pid, "skip",
                                           "not in OPM or no TM annotation"))
                print(f"[{pid}] SKIP — no transmembrane annotation in OPM")
                _maybe_sleep(i, pdb_ids)
                continue

            if not text.strip():
                results.append(FetchResult(pid, "skip", "no TM segments"))
                print(f"[{pid}] SKIP — no TM segments")
                _maybe_sleep(i, pdb_ids)
                continue

            with open(tm_path, "w") as fh:
                fh.write(text)
            print(f"[{pid}] TM   -> {tm_path}")

        # ----   Health check  ----
        status, reason = _validate(pdb_path, tm_path)
        results.append(FetchResult(pid, status, reason, pdb_path, tm_path))
        if status == "warn":
            print(f"[{pid}] WARN — {reason}")
        elif status == "skip":
            print(f"[{pid}] SKIP — {reason}")
        else:
            print(f"[{pid}] OK")

        _maybe_sleep(i, pdb_ids)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regexes for lightweight PDB parsing (avoid import from .preprocess to keep
# this module standalone).
_PDB_CA_RE = re.compile(r"^(?:ATOM|HETATM).{7}CA .{4}[A-Z]"  # skip DUM
                        r"(.{5})"                              # residue number
                        r".{5}(.)")                             # chain
_TM_HEADER_RE = re.compile(r"^\s*([A-Za-z0-9])\s*-.*?TM segments\s*:(.*)$")


def _extract_pdb_chains(pdb_path: str) -> Set[str]:
    """Return the set of chain IDs that have at least one C-alpha atom.

    Excludes HETATM records (lipids, ligands, UNK fragments) to match the
    ATOM-only default of ``preprocess.parse_pdb_ca``."""
    chains: Set[str] = set()
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line[:4] != "ATOM":          # only ATOM records (ignore HETATM)
                    continue
                if line[12:16].strip() != "CA" or line[17:20].strip() == "DUM":
                    continue
                chains.add(line[21])
    except OSError:
        pass
    return chains


def _extract_tm_chains(tm_path: str) -> Set[str]:
    """Return the set of chain IDs mentioned in a tm.txt file."""
    chains: Set[str] = set()
    try:
        with open(tm_path) as fh:
            for line in fh:
                m = _TM_HEADER_RE.match(line)
                if m:
                    chains.add(m.group(1))
    except OSError:
        pass
    return chains


def _validate(pdb_path: str, tm_path: str) -> tuple[str, str]:
    """Check consistency between a PDB file and its tm.txt.

    Returns ``(status, reason)``:
        ok   — chains match, everything consistent
        warn — chains partially overlap (possible renaming)
        skip — zero overlap (annotation is for a different assembly)
    """
    pdb_chains = _extract_pdb_chains(pdb_path)
    tm_chains = _extract_tm_chains(tm_path)

    if not pdb_chains:
        return "skip", "PDB contains no C-alpha atoms"
    if not tm_chains:
        return "skip", "tm.txt contains no chain annotations"

    common = pdb_chains & tm_chains
    pdb_only = pdb_chains - tm_chains
    tm_only = tm_chains - pdb_chains

    if not common:
        msg = (f"chain ID mismatch: PDB has {sorted(pdb_chains)}, "
               f"tm.txt has {sorted(tm_chains)} — "
               "likely different assembly numbering; run without --fetch to keep current tm.txt")
        return "skip", msg

    if tm_only:
        msg = (f"some API chains not in PDB: {sorted(tm_only)} "
               f"(PDB chains: {sorted(pdb_chains)})")
        return "warn", msg
    if pdb_only:
        msg = (f"some PDB chains not in TM annotation: {sorted(pdb_only)}")
        return "warn", msg

    return "ok", ""


def _maybe_sleep(idx: int, ids: list) -> None:
    """Sleep between API calls unless this is the last one."""
    if idx < len(ids) - 1:
        time.sleep(_API_DELAY)


def _get_protein_detail(pdb_id: str) -> dict | None:
    """Retrieve the full protein detail from the OPM API (includes embedded
    ``subunits`` with ``segment`` strings).  Returns ``None`` on any error."""
    numeric_id = _search_protein(pdb_id)
    if numeric_id is None:
        return None

    try:
        resp = requests.get(f"{OPM_API}/primary_structures/{numeric_id}",
                            timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return resp.json()


def _search_protein(pdb_id: str) -> int | None:
    """Look up the numeric OPM id for a PDB identifier.

    The list endpoint does **not** embed TM segments; only the detail
    endpoint (by numeric id) does, so this two-step dance is necessary.
    """
    try:
        resp = requests.get(
            f"{OPM_API}/primary_structures",
            params={"search": pdb_id.lower(), "page_size": 1},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    data = resp.json()
    objects: List[dict] = data.get("objects", [])
    if not objects:
        return None
    return int(objects[0]["id"])
