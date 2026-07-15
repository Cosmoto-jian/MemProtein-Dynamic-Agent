#!/usr/bin/env python3
"""Download ALL OPM PDBs + query TM annotations, then filter flat+TM locally.

Phase 1: Parallel PDB download (~15 min, ~3 GB to data/raw/)
Phase 2: Parse Cα → compute η + structural features
Phase 3: Query OPM API for TM segment status
Phase 4: Filter: flat (η≤1) AND has TM → output candidate list
"""

import os, time, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from scipy.spatial import cKDTree

PDB_URL = "https://storage.googleapis.com/opm-assets/pdb/{pdb_id}.pdb"
API = "https://opm-back.cc.lehigh.edu/opm-backend/primary_structures"
RAW_DIR = "data/raw"
WORKERS = 16
TIMEOUT = 10

os.makedirs(RAW_DIR, exist_ok=True)

# ---- load PDB list ----
with open("data/all_gcs_pdbs.txt") as f:
    all_pdbs = sorted(f.read().strip().split(","))
print(f"Total PDBs in GCS: {len(all_pdbs)}")

# ---- Phase 1: download PDBs in parallel ----
def download_one(pid: str) -> str:
    """Download PDB to data/raw/. Returns pid on success, 'skip' if exists, 'fail' on error."""
    path = os.path.join(RAW_DIR, f"{pid}.pdb")
    if os.path.exists(path):
        return "skip"
    try:
        resp = requests.get(PDB_URL.format(pdb_id=pid), timeout=TIMEOUT)
        resp.raise_for_status()
        with open(path, "w") as f:
            f.write(resp.text)
        return "ok"
    except Exception:
        return "fail"

to_dl = [p for p in all_pdbs if not os.path.exists(os.path.join(RAW_DIR, f"{p}.pdb"))]
print(f"\nPhase 1: Downloading {len(to_dl)} PDBs ({len(all_pdbs)-len(to_dl)} already cached)")
if to_dl:
    dl_ok = dl_skip = dl_fail = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_one, p): p for p in to_dl}
        for fut in as_completed(futures):
            status = fut.result()
            if status == "ok": dl_ok += 1
            elif status == "skip": dl_skip += 1
            else: dl_fail += 1
            done = dl_ok + dl_skip + dl_fail
            if done % 500 == 0:
                elapsed = time.time() - start
                print(f"  {done}/{len(to_dl)} ({100*done/len(to_dl):.0f}%) "
                      f"{done/elapsed:.1f}/s ok={dl_ok} fail={dl_fail}", flush=True)
    print(f"Phase 1 done: ok={dl_ok} skip={dl_skip} fail={dl_fail}")

# ---- Phase 2+3: parse PDB + check TM in one pass ----
def process_one(pid: str) -> dict | None:
    """Parse PDB, compute η, check TM via API. Returns dict or None."""
    path = os.path.join(RAW_DIR, f"{pid}.pdb")
    if not os.path.exists(path):
        return None

    # Parse Cα
    coords = []
    try:
        with open(path) as fh:
            for line in fh:
                if line[:4] != "ATOM": continue
                if line[12:16].strip() != "CA" or line[17:20].strip() == "DUM": continue
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    except Exception:
        return {"pid": pid, "N": 0, "eta": float('nan'), "has_tm": False, "status": "bad_pdb"}

    N = len(coords)
    if N < 20:
        return {"pid": pid, "N": N, "eta": float('nan'), "has_tm": False, "status": "too_small"}
    c = np.array(coords, dtype=np.float64)
    cc = c - c.mean(axis=0)
    G = cc.T @ cc / N
    Lxy = float(np.sqrt(G[0, 0] + G[1, 1]))
    Lz = float(np.sqrt(G[2, 2]))
    eta = Lz / Lxy if Lxy > 1e-6 else float('nan')

    # ENM
    pairs = cKDTree(c).query_pairs(10.0, output_type="ndarray")
    E = len(pairs)
    if E > 0:
        d = c[pairs[:, 1]] - c[pairs[:, 0]]
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        d_unit = d / norms
        in_plane_frac = float(np.mean(d_unit[:, 0]**2 + d_unit[:, 1]**2))
    else:
        in_plane_frac = float('nan')

    # Check TM via API
    has_tm = False
    try:
        resp = requests.get(API, params={"search": pid, "page_size": 1}, timeout=TIMEOUT)
        resp.raise_for_status()
        objs = resp.json().get("objects", [])
        if objs and objs[0].get("subunit_segments", 0) > 0:
            has_tm = True
    except Exception:
        pass

    return {"pid": pid, "N": N, "E": E, "Lxy": Lxy, "Lz": Lz, "eta": eta,
            "in_plane_frac": in_plane_frac, "has_tm": has_tm, "spring_density": E/N if N>0 else 0,
            "status": "ok"}

print(f"\nPhase 2+3: Parsing PDBs + checking TM annotations ...")
all_processed = []
ok_count = err_count = flat_tm_count = start_t = 0
start_t = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(process_one, p): p for p in all_pdbs}
    for fut in as_completed(futures):
        r = fut.result()
        if r is None:
            err_count += 1; continue
        all_processed.append(r)
        ok_count += 1
        if r.get("eta", 2) <= 1.0 and r.get("has_tm"):
            flat_tm_count += 1
        if ok_count % 1000 == 0:
            elapsed = time.time() - start_t
            print(f"  {ok_count}/{len(all_pdbs)} ({100*ok_count/len(all_pdbs):.0f}%) "
                  f"{ok_count/elapsed:.1f}/s flat+TM={flat_tm_count}", flush=True)

print(f"Phase 2+3 done: {ok_count} processed, {err_count} errors")

# ---- Filter & save ----
flat_tm = [r for r in all_processed if r.get("eta", 2) <= 1.0 and r.get("has_tm")]
flat_no_tm = [r for r in all_processed if r.get("eta", 2) <= 1.0 and not r.get("has_tm")]
not_flat_tm = [r for r in all_processed if r.get("eta", 2) > 1.0 and r.get("has_tm")]

print(f"\n{'='*60}")
print(f"Filter Results")
print(f"{'='*60}")
print(f"Total processed:     {ok_count}")
print(f"Flat + TM (candidates): {len(flat_tm)}")
print(f"Flat, no TM:            {len(flat_no_tm)} (no OPM annotation)")
print(f"Not flat, has TM:       {len(not_flat_tm)}")

# Save comprehensive CSV
csv_path = "data/opm_full_catalog.csv"
with open(csv_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["pid","N","E","Lxy","Lz","eta","in_plane_frac","spring_density","has_tm","status"])
    writer.writeheader()
    for r in sorted(all_processed, key=lambda x: x.get("eta", 999)):
        writer.writerow(r)
print(f"\nSaved: {csv_path}")

# Save flat+TM candidate list for simulation
flat_tm_ids = ",".join(sorted(r["pid"] for r in flat_tm))
with open("data/flat_tm_candidates.txt", "w") as f:
    f.write(flat_tm_ids + "\n")
print(f"Flat+TM candidates: data/flat_tm_candidates.txt ({len(flat_tm)} proteins)")
print("DONE")
