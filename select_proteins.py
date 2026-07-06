#!/usr/bin/env python3
"""Select ~100 eukaryotic plasma membrane proteins with TM segments
by sampling from the GCS PDB listing and validating via the OPM API."""

import random
import time
import requests

API = "https://opm-back.cc.lehigh.edu/opm-backend/primary_structures"
EXISTING = {"1bl8", "1c3w", "1j4n", "1u19", "2oar", "2oau",
            "2rh1", "4bw5", "6b3r", "6mgv", "6w7b", "7aa5"}
N_TARGET = 100
BATCH_SIZE = 300
random.seed(42)

# Load all PDBs from GCS
with open("data/all_gcs_pdbs.txt") as f:
    all_pdbs = f.read().strip().split(",")
print(f"GCS PDB pool: {len(all_pdbs)}")

# Remove existing
pool = [p for p in all_pdbs if p not in EXISTING]

candidates = []
sampled: set = set()

while len(candidates) < N_TARGET - len(EXISTING) and len(sampled) < len(pool):
    # Sample a batch
    batch = [p for p in pool if p not in sampled]
    random.shuffle(batch)
    batch = batch[:BATCH_SIZE]
    sampled.update(batch)

    print(f"\nValidating {len(batch)} PDBs via API (total sampled: {len(sampled)}) ...")

    for i, pid in enumerate(batch):
        try:
            resp = requests.get(API, params={"search": pid, "page_size": 1}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("objects"):
                o = data["objects"][0]
                if o.get("membrane_id") == 4 and o.get("subunit_segments", 0) > 0:
                    candidates.append(pid)
                    print(f"  ✓ {pid}  ({len(candidates)} so far)  {o.get('name','')[:50]}")
        except Exception:
            pass
        time.sleep(0.08)  # Be gentle to the API

        if len(candidates) >= N_TARGET - len(EXISTING):
            break

    print(f"  Candidates: {len(candidates)}")

selected_new = candidates[:N_TARGET - len(EXISTING)]
all_ids = selected_new + sorted(EXISTING)
random.shuffle(all_ids)

ids_str = ",".join(all_ids)
print(f"\n{'='*60}")
print(f"Selected {len(all_ids)} proteins ({len(selected_new)} new + {len(EXISTING)} existing)")
print(f"\n--pdb-ids {ids_str}")

with open("data/selected_100.txt", "w") as f:
    f.write(ids_str + "\n")
print(f"\n(saved to data/selected_100.txt)")
