# This program converts protein files from the OPM database to coarse-grained
# files (one bead per C-alpha residue).
# Output: a coarse-grained PDB that PRESERVES each residue's real chain id and
# residue number, so downstream steps (e.g. matching OPM TM segments) work
# correctly across all chains.
#
# Fix history: the previous version hard-coded every output atom to chain "A"
# and residue number 1, losing the fact that residues belong to different
# chains (A/C/E for Piezo1). This version keeps the true chain id and resid.

import argparse

import numpy as np


def parse_pdb_ca(filename, include_hetatm=True):
    """Read C-alpha atoms in file order, keeping chain/resid/resname.

    Returns a list of records: (chain, resid, resname, coord[np.ndarray(3)]).
    DUM (OPM membrane dummy) atoms are always skipped. HETATM C-alpha atoms
    (e.g. OPM "UNK" fragments) are kept by default.
    """
    records = []
    with open(filename, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if rec == "HETATM" and not include_hetatm:
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip()
            if resname == "DUM":
                continue
            chain = line[21]
            resid = int(line[22:26])
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            records.append((chain, resid, resname, coord))
    return records


def coarse_grain_ca_grouped(records, ca_per_group=1):
    """Group every `ca_per_group` consecutive C-alpha atoms WITHIN the same
    chain and average their coordinates.

    With ca_per_group=1 (default) this is a 1:1 mapping that keeps every real
    chain id and residue number. With larger groups, each bead inherits the
    chain id and residue number of the first residue in its group, and grouping
    never crosses a chain boundary.

    Returns a list of beads: (chain, resid, resname, coord).
    """
    beads = []
    i = 0
    n = len(records)
    while i < n:
        chain = records[i][0]
        # collect up to ca_per_group residues, stopping at a chain change
        group = []
        while i < n and len(group) < ca_per_group and records[i][0] == chain:
            group.append(records[i])
            i += 1
        coords = np.array([g[3] for g in group])
        centroid = coords.mean(axis=0)
        beads.append((group[0][0], group[0][1], group[0][2], centroid))
    return beads


def write_coarse_grained_pdb(beads, output_filename):
    """Write beads as standard-column PDB ATOM records with correct chain id
    and residue number. Column layout matches what the downstream parsers read
    (chain at col 22, resSeq at cols 23-26, coords at cols 31-54)."""
    with open(output_filename, "w") as f:
        for i, (chain, resid, resname, coord) in enumerate(beads, start=1):
            x, y, z = coord
            f.write(
                f"ATOM  {i:>5d}  CA  {resname:>3s} {chain:1s}{resid:>4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
        f.write("END\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Coarse-grain a PDB to one C-alpha bead per residue (chain-aware).")
    ap.add_argument("--pdb", required=True, help="Input PDB (e.g. an OPM-aligned all-atom structure)")
    ap.add_argument("--out", default="Raw/piezo-cg.pdb", help="Output coarse-grained PDB")
    ap.add_argument("--group", type=int, default=1, help="C-alpha atoms per coarse-grained bead (default 1)")
    ap.add_argument("--drop-hetatm", action="store_true", help="Exclude HETATM C-alpha (e.g. OPM UNK fragments)")
    args = ap.parse_args()

    records = parse_pdb_ca(args.pdb, include_hetatm=not args.drop_hetatm)
    beads = coarse_grain_ca_grouped(records, ca_per_group=args.group)

    chains = {}
    for ch, *_ in beads:
        chains[ch] = chains.get(ch, 0) + 1

    write_coarse_grained_pdb(beads, args.out)
    print(f"Read {len(records)} C-alpha atoms, wrote {len(beads)} coarse-grained beads to {args.out}")
    print("Beads per chain: " + ", ".join(f"{c}={chains[c]}" for c in sorted(chains)))
