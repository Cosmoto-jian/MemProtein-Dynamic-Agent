# This script extracts residue numbers of membrane proteins on the membrane (coarse-grained residue numbers)
# Extract Cα atoms (representing atoms after coarse-graining) in transmembrane segments from OPM-aligned PDB files
# The output residue numbers will be used to generate Raw/targetNode.txt (force-loading target node ID list)

from Bio.PDB import PDBParser
import csv

# Read PDB file
pdb_file =  r"C:\Users\79249\Desktop\6b3r.pdb" # Make sure to use OPM-aligned PDB file
output_csv = "tm_CA_atoms2.csv"
parser = PDBParser(QUIET=True)
structure = parser.get_structure('PIEZO1', pdb_file)

# Define residue ranges for transmembrane segments
tm_segments = [
    (581, 599), (610, 629), (631, 652), (686, 711), (790, 823),
    (829, 844), (845, 866), (924, 946), (977, 1009), (1010, 1027),
    (1028, 1050), (1091, 1112), (1152, 1179), (1180, 1199),
    (1205, 1229), (1280, 1298), (1688, 1705), (1706, 1721),
    (1727, 1747), (1781, 1801), (1977, 1996), (2018, 2039),
    (2043, 2064), (2077, 2098), (2187, 2209), (2466, 2487)
]

# Extract and save Cα atoms in transmembrane regions
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "chain", "resname", "resid", "x", "y", "z"])
    for model in structure:
        for chain in model:
            if chain.id not in ['A', 'C', 'E']:
                continue
            for residue in chain:
                res_id = residue.get_id()[1]
                for start, end in tm_segments:
                    if start <= res_id <= end:
                        if 'CA' in residue:
                            atom = residue['CA']
                            x, y, z = atom.get_coord()
                            writer.writerow([
                                model.id,
                                chain.id,
                                residue.get_resname(),
                                res_id,
                                round(x, 3), round(y, 3), round(z, 3)
                            ])
                        break  # If a segment matches, break out of inner loop
print(f"Cα atom coordinates in transmembrane region saved to {output_csv}")
