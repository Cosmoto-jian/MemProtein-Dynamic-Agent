# This program builds an elastic network model unit connectivity based on the coarse-grained file Raw/piezo-cg.pdb
# Output the connection units of Raw/MODEL.txt

import numpy as np
import pandas as pd
import re

def read_coords_from_simple_pdb(pdb_filename):
    coords = []
    with open(pdb_filename, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
    return np.array(coords)


def build_gnm_kirchhoff(coords, cutoff=32.0):
    n = len(coords)
    kirchhoff = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                kirchhoff[i, j] = kirchhoff[j, i] = -1
    for i in range(n):
        kirchhoff[i, i] = -np.sum(kirchhoff[i])
    return kirchhoff

if __name__ == "__main__":
    pdb_filename = r"C:\Users\ADMIN\Desktop\0523\20-pie-cg.pdb"  # Replace with your actual path
    coords = read_coords_from_simple_pdb(pdb_filename)
    print(f"Total nodes read: {coords.shape[0]}")

    cutoff = 32.0
    kirchhoff = build_gnm_kirchhoff(coords, cutoff)

    # Output each connection only once (i < j)
    connection_pairs = []
    n = kirchhoff.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if kirchhoff[i, j] == -1:
                connection_pairs.append({'Connection': f"{i+1}-{j+1}"})

    df_conn = pd.DataFrame(connection_pairs)
    df_conn.to_excel('Kirchhoff_20-pie-cg.xlsx', index=False)
    print("Excel file exported: Kirchhoff_Connection_Pairs.xlsx")
