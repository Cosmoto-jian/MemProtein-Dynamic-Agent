# This program converts protein files from OPM database to coarse-grained files (Cα residues)
# Output: piezo-cg.pdb

import numpy as np
from Bio import PDB

def parse_pdb(filename):
    """Parse PDB file and return coordinates of all Cα atoms"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", filename)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_id() == 'CA':
                        ca_coords.append(atom.coord)
    return ca_coords

def coarse_grain_ca_grouped(ca_coords, ca_per_group=1):
    """每ca_per_group个Cα原子分一组，计算组内Cα原子的平均坐标"""
    grouped_coords = []
    for i in range(0, len(ca_coords), ca_per_group):
        group = ca_coords[i:i + ca_per_group]
        group = np.array(group)
        if len(group) > 0:
            centroid = np.mean(group, axis=0)
            grouped_coords.append(centroid)
    return grouped_coords

def write_coarse_grained_pdb(grouped_coords, output_filename):
    """将粗粒化后的坐标写入PDB文件"""
    with open(output_filename, 'w') as f:
        for i, coord in enumerate(grouped_coords):
            # 在PDB格式中写入每个粗粒化粒子的坐标，标记为“CA”
            f.write(f"ATOM  {i+1:5d}  CA  CG  A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}\n")

if __name__ == "__main__":
    pdb_filename = r"C:\Users\ADMIN\Desktop\4ym8.pdb"
    ca_coords = parse_pdb(pdb_filename)

    ca_per_group = 1
    grouped_coords = coarse_grain_ca_grouped(ca_coords, ca_per_group=ca_per_group)

    output_filename = r"C:\Users\ADMIN\Desktop\0527\1-pie-cg.pdb"
    write_coarse_grained_pdb(grouped_coords, output_filename)
    print(f"粗粒化PDB文件已保存为 {output_filename}")
