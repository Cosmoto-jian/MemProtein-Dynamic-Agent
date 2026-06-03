"""
Shared utility functions module
Contains 3D truss element force calculation and ramp functions for general use
"""

import numpy as np
from scipy.io import loadmat


def truss3d_force(x1a, x1t, x2a, x2t, E, A, f):
    """
    Calculate internal force of 3D truss element
    
    Parameters:
    x1a, x1t: Coordinates of node 1 at times ta and t
    x2a, x2t: Coordinates of node 2 at times ta and t
    E: Elastic modulus
    A: Cross-sectional area
    f: Previous element internal force
    
    Returns:
    ef1, ef2: Force vectors at nodes 1 and 2
    f_new: Updated element internal force
    """
    # Calculate element length
    lt = np.sqrt(np.sum((x2t - x1t)**2))  # Element length at time t
    la = np.sqrt(np.sum((x2a - x1a)**2))  # Element length at time ta
    
    # Calculate element internal force
    if lt >= 2:
        E = 0
        f_new = 0
    else:
        f_new = f + E * A * (lt - la) / la
    
    # Element direction vector
    e_vector = (x2t - x1t) / lt
    
    # Nodal forces (global coordinate system)
    ef2 = f_new * e_vector  # Node 2
    ef1 = -ef2              # Node 1
    
    return ef1, ef2, f_new


def ramp_step(start, stop, time):
    """
    Ramp-step loading function
    
    Parameters:
    start: Ramp start time
    stop: Ramp end time
    time: Current time
    
    Returns:
    Pt: Loading factor (between 0 and 1)
    """
    if time >= stop:
        return 1.0
    elif time <= start:
        return 0.0
    else:
        return (time - start) / (stop - start)


def load_matlab_data(filename):
    """
    Load MATLAB .mat file
    
    Parameters:
    filename: Path to .mat file
    
    Returns:
    data: Dictionary of loaded data
    """
    return loadmat(filename)


def load_target_nodes(filename):
    """
    Load target node IDs from file
    
    Parameters:
    filename: Text file containing node IDs
    
    Returns:
    ids: Array of node IDs
    """
    return np.loadtxt(filename)


def load_mass_data(filename):
    """
    Load mass data from file
    
    Parameters:
    filename: Text file containing mass data
    
    Returns:
    mass: Array of mass values
    """
    return np.loadtxt(filename)


def read_model_file(filename):
    """
    Read model file
    
    Parameters:
    filename: Path to MODEL.txt file
    
    Returns:
    nodes: Node data [NodeID, X, Y, Z]
    elements: Element data [ElementID, Node1, Node2]
    constraints: Constraint data [NodeID, X, Y, theta]
    """
    with open(filename, 'r') as f:
        # Read node data
        node_number = int(f.readline().strip())
        nodes = np.zeros((node_number, 4))
        for i in range(node_number):
            line = f.readline().strip().split()
            nodes[i] = [float(x) for x in line]
        
        # Read element data
        element_number = int(f.readline().strip())
        elements = np.zeros((element_number, 3), dtype=int)
        for i in range(element_number):
            line = f.readline().strip().split()
            elements[i] = [int(x) for x in line]
        
        # Read constraint data
        constraint_node_number = int(f.readline().strip())
        constraints = np.zeros((constraint_node_number, 4), dtype=int)
        for i in range(constraint_node_number):
            line = f.readline().strip().split()
            constraints[i] = [int(x) for x in line]
    
    return nodes, elements, constraints
# --- New functions ---
def calculate_element_stiffness(node_p_idx, node_q_idx, current_coords, initial_coords, E, A):
    """
    Calculate the stiffness contribution of a single 3D bar element to a node using analytical differentiation.

    Parameters:
        node_p_idx (int): 0-based index of the target node.
        node_q_idx (int): 0-based index of the other node in the element.
        current_coords (np.array): Current coordinates of all nodes in the model (N x 3).
        initial_coords (np.array): Initial coordinates of all nodes in the model (N x 3).
        E (float): Elastic modulus.
        A (float): Cross-sectional area.

    Returns:
        np.array: 3x3 element stiffness contribution matrix (k_pp).
    """
    X_p = current_coords[node_p_idx]
    X_q = current_coords[node_q_idx]

    # Ensure consistent element vector direction, with p as the element starting point
    L_vec = X_q - X_p
    l = np.linalg.norm(L_vec)  # Current length

    if l < 1e-9:
        return np.zeros((3, 3)) # Avoid division by zero

    # Initial length l0
    X_p_initial = initial_coords[node_p_idx]
    X_q_initial = initial_coords[node_q_idx]
    l0 = np.linalg.norm(X_q_initial - X_p_initial)

    if l0 < 1e-9:
        return np.zeros((3, 3))

    # Element axial stiffness k = EA/l0
    k = (E * A) / l0
    
    # Element internal force T = k * (l - l0)
    T = k * (l - l0)
    
    # Material tangent modulus dT/dl (for linear material, dT/dl = k)
    dT_dl = k
    
    # Unit direction vector n
    n = L_vec / l
    
    # Outer product of direction vector n*n^T
    nnT = np.outer(n, n)
    
    # 3x3 identity matrix
    I = np.identity(3)
    
    # Relevant block of material stiffness matrix k_mat
    k_mat_pp = dT_dl * nnT
    
    # Relevant block of geometric stiffness matrix k_geo
    k_geo_pp = (T / l) * (nnT - I)
    
    # Stiffness contribution at node p: k_pp = k_mat_pp + k_geo_pp
    k_pp_contribution = k_mat_pp + k_geo_pp

    return k_pp_contribution
