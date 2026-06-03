#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Interface (model_interface.py)

Features:
- Load static node coordinates and element connectivity data from MODEL.txt.
- Provide a clear interface for PyVista visualization programs to call.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

# Define the base data path relative to the current file
BASE_DATA_PATH = Path(__file__).parent.resolve()

class ModelInterface:
    """
    A concise model data interface class.
    Responsible for loading and providing static model data, specifically for visualization.
    """
    
    def __init__(self, model_file: Path = BASE_DATA_PATH / 'MODEL.txt'):
        """
        Initialize the interface and load model data.
        """
        self.model_file = model_file
        
        # Internal storage
        self._node_coords: np.ndarray = np.array([])
        self._elements: np.ndarray = np.array([])
        self._node_id_to_idx_map: Dict[int, int] = {}
        
        # Load data
        self._load_model_data()

    def _load_model_data(self):
        """
        Internal method to load node and element data from MODEL.txt file.
        """
        print(f"Loading model data from '{self.model_file}'...")
        try:
            with open(self.model_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Model file '{self.model_file}' not found.")
            return

        # 1. Read node data
        n_nodes = int(lines[0].strip())
        
        temp_node_coords: List[List[float]] = []
        for i in range(1, n_nodes + 1):
            parts = lines[i].strip().split()
            node_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            # Store coordinates
            temp_node_coords.append([x, y, z])
            # Create a mapping from node ID to its position (index) in the array
            self._node_id_to_idx_map[node_id] = len(temp_node_coords) - 1
        
        self._node_coords = np.array(temp_node_coords, dtype=np.float64)
        print(f"  Loaded {self.n_nodes} nodes.")
        
        # 2. Read element data
        element_start_line = n_nodes + 1
        n_elements = int(lines[element_start_line].strip())
        
        temp_elements: List[List[int]] = []
        for i in range(element_start_line + 1, element_start_line + 1 + n_elements):
            parts = lines[i].strip().split()
            # [Important fix] Store raw node IDs directly, not converted to indices
            node1_id = int(parts[1])
            node2_id = int(parts[2])
            temp_elements.append([node1_id, node2_id])
        
        self._elements = np.array(temp_elements, dtype=np.int32)
        print(f"  Loaded {self.n_elements} elements.")

    def get_indices_from_ids(self, node_ids: List[int] or np.ndarray) -> np.ndarray:
        """
        Given a list of raw node IDs, get their 0-based index list in the coordinate array.
        Any IDs not found in the model will be ignored.
        
        Args:
            node_ids: List or NumPy array of raw node IDs, e.g., [4424, 4425, 9999]
            
        Returns:
            NumPy array of corresponding 0-based indices. e.g., np.array([123, 124])
        """
        indices = [self._node_id_to_idx_map.get(node_id) for node_id in node_ids]
        # Filter out unfound nodes (their index is None)
        valid_indices = [idx for idx in indices if idx is not None]
        return np.array(valid_indices, dtype=int)

    # --- Data Access Interface ---
    
    @property
    def node_coords(self) -> np.ndarray:
        """Get node coordinate array (n_nodes, 3)"""
        return self._node_coords
    
    @property
    def elements(self) -> np.ndarray:
        """Get element connectivity array (n_elements, 2), containing raw node IDs."""
        return self._elements
    
    @property
    def n_nodes(self) -> int:
        """Get total number of nodes"""
        return len(self._node_coords)
    
    @property
    def n_elements(self) -> int:
        """Get total number of elements"""
        return len(self._elements)
