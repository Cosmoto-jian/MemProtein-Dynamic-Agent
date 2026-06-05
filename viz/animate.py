#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM simulation result visualization tool
Supports frame-by-frame playback of HDF5-format simulation results
"""
import numpy as np
import pyvista as pv
import h5py
import os
import sys
import argparse

class FEMSequentialViewer:
    """FEM frame-by-frame viewer"""

    def __init__(self, hdf5_file, display_interval=50):
        """
        Initialize the viewer

        Parameters:
            hdf5_file: Path to the HDF5 data file
            display_interval: Display interval (show one frame every N calculation steps)
        """
        self.hdf5_file = hdf5_file
        self.display_interval = display_interval

        # Load data
        self._load_data()

        print(f"Data loaded: {self.total_frames} frames")
    
    def _load_data(self):
        """Load data"""
        print("Loading data file...")
        
        with h5py.File(self.hdf5_file, 'r') as f:
            timeseries = f['timeseries']
            
            # Get basic information
            self.total_calc_steps = timeseries['node_coords'].shape[0]
            self.elements = timeseries['element_connectivity'][0].copy()
            self.total_frames = (self.total_calc_steps + self.display_interval - 1) // self.display_interval
            
            # Preload node data for all displayed frames
            self.frames_data = []
            self.time_labels = []
            
            for frame_idx in range(self.total_frames):
                calc_step = min(frame_idx * self.display_interval, self.total_calc_steps - 1)
                
                # Load node coordinates (convert to meters)
                nodes = timeseries['node_coords'][calc_step] * 1e10
                self.frames_data.append(nodes.copy())
                
                # Create time label
                if 'time' in timeseries:
                    time_ps = timeseries['time'][calc_step] * 1e12
                    label = f"Time: {time_ps:.3f} ps\nCalculation step: {calc_step + 1}/{self.total_calc_steps}\nDisplay frame: {frame_idx + 1}/{self.total_frames}"
                else:
                    label = f"Display frame: {frame_idx + 1}/{self.total_frames}\nCalculation step: {calc_step + 1}/{self.total_calc_steps}"
                
                self.time_labels.append(label)
        
        print(f"Number of nodes: {len(self.frames_data[0])}, number of elements: {len(self.elements)}")
    
    def show_frame(self, frame_idx):
        """Display a single frame"""
        if not (0 <= frame_idx < self.total_frames):
            print(f"Invalid frame index: {frame_idx}")
            return
        
        print(f"\nDisplaying frame {frame_idx + 1}/{self.total_frames}")
        
        # Create a new plotter
        plotter = pv.Plotter(window_size=[2400, 1600], title=f"FEM帧查看器 - 帧{frame_idx + 1}")
        plotter.set_background('white')
        
        # Add axes
        plotter.add_axes(line_width=2)
        
        # Create mesh
        n_elements = len(self.elements)
        lines = np.empty(n_elements * 3, dtype=np.int32)
        lines[0::3] = 2  # Two points per line
        lines[1::3] = self.elements[:, 0]
        lines[2::3] = self.elements[:, 1]
        
        mesh = pv.PolyData()
        mesh.points = self.frames_data[frame_idx]
        mesh.lines = lines
        
        # Add nodes (blue points)
        plotter.add_points(
            self.frames_data[frame_idx],
            color='blue',
            point_size=10,
            render_points_as_spheres=True
        )
        
        # Add lines (black lines)
        plotter.add_mesh(
            mesh,
            color='black',
            line_width=1.0,
            style='wireframe'
        )
        
        # Add time display text
        plotter.add_text(
            self.time_labels[frame_idx],
            position='upper_left',
            font_size=14,
            color='black'
        )
        
        # Add usage instructions
        help_text = f"""Current: Frame {frame_idx + 1}/{self.total_frames}
    • Close the window to show the next frame
• ESC键退出
    • Display one frame every {self.display_interval} calculation steps"""
        
        plotter.add_text(
            help_text,
            position='lower_right',
            font_size=11,
            color='black'
        )
        
        # Set the camera and display
        plotter.camera_position = 'iso'
        plotter.reset_camera()
        
        print("Close the window to continue to the next frame...")
        plotter.show()
    
    def show_all_frames(self):
        """Display all frames one by one"""
        print("\n=== FEM Frame-by-Frame Viewer ===")
        print(f"Will display {self.total_frames} frames in sequence")
        print("The next frame will appear automatically after each window is closed")
        print("Press Ctrl+C to exit at any time\n")
        
        try:
            for frame_idx in range(self.total_frames):
                self.show_frame(frame_idx)
        except KeyboardInterrupt:
            print("\nInterrupted by user, exiting viewer")
        
        print("\nAll frames displayed!")

def main():
    """Main program"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FEM仿真结果可视化工具')
    parser.add_argument('hdf5_file', nargs='?', default=None,
                       help='Path to the HDF5 data file (optional, auto-search if not provided)')
    parser.add_argument('--interval', type=int, default=50,
                       help='Display interval (show one frame every N calculation steps, default 50)')
    parser.add_argument('--frame', type=int, default=None,
                       help='Show only the specified frame (optional)')
    args = parser.parse_args()

    print("=== FEM Frame-by-Frame Viewer ===")

    # Get the script directory and project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # VEND project root

    # Determine the HDF5 file
    if args.hdf5_file:
        # User specified a file
        hdf5_file = args.hdf5_file
        if not os.path.isabs(hdf5_file):
            # Relative path: try the current directory first, then the project root
            if not os.path.exists(hdf5_file):
                alt_path = os.path.join(project_root, hdf5_file)
                if os.path.exists(alt_path):
                    hdf5_file = alt_path

        if not os.path.exists(hdf5_file):
            print(f"Error: specified file does not exist: {hdf5_file}")
            sys.exit(1)

        print(f"Using specified data file: {hdf5_file}")
    else:
        # Automatically search for HDF5 files
        search_paths = [
            os.path.join(project_root, 'data', 'results'),  # default results dir
            '.',                    # Current working directory
            script_dir,             # viz directory
            project_root,           # Project root directory
        ]

        hdf5_files = []
        for search_path in search_paths:
            if os.path.exists(search_path):
                files = [os.path.join(search_path, f) for f in os.listdir(search_path) if f.endswith('.h5')]
                hdf5_files.extend(files)

        if not hdf5_files:
            print("Error: no HDF5 file found (.h5)")
            print(f"Searched the following directories:")
            for path in search_paths:
                print(f"  - {os.path.abspath(path)}")
            print("\nTip: You can specify a file like this:")
            print(f"  python {os.path.basename(__file__)} <文件路径>")
            sys.exit(1)

        # Use the most recently modified HDF5 file
        hdf5_file = max(hdf5_files, key=os.path.getmtime)
        print(f"Found {len(hdf5_files)} HDF5 files")
        print(f"Using the latest data file: {hdf5_file}")
    
    try:
        # Create and run the frame-by-frame viewer
        viewer = FEMSequentialViewer(hdf5_file, display_interval=args.interval)

        # If a single frame is specified, show only that frame
        if args.frame is not None:
            if 0 <= args.frame < viewer.total_frames:
                print(f"\nShowing only frame {args.frame}")
                viewer.show_frame(args.frame)
            else:
                print(f"Error: frame index {args.frame} out of range (0-{viewer.total_frames-1})")
                sys.exit(1)
        else:
            # Show all frames
            viewer.show_all_frames()

    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
