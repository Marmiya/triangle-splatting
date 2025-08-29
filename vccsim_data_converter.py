#!/usr/bin/env python3
"""
VCCSim Data Converter

Utility script for converting VCCSim data formats to Triangle Splatting compatible formats.
This script can be used to validate coordinate transformations and data conversion.

Copyright (C) 2025 Visual Computing Research Center, Shenzhen University
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


def parse_vccsim_pose_file(pose_file_path: str) -> List[Dict]:
    """Parse VCCSim pose file format"""
    poses = []
    
    with open(pose_file_path, 'r') as f:
        lines = f.readlines()
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        values = line.split()
        
        if len(values) == 7:
            # Recorder format: Timestamp X Y Z Roll Pitch Yaw
            timestamp, x, y, z, roll, pitch, yaw = map(float, values)
            pose = {
                'index': line_idx,
                'timestamp': timestamp,
                'location': [x, y, z],
                'rotation': [roll, pitch, yaw],  # RPY format
                'format': 'recorder'
            }
        elif len(values) == 6:
            # Panel format: X Y Z Pitch Yaw Roll  
            x, y, z, pitch, yaw, roll = map(float, values)
            pose = {
                'index': line_idx,
                'timestamp': 0.0,
                'location': [x, y, z], 
                'rotation': [roll, pitch, yaw],  # Convert to RPY
                'format': 'panel'
            }
        else:
            print(f"Warning: Invalid pose format at line {line_idx + 1}: {line}")
            continue
        
        poses.append(pose)
    
    return poses


def convert_ue_to_triangle_splatting_coordinates(ue_location: List[float], 
                                               ue_rotation: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert UE coordinates to Triangle Splatting coordinates
    
    Based on the corrected C++ implementation that reverses DJI Terra to UE conversion
    """
    # Convert location using the same logic as C++ implementation
    ue_loc = np.array(ue_location)
    location_m = ue_loc * 0.01  # cm -> m
    
    # Step 1: Reverse the X<->Y swap (like C++ implementation)
    swapped_location = np.array([location_m[1], location_m[0], location_m[2]])  # Y, X, Z
    
    # Step 2: Reverse the 90-degree Z-axis rotation
    theta = np.radians(90.0)  # +90 degrees to reverse -90 from original
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    ts_location = np.array([
        swapped_location[0] * cos_theta - swapped_location[1] * sin_theta,
        swapped_location[0] * sin_theta + swapped_location[1] * cos_theta,
        swapped_location[2]
    ])
    
    # Convert rotation (based on DJI Terra script reversal)
    roll, pitch, yaw = ue_rotation
    
    # Reverse the angle adjustments from DJI Terra script
    adjusted_roll = roll + 180.0    # Reverse roll -= 180
    adjusted_pitch = pitch + 90.0   # Reverse pitch -= 90
    adjusted_yaw = yaw - 90.0       # Reverse yaw += 90
    
    # Build rotation matrix
    try:
        from scipy.spatial.transform import Rotation
        
        # Create rotation with adjusted angles
        adjusted_rotation_rad = np.radians([adjusted_roll, adjusted_pitch, adjusted_yaw])
        r_adjusted = Rotation.from_euler('xyz', [adjusted_rotation_rad[1], adjusted_rotation_rad[2], adjusted_rotation_rad[0]])  # pitch, yaw, roll
        
        # Apply coordinate transformation matrix (like C++ swap_XY_matrix)
        coord_transform = np.array([[0, 1, 0],   # new X = old Y
                                  [1, 0, 0],   # new Y = old X
                                  [0, 0, 1]])  # Z unchanged
        
        rotation_matrix = coord_transform @ r_adjusted.as_matrix() @ coord_transform.T
        
    except ImportError:
        print("Warning: scipy not available, using identity rotation")
        rotation_matrix = np.eye(3)
    
    return ts_location, rotation_matrix


def export_cameras_to_ply(poses: List[Dict], output_path: str, 
                         camera_params: Dict, coordinate_conversion: bool = True):
    """Export camera poses as PLY file for MeshLab verification"""
    
    ply_content = []
    
    # PLY header
    ply_content.append("ply")
    ply_content.append("format ascii 1.0")
    ply_content.append(f"element vertex {len(poses)}")
    ply_content.append("property float x")
    ply_content.append("property float y")
    ply_content.append("property float z")
    ply_content.append("property float nx")
    ply_content.append("property float ny")
    ply_content.append("property float nz")
    ply_content.append("property uchar red")
    ply_content.append("property uchar green")
    ply_content.append("property uchar blue")
    ply_content.append("end_header")
    
    # Process each pose
    for i, pose in enumerate(poses):
        if coordinate_conversion:
            # Apply coordinate transformation
            ts_location, ts_rotation = convert_ue_to_triangle_splatting_coordinates(
                pose['location'], pose['rotation']
            )
            position = ts_location
            forward = ts_rotation[:, 0]  # First column is forward direction
        else:
            # Use original UE coordinates
            position = np.array(pose['location'])
            # Simple forward direction (just X-axis)
            forward = np.array([1, 0, 0])
        
        # Color coding for easy identification
        red = (i * 137) % 256
        green = (i * 197) % 256  
        blue = (i * 73) % 256
        
        ply_content.append(f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                          f"{forward[0]:.6f} {forward[1]:.6f} {forward[2]:.6f} "
                          f"{red} {green} {blue}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(ply_content))
    
    print(f"Exported {len(poses)} cameras to: {output_path}")


def export_mesh_points_to_ply(mesh_path: str, output_path: str, 
                             coordinate_conversion: bool = True, sample_count: int = 10000):
    """Export mesh points as PLY file (placeholder - would need actual mesh loading)"""
    
    # This is a placeholder function
    # In a real implementation, you would load the mesh from the UE path
    # and sample points from it
    
    print(f"Note: Mesh export is not implemented yet")
    print(f"Expected mesh path: {mesh_path}")
    print(f"Would export to: {output_path}")
    
    # Generate random points for demonstration
    np.random.seed(42)
    points = np.random.randn(sample_count, 3) * 5.0  # Random points in 5x5x5 cube
    
    if coordinate_conversion:
        # Apply coordinate transformation to points using the corrected logic
        for i in range(len(points)):
            # Simulate UE coordinates (cm) to TS coordinates
            ue_point = points[i] * 100  # Scale up to simulate cm units
            
            # Apply the same conversion as C++ implementation
            # Step 1: Convert to meters
            location_m = ue_point * 0.01
            
            # Step 2: X<->Y swap
            swapped = np.array([location_m[1], location_m[0], location_m[2]])
            
            # Step 3: 90-degree rotation
            theta = np.radians(90.0)
            ts_point = np.array([
                swapped[0] * np.cos(theta) - swapped[1] * np.sin(theta),
                swapped[0] * np.sin(theta) + swapped[1] * np.cos(theta),
                swapped[2]
            ])
            
            points[i] = ts_point
    
    # PLY header
    ply_content = []
    ply_content.append("ply")
    ply_content.append("format ascii 1.0")
    ply_content.append(f"element vertex {len(points)}")
    ply_content.append("property float x")
    ply_content.append("property float y")
    ply_content.append("property float z")
    ply_content.append("property uchar red")
    ply_content.append("property uchar green")
    ply_content.append("property uchar blue")
    ply_content.append("end_header")
    
    # Add points
    for point in points:
        # Simple color based on position
        red = int(abs(point[0]) * 255) % 256
        green = int(abs(point[1]) * 255) % 256
        blue = int(abs(point[2]) * 255) % 256
        
        ply_content.append(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {red} {green} {blue}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(ply_content))
    
    print(f"Exported {len(points)} mesh points to: {output_path}")


def convert_vccsim_to_triangle_splatting_format(config_path: str, output_dir: str):
    """Convert VCCSim data to Triangle Splatting format"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=== VCCSim to Triangle Splatting Converter ===")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse pose file
    pose_file = config['pose_file']
    if not os.path.exists(pose_file):
        print(f"Error: Pose file not found: {pose_file}")
        return False
    
    poses = parse_vccsim_pose_file(pose_file)
    print(f"Loaded {len(poses)} poses from {pose_file}")
    
    if len(poses) == 0:
        print("Error: No valid poses found")
        return False
    
    # Camera parameters
    camera_params = config.get('camera', {})
    
    # Export cameras (both original and transformed)
    camera_ue_path = os.path.join(output_dir, "cameras_ue_coordinates.ply")
    camera_ts_path = os.path.join(output_dir, "cameras_ts_coordinates.ply")
    
    export_cameras_to_ply(poses, camera_ue_path, camera_params, coordinate_conversion=False)
    export_cameras_to_ply(poses, camera_ts_path, camera_params, coordinate_conversion=True)
    
    # Export mesh if specified
    mesh_config = config.get('mesh', {})
    if mesh_config.get('use_mesh_initialization', False) and 'mesh_path' in mesh_config:
        mesh_ue_path = os.path.join(output_dir, "mesh_ue_coordinates.ply")
        mesh_ts_path = os.path.join(output_dir, "mesh_ts_coordinates.ply")
        
        export_mesh_points_to_ply(mesh_config['mesh_path'], mesh_ue_path, 
                                 coordinate_conversion=False)
        export_mesh_points_to_ply(mesh_config['mesh_path'], mesh_ts_path, 
                                 coordinate_conversion=True)
    
    # Create summary report
    summary_path = os.path.join(output_dir, "conversion_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("VCCSim to Triangle Splatting Conversion Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration file: {config_path}\n")
        f.write(f"Pose file: {pose_file}\n")
        f.write(f"Number of poses: {len(poses)}\n")
        f.write(f"Image directory: {config.get('image_directory', 'N/A')}\n")
        f.write(f"Camera parameters: {camera_params}\n\n")
        f.write("Generated files:\n")
        f.write(f"- {camera_ue_path} (UE coordinates)\n")
        f.write(f"- {camera_ts_path} (Triangle Splatting coordinates)\n")
        if mesh_config.get('use_mesh_initialization', False):
            f.write(f"- {mesh_ue_path} (UE mesh coordinates)\n")
            f.write(f"- {mesh_ts_path} (Triangle Splatting mesh coordinates)\n")
        f.write("\nValidation:\n")
        f.write("1. Open the PLY files in MeshLab\n")
        f.write("2. Compare UE vs TS coordinate versions\n")
        f.write("3. Verify camera orientations (normals) point correctly\n")
        f.write("4. Check that scene appears correctly oriented in right-handed system\n")
    
    print(f"Conversion completed. Summary saved to: {summary_path}")
    print("\nTo validate coordinate transformation:")
    print("1. Open the generated PLY files in MeshLab")
    print("2. Compare UE coordinates vs Triangle Splatting coordinates")
    print("3. Verify the scene orientation looks correct in the right-handed system")
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VCCSim Data Converter")
    parser.add_argument('--config', required=True, help='Path to VCCSim config JSON file')
    parser.add_argument('--output', required=True, help='Output directory for PLY files')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    # Run conversion
    success = convert_vccsim_to_triangle_splatting_format(args.config, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())