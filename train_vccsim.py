#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved VCCSim Triangle Splatting Training Script

This script integrates Triangle Splatting training with VCCSim data format.
It uses the actual Triangle Splatting training infrastructure while supporting VCCSim data.

Copyright (C) 2025 Visual Computing Research Center, Shenzhen University
"""

import os
import sys
import json

# Set console encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add triangle splatting modules to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

try:
    # Import triangle splatting modules
    from scene import Scene, TriangleModel
    from scene.triangle_model import TriangleModel
    from scene.cameras import Camera
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from utils.general_utils import safe_state
    from triangle_renderer import render
    
    # Try to import network_gui (optional, used for GUI features)
    try:
        from triangle_renderer import network_gui
    except ImportError:
        network_gui = None
        print("Note: network_gui not available, continuing without GUI support")
    from utils.loss_utils import l1_loss, ssim, l2_loss
    from utils.image_utils import psnr
    import torch
    from tqdm import tqdm
    from random import randint
    from utils.loss_utils import l1_loss, ssim
    from utils.general_utils import PILtoTorch
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, OptimizationParams
    import uuid
except ImportError as e:
    print(f"Error importing triangle splatting modules: {e}")
    print("Please ensure triangle splatting environment is properly set up.")
    print("Required dependencies: torch, torchvision, PIL, tqdm, etc.")
    sys.exit(1)



def load_vccsim_cameras(config: Dict, device: torch.device) -> List[Camera]:
    """Load VCCSim camera information from C++ generated camera_info.json"""
    
    # Look for camera_info.json in the training output config directory
    output_dir = Path(config['output_directory'])
    camera_info_path = output_dir / 'config' / 'camera_info.json'
    
    if not camera_info_path.exists():
        raise FileNotFoundError(f"camera_info.json not found at {camera_info_path}. "
                               f"Make sure to run VCCSimDataConverter first to generate camera data.")
    
    print(f"Loading camera info from: {camera_info_path}")
    
    # Load camera information from JSON
    import json
    with open(camera_info_path, 'r') as f:
        camera_data = json.load(f)
    
    print(f"Loaded {len(camera_data)} cameras from C++ generated data")
    
    cameras = []
    
    for camera_info in camera_data:
        try:
            # Show progress every 50 images
            idx = camera_info['uid']
            if idx % 50 == 0 or idx == len(camera_data) - 1:
                print(f"Loading camera {idx+1}/{len(camera_data)}: {camera_info['image_name']}")
            
            # Load and process image
            from PIL import Image
            image_path = camera_info['image_path']
            
            if not Path(image_path).exists():
                print(f"Warning: Image not found: {image_path}")
                continue
                
            image_pil = Image.open(image_path).convert('RGB')
            
            # Get image dimensions from camera info (already correctly sized by C++)
            width = camera_info['width']
            height = camera_info['height']
            
            # Resize if necessary
            if image_pil.size != (width, height):
                image_pil = image_pil.resize((width, height), Image.LANCZOS)
            
            # Convert to tensor
            image_tensor = PILtoTorch(image_pil, (height, width)).to(device)
            
            # Extract camera parameters (already converted by C++)
            focal_x = camera_info['focal_x']
            focal_y = camera_info['focal_y']
            
            # Calculate FOV in radians
            fov_x_rad = 2 * np.arctan(width / (2 * focal_x))
            fov_y_rad = 2 * np.arctan(height / (2 * focal_y))
            
            # Extract rotation matrix and translation (already in world-to-camera format from C++)
            rotation_flat = camera_info['rotation']  # 9-element array
            R_w2c = np.array(rotation_flat).reshape(3, 3).astype(np.float32)
            T_w2c = np.array(camera_info['translation']).astype(np.float32)
            
            camera = Camera(
                colmap_id=idx,
                R=R_w2c,
                T=T_w2c,
                FoVx=fov_x_rad,
                FoVy=fov_y_rad,
                image=image_tensor,
                gt_alpha_mask=None,
                image_name=camera_info['image_name'],
                uid=idx,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=1.0,
                data_device=device
            )
            
            cameras.append(camera)
            
        except Exception as e:
            print(f"Error loading camera {camera_info.get('uid', 'unknown')}: {e}")
            continue
    
    print(f"Successfully loaded {len(cameras)} cameras from C++ converted data")
    return cameras


def create_vccsim_scene(config: Dict, device: torch.device) -> Scene:
    """Create Triangle Splatting scene from VCCSim configuration - COMPLETELY BYPASSES COLMAP"""
    
    # Load cameras directly from VCCSim data
    cameras = load_vccsim_cameras(config, device)
    
    if len(cameras) == 0:
        raise ValueError("No valid cameras loaded from VCCSim data")
    
    print(f"[OK] Loaded {len(cameras)} cameras directly from VCCSim (bypassing COLMAP)")
    
    # Calculate scene bounds from camera positions  
    camera_positions = np.array([cam.T for cam in cameras])
    scene_center = np.mean(camera_positions, axis=0)
    scene_radius = np.max(np.linalg.norm(camera_positions - scene_center, axis=1)) * 0.5
    
    print(f"Scene bounds: center={scene_center}, radius={scene_radius}")
    
    # Initialize point cloud
    mesh_config = config.get('mesh', {})
    use_mesh_init = mesh_config.get('use_mesh_initialization', False)
    
    if use_mesh_init and 'mesh_path' in mesh_config:
        print(f"[MESH] Using mesh initialization from: {mesh_config['mesh_path']}")
        # Load mesh points from UE-exported PLY file (created by VCCSimDataConverter)
        mesh_ply_path = os.path.join(config['output_directory'], 'config', 'init_points.ply')
        if os.path.exists(mesh_ply_path):
            # Load PLY point cloud
            point_cloud = load_ply_point_cloud(mesh_ply_path)
            if point_cloud is not None:
                print(f"[OK] Loaded {len(point_cloud.points)} points from UE mesh")
            else:
                print("[WARN] PLY loading failed, falling back to random points")
                point_cloud = generate_scene_point_cloud(camera_positions, scene_center, scene_radius)
        else:
            print("[WARN] Mesh PLY not found, falling back to scene-bound random points")
            point_cloud = generate_scene_point_cloud(camera_positions, scene_center, scene_radius)
    else:
        print("[RANDOM] Using random point cloud initialization within scene bounds")
        point_cloud = generate_scene_point_cloud(camera_positions, scene_center, scene_radius)
    
    # Create Triangle Model for initialization (sh_degree = 3 is common for Triangle Splatting)
    triangles = TriangleModel(sh_degree=3)
    
    # Training parameters from config
    training_config = config.get('training', {})
    
    # Initialize triangle model from point cloud directly
    # nb_points should be points per triangle (3 for triangles)
    print(f"[DEBUG] Before create_from_pcd: point_cloud has {len(point_cloud.points)} points")
    print(f"[DEBUG] Triangle model triangles_points shape before: {triangles._triangles_points.shape}")
    
    try:
        triangles.create_from_pcd(
            pcd=point_cloud,
            spatial_lr_scale=scene_radius,  # Use scene radius for spatial learning rate
            init_opacity=0.1,
            init_size=0.01, 
            nb_points=3,  # Points per triangle (should always be 3 for triangles)
            set_sigma=0.1,  # Set sigma value instead of boolean
            no_dome=False
        )
        print(f"[DEBUG] create_from_pcd completed successfully")
    except Exception as e:
        print(f"[ERROR] create_from_pcd failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[DEBUG] After create_from_pcd: triangle model triangles_points shape: {triangles._triangles_points.shape}")
    print(f"[DEBUG] Triangle model get_triangles_points shape: {triangles.get_triangles_points.shape}")
    print(f"[DEBUG] Triangle model has {triangles._triangles_points.size(0)} triangles")
    
    # Create custom scene object without using the Scene class auto-detection
    # This bypasses the "Could not recognize scene type!" error
    train_cameras_dict = {1.0: cameras[:-len(cameras)//10] if len(cameras) > 10 else cameras}
    test_cameras_dict = {1.0: cameras[-len(cameras)//10:] if len(cameras) > 10 else cameras[:1]}
    
    class CustomScene:
        def __init__(self):
            self.triangles = triangles
            self.train_cameras = train_cameras_dict
            self.test_cameras = test_cameras_dict
            self.cameras_extent = scene_radius
            self.model_path = config['output_directory']
            self.white_background = False  # Set background to black by default
            
        def getTrainCameras(self, resolution_scale=1.0):
            return self.train_cameras[resolution_scale]
            
        def getTestCameras(self, resolution_scale=1.0):
            return self.test_cameras[resolution_scale]
            
        def save(self, iteration):
            pass  # Placeholder save method
    
    scene = CustomScene()
    
    print(f"[OK] Scene created successfully - {len(scene.train_cameras[1.0])} training cameras")
    print(f"[INFO] Point cloud initialization: {len(point_cloud.points)} points")
    
    return scene


def load_ply_point_cloud(ply_path: str):
    """Load point cloud from PLY file exported by VCCSimDataConverter"""
    try:
        import plyfile
        
        # Read PLY file
        print(f"[DEBUG] Reading PLY file: {ply_path}")
        plydata = plyfile.PlyData.read(ply_path)
        
        # Handle different PLY data structures
        if hasattr(plydata, 'elements'):
            if hasattr(plydata.elements, 'keys'):
                print(f"[DEBUG] PLY data loaded, elements: {list(plydata.elements.keys())}")
            else:
                element_names = [element.name for element in plydata.elements]
                print(f"[DEBUG] PLY data loaded, elements: {element_names}")
        
        vertices = plydata['vertex']
        print(f"[DEBUG] Vertex data type: {type(vertices)}")
        print(f"[DEBUG] Vertex dtype: {vertices.dtype}")
        print(f"[DEBUG] Vertex dtype names: {getattr(vertices.dtype, 'names', 'No names attribute')}")
        
        # Extract coordinates
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        print(f"[DEBUG] Extracted {len(points)} points")
        
        # Extract colors (if available) - check directly in vertex data
        try:
            # Try to access RGB color data directly
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
            print("[DEBUG] Extracted RGB colors from PLY")
        except (ValueError, KeyError) as e:
            colors = np.random.rand(len(points), 3)  # Random colors
            print(f"[DEBUG] Using random colors (no RGB data in PLY): {e}")
        
        # Extract normals (if available)
        try:
            # Try to access normal data directly
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            print("[DEBUG] Extracted normals from PLY")
        except (ValueError, KeyError) as e:
            normals = np.random.randn(len(points), 3)  # Random normals
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            print(f"[DEBUG] Generated random normals: {e}")
        
        # Create point cloud object (Triangle Splatting format)
        point_cloud = type('BasicPointCloud', (), {
            'points': points,
            'colors': colors, 
            'normals': normals
        })()
        
        print(f"[OK] PLY loaded successfully: {len(points)} points")
        return point_cloud
        
    except ImportError:
        print("[WARN] plyfile not available, falling back to manual PLY parsing")
        return parse_ply_manually(ply_path)
    except Exception as e:
        import traceback
        print(f"[WARN] Error loading PLY: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        print("[WARN] Falling back to random points")
        return None


def parse_ply_manually(ply_path: str):
    """Manual PLY parsing as fallback"""
    try:
        points = []
        colors = []
        normals = []
        
        with open(ply_path, 'r') as f:
            lines = f.readlines()
            
        in_data = False
        for line in lines:
            if line.startswith('end_header'):
                in_data = True
                continue
            if in_data and line.strip():
                values = line.strip().split()
                if len(values) >= 6:
                    # x, y, z, nx, ny, nz, r, g, b
                    points.append([float(values[0]), float(values[1]), float(values[2])])
                    normals.append([float(values[3]), float(values[4]), float(values[5])])
                    if len(values) >= 9:
                        colors.append([float(values[6])/255.0, float(values[7])/255.0, float(values[8])/255.0])
                    else:
                        colors.append([0.5, 0.5, 0.5])  # Default gray
        
        point_cloud = type('BasicPointCloud', (), {
            'points': np.array(points),
            'colors': np.array(colors),
            'normals': np.array(normals)
        })()
        
        return point_cloud
    except Exception as e:
        print(f"[WARN] Manual PLY parsing failed: {e}")
        return None


def generate_scene_point_cloud(camera_positions: np.ndarray, scene_center: np.ndarray, scene_radius: float):
    """Generate random point cloud within scene bounds"""
    num_points = 100000  # Standard Triangle Splatting initialization
    
    # Generate random points within scene bounds
    np.random.seed(42)  # Reproducible initialization
    
    points = []
    for _ in range(num_points):
        # Random point within scene sphere
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        distance = np.random.uniform(0, scene_radius * 0.8)  # Stay within 80% of bounds
        point = scene_center + direction * distance
        points.append(point)
    
    points = np.array(points)
    colors = np.random.rand(num_points, 3)  # Random colors
    normals = np.random.randn(num_points, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    point_cloud = type('BasicPointCloud', (), {
        'points': points,
        'colors': colors,
        'normals': normals
    })()
    
    print(f"Generated {num_points} random points for initialization")
    
    return point_cloud


def training_step(scene: Scene, triangle_model: TriangleModel, cameras: List[Camera], 
                 iteration: int, opt_params: OptimizationParams, pipe_params: PipelineParams,
                 background: torch.Tensor, device: torch.device) -> Dict[str, float]:
    """Perform one training step"""
    
    # Select random camera
    camera_idx = randint(0, len(cameras) - 1)
    viewpoint_camera = cameras[camera_idx]
    
    # Render
    render_pkg = render(viewpoint_camera, triangle_model, pipe_params, background)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
    # Calculate loss
    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
    
    loss.backward()
    
    # Calculate metrics
    with torch.no_grad():
        psnr_value = psnr(image, gt_image).mean().double()
    
    return {
        'loss': loss.item(),
        'l1_loss': Ll1.item(),
        'psnr': psnr_value.item()
    }


def train_vccsim_triangle_splatting(config_path: str, output_dir: str, log_file: str):
    """Main training function for VCCSim Triangle Splatting"""
    
    print("=== VCCSim Triangle Splatting Training ===")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Log: {log_file}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Configuration loaded successfully")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"VCCSim Triangle Splatting Training Started\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Device: {device}\n\n")
    
    try:
        # Create scene
        print("Creating Triangle Splatting scene...")
        scene = create_vccsim_scene(config, device)
        print("Scene created successfully")
        
        # Use the triangle model from the scene (already initialized with point cloud)
        print("Using Triangle Splatting model from scene...")
        triangle_model = scene.triangles
        print(f"[DEBUG] Triangle model from scene has shape: {triangle_model.get_triangles_points.shape}")
        
        # Use original Triangle Splatting optimization parameters
        opt_params = OptimizationParams(ArgumentParser())
        triangle_model.training_setup(
            opt_params,
            opt_params.lr_mask,
            opt_params.feature_lr,
            opt_params.opacity_lr,
            opt_params.lr_sigma,
            opt_params.lr_triangles_points_init
        )
        
        print(f"  Triangle Splatting learning rates:")
        print(f"    Feature LR: {opt_params.feature_lr}")
        print(f"    Opacity LR: {opt_params.opacity_lr}")
        print(f"    Sigma LR: {opt_params.lr_sigma}")
        print(f"    Triangle Points LR: {opt_params.lr_triangles_points_init}")
        print(f"    Mask LR: {opt_params.lr_mask}")
        
        # Set up background
        bg_color = [1, 1, 1] if scene.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        # Training parameters
        training_config = config.get('training', {})
        max_iterations = training_config.get('max_iterations', 5000)
        
        print(f"Training parameters:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Using Triangle Splatting original learning rate schedule")
        print(f"  Cameras: {len(scene.getTrainCameras())}")
        
        # Training loop
        print("Starting training...")
        progress_bar = tqdm(range(1, max_iterations + 1), desc="Training")
        
        for iteration in progress_bar:
            triangle_model.update_learning_rate(iteration)
            
            # Training step
            triangle_model.optimizer.zero_grad(set_to_none=True)
            
            metrics = training_step(
                scene, triangle_model, scene.getTrainCameras(),
                iteration, opt_params, 
                PipelineParams(ArgumentParser()), background, device
            )
            
            triangle_model.optimizer.step()
            
            # Update progress with triangle count
            triangle_count = triangle_model.get_triangles_points.shape[0]
            progress_bar.set_postfix({
                'Loss': f"{metrics['loss']:.6f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'Triangles': f"{triangle_count}"
            })
            
            # Log progress periodically
            if iteration % 100 == 0 or iteration == max_iterations:
                log_message = f"Iteration {iteration}/{max_iterations} - Loss: {metrics['loss']:.6f}, PSNR: {metrics['psnr']:.2f}\n"
                
                # Safe file writing with error handling
                try:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(log_message)
                        f.write(f"Progress: {iteration / max_iterations * 100:.1f}%\n")
                        f.flush()  # Ensure immediate write
                except PermissionError as e:
                    print(f"Warning: Could not write to log file: {e}")
                    # Continue training without logging to file
                except Exception as e:
                    print(f"Warning: Log file error: {e}")
            
            # Periodic operations (from original Triangle Splatting training)
            if iteration >= opt_params.densify_from_iter and iteration < opt_params.densify_until_iter:
                if iteration % opt_params.densification_interval == 0:
                    # Basic densification - add new triangles based on importance
                    current_count = triangle_model.get_triangles_points.shape[0]
                    max_triangles = int(current_count * 1.2)  # Allow 20% growth
                    triangle_model.add_new_gs(max_triangles)
                    
                    # Log densification
                    new_count = triangle_model.get_triangles_points.shape[0]
                    if new_count != current_count:
                        print(f"[DENSIFY] Iteration {iteration}: {current_count} -> {new_count} triangles")
            
            # Opacity reset logic - typically done every 3000 iterations in Triangle Splatting
            if iteration % 3000 == 0:
                triangle_model.reset_opacity(0.01)
                
            # Learning rate scheduling (exponential decay)
            if iteration % 1000 == 0 and iteration > 0:
                # Reduce learning rates by 10% every 1000 iterations for better convergence
                for param_group in triangle_model.optimizer.param_groups:
                    param_group['lr'] *= 0.95
                    
                # Log LR update every 5000 iterations  
                if iteration % 5000 == 0:
                    current_lrs = [group['lr'] for group in triangle_model.optimizer.param_groups]
                    print(f"[LR UPDATE] Iteration {iteration}: Learning rates: {[f'{lr:.6f}' for lr in current_lrs]}")
        
        # Save final model
        print("Saving final model...")
        model_path = os.path.join(output_dir, "final_model.ply")
        triangle_model.save_ply(model_path)
        
        # Training completed
        print("Training completed successfully!")
        
        with open(log_file, 'a') as f:
            f.write("Training completed successfully!\n")
            f.write(f"Final iteration: {max_iterations}\n")
            f.write(f"Model saved to: {model_path}\n")
        
        return True
        
    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}"
        print(error_msg)
        print("Full traceback:")
        traceback.print_exc()
        
        with open(log_file, 'a') as f:
            f.write(f"ERROR: {error_msg}\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VCCSim Triangle Splatting Training")
    parser.add_argument('--config', required=True, help='Path to VCCSim config JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--log', required=True, help='Log file path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    # Run training
    success = train_vccsim_triangle_splatting(args.config, args.output, args.log)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())