#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCCSim Triangle Splatting Render Test Script with Training

Test rendering with VCCSim data using Triangle Splatting after performing training iterations.
This script performs a few training iterations to optimize the triangles before rendering.
"""

import os
import sys
import json
import torch
import torchvision
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from random import randint

# Add triangle splatting modules to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

try:
    from scene import Scene, TriangleModel
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from triangle_renderer import render
    from utils.general_utils import safe_state
    from utils.loss_utils import l1_loss, ssim
    from argparse import ArgumentParser
    from tqdm import tqdm
    
except ImportError as e:
    print(f"Error importing triangle splatting modules: {e}")
    sys.exit(1)


def prepare_output_and_logger(args):
    """Setup output directories"""
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    
    return None  # We don't use tensorboard for this simple test


def training_step(triangles, viewpoint_cam, pipe, background, optimizer, iteration):
    """Perform one training step"""
    
    # Clear gradients
    triangles.optimizer.zero_grad()
    
    # Render
    render_pkg = render(viewpoint_cam, triangles, pipe, background)
    image = render_pkg["render"]
    
    # Loss
    gt_image = viewpoint_cam.original_image.cuda()
    pixel_loss = l1_loss(image, gt_image)
    loss_image = (1.0 - 0.2) * pixel_loss + 0.2 * (1.0 - ssim(image, gt_image))
    
    # Simple opacity regularization
    loss_opacity = torch.abs(triangles.get_opacity).mean() * 0.01
    
    # Total loss
    loss = loss_image + loss_opacity
    
    # Backward pass
    loss.backward()
    
    with torch.no_grad():
        # Update learning rate
        triangles.update_learning_rate(iteration)
        
        # Optimizer step
        triangles.optimizer.step()
        
        # Update SH degree every 1000 iterations
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()
    
    return loss.item(), pixel_loss.item()


def main():
    parser = argparse.ArgumentParser(description="VCCSim Triangle Splatting Render Test with Training")
    parser.add_argument("--config_dir", 
                       default="C:\\UEProjects\\VCCSimDev\\Saved\\TriangleSplatting\\TriangleSplatting\\training_sessions\\session_20250905_013448\\config",
                       help="VCCSim config directory containing camera_info.json and init_points.ply")
    parser.add_argument("--output", 
                       default="C:\\UEProjects\\VCCSimDev\\Saved\\TriangleSplatting\\TriangleSplatting\\training_sessions\\session_20250905_013448\\test_with_training",
                       help="Output directory for test renders")
    parser.add_argument("--images", 
                       default=None,
                       help="Images directory (optional, will use image_path from camera_info.json)")
    parser.add_argument("--training_iterations", type=int, default=10000,
                       help="Number of training iterations to perform before rendering (default: 10)")
    
    args = parser.parse_args()
    
    print("=== VCCSim Triangle Splatting Render Test with Training ===")
    print(f"Config directory: {args.config_dir}")
    print(f"Output: {args.output}")
    print(f"Training iterations: {args.training_iterations}")
    
    # Validate VCCSim config directory
    required_files = [
        os.path.join(args.config_dir, 'camera_info.json'),
        os.path.join(args.config_dir, 'init_points.ply')
    ]
    
    for required_file in required_files:
        if not os.path.exists(required_file):
            print(f"[ERROR] Required file not found: {required_file}")
            return 1
    
    # Initialize system state
    safe_state(False)
    
    # Initialize CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create model parameters for VCCSim Scene
        model_parser = ArgumentParser()
        model = ModelParams(model_parser, sentinel=True)
        opt_parser = ArgumentParser()
        opt = OptimizationParams(opt_parser)
        pipe_parser = ArgumentParser()
        pipe = PipelineParams(pipe_parser)
        
        # Create args for VCCSim Scene
        class VCCSimArgs:
            def __init__(self):
                self.source_path = args.config_dir
                self.images = args.images
                self.model_path = args.output
                self.sh_degree = 3
                self.white_background = False
                self.resolution = -1
                self.data_device = "cuda"
                self.eval = False
        
        scene_args = VCCSimArgs()
        
        print(f"[INFO] Using VCCSim data pipeline")
        print(f"[INFO] Config directory: {scene_args.source_path}")
        print(f"[INFO] Images directory: {scene_args.images or 'from camera_info.json'}")
        
        # Create output directory early (Scene needs it)
        os.makedirs(args.output, exist_ok=True)
        print(f"[INFO] Created output directory: {args.output}")
        
        # Extract model parameters
        model_params = model.extract(scene_args)
        
        # Create triangle model
        triangles = TriangleModel(model_params.sh_degree)
        
        print("[INFO] Loading VCCSim scene...")
        
        # Load scene using VCCSim Scene class
        scene = Scene(
            args=model_params,
            triangles=triangles,
            init_opacity=opt.set_opacity,      # 0.28
            init_size=opt.triangle_size,       # 2.23
            nb_points=opt.nb_points,           # 3
            set_sigma=True,                    # Enable sigma
            no_dome=False,
            load_iteration=None,
            shuffle=False
        )
        
        # Setup training (use original opt object, not extracted)
        triangles.training_setup(
            opt,               # Use original OptimizationParams object
            opt.lr_mask,       # 0.01
            opt.feature_lr,    # 0.0025
            opt.opacity_lr,    # 0.014
            opt.lr_sigma,      # 0.0008
            opt.lr_triangles_points_init  # 0.0018
        )
        
        print("[OK] VCCSim scene loaded successfully")
        
        # Get camera data from scene
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        
        print(f"[INFO] Scene loaded:")
        print(f"[INFO]   Training cameras: {len(train_cameras)}")
        print(f"[INFO]   Test cameras: {len(test_cameras)}")
        print(f"[INFO]   Total triangles: {triangles.get_triangles_points.shape[0]}")
        
        # Detailed coordinate analysis and logging
        if len(train_cameras) > 0:
            print("\n" + "="*80)
            print("[COORDINATE ANALYSIS]")
            print("="*80)
            
            # Camera coordinate analysis
            all_cameras = train_cameras + test_cameras
            camera_positions = torch.stack([cam.camera_center for cam in all_cameras])
            
            print(f"[CAMERAS] Total cameras analyzed: {len(all_cameras)}")
            print(f"[CAMERAS] Position ranges:")
            print(f"  X: [{camera_positions[:, 0].min():.4f}, {camera_positions[:, 0].max():.4f}] (range: {camera_positions[:, 0].max() - camera_positions[:, 0].min():.4f})")
            print(f"  Y: [{camera_positions[:, 1].min():.4f}, {camera_positions[:, 1].max():.4f}] (range: {camera_positions[:, 1].max() - camera_positions[:, 1].min():.4f})")
            print(f"  Z: [{camera_positions[:, 2].min():.4f}, {camera_positions[:, 2].max():.4f}] (range: {camera_positions[:, 2].max() - camera_positions[:, 2].min():.4f})")
            
            # Camera center statistics
            cam_center = camera_positions.mean(dim=0)
            cam_std = camera_positions.std(dim=0)
            print(f"[CAMERAS] Center: [{cam_center[0]:.4f}, {cam_center[1]:.4f}, {cam_center[2]:.4f}]")
            print(f"[CAMERAS] Std Dev: [{cam_std[0]:.4f}, {cam_std[1]:.4f}, {cam_std[2]:.4f}]")
            
            # Triangle coordinate analysis
            triangles_points = triangles.get_triangles_points  # Shape: [N_triangles, 3, 3]
            triangles_flat = triangles_points.view(-1, 3)  # Flatten to [N_triangles*3, 3]
            
            print(f"\n[TRIANGLES] Total triangles: {triangles_points.shape[0]}")
            print(f"[TRIANGLES] Total triangle vertices: {triangles_flat.shape[0]}")
            print(f"[TRIANGLES] Vertex coordinate ranges:")
            print(f"  X: [{triangles_flat[:, 0].min():.4f}, {triangles_flat[:, 0].max():.4f}] (range: {triangles_flat[:, 0].max() - triangles_flat[:, 0].min():.4f})")
            print(f"  Y: [{triangles_flat[:, 1].min():.4f}, {triangles_flat[:, 1].max():.4f}] (range: {triangles_flat[:, 1].max() - triangles_flat[:, 1].min():.4f})")
            print(f"  Z: [{triangles_flat[:, 2].min():.4f}, {triangles_flat[:, 2].max():.4f}] (range: {triangles_flat[:, 2].max() - triangles_flat[:, 2].min():.4f})")
            
            # Triangle statistics
            tri_center = triangles_flat.mean(dim=0)
            tri_std = triangles_flat.std(dim=0)
            print(f"[TRIANGLES] Center: [{tri_center[0]:.4f}, {tri_center[1]:.4f}, {tri_center[2]:.4f}]")
            print(f"[TRIANGLES] Std Dev: [{tri_std[0]:.4f}, {tri_std[1]:.4f}, {tri_std[2]:.4f}]")
            
            # Bounding box analysis
            print(f"\n[BBOX ANALYSIS]")
            cam_bbox_min = camera_positions.min(dim=0)[0]
            cam_bbox_max = camera_positions.max(dim=0)[0]
            tri_bbox_min = triangles_flat.min(dim=0)[0]
            tri_bbox_max = triangles_flat.max(dim=0)[0]
            
            print(f"Camera BBox: Min[{cam_bbox_min[0]:.4f}, {cam_bbox_min[1]:.4f}, {cam_bbox_min[2]:.4f}] Max[{cam_bbox_max[0]:.4f}, {cam_bbox_max[1]:.4f}, {cam_bbox_max[2]:.4f}]")
            print(f"Triangle BBox: Min[{tri_bbox_min[0]:.4f}, {tri_bbox_min[1]:.4f}, {tri_bbox_min[2]:.4f}] Max[{tri_bbox_max[0]:.4f}, {tri_bbox_max[1]:.4f}, {tri_bbox_max[2]:.4f}]")
            
            # Scale difference analysis
            cam_scale = (cam_bbox_max - cam_bbox_min).norm()
            tri_scale = (tri_bbox_max - tri_bbox_min).norm()
            print(f"Camera scene scale: {cam_scale:.4f}")
            print(f"Triangle scene scale: {tri_scale:.4f}")
            print(f"Scale ratio (tri/cam): {tri_scale/cam_scale:.4f}")
            
            # Opacity and other triangle properties
            if hasattr(triangles, '_opacity'):
                opacities = triangles.get_opacity
                print(f"\n[TRIANGLE PROPERTIES]")
                print(f"Opacity: [{opacities.min():.4f}, {opacities.max():.4f}] (mean: {opacities.mean():.4f}, std: {opacities.std():.4f})")
            
            if hasattr(triangles, '_sigma'):
                sigmas = triangles.get_sigma
                print(f"Sigma: [{sigmas.min():.4f}, {sigmas.max():.4f}] (mean: {sigmas.mean():.4f}, std: {sigmas.std():.4f})")
                
            print("="*80 + "\n")
        
        if len(train_cameras) == 0:
            print("[ERROR] No training cameras found!")
            return 1
        
        # Background color (black)
        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Perform training iterations
        print(f"[INFO] Starting {args.training_iterations} training iterations...")
        
        # Create training output directories for intermediate renders
        training_render_dir = os.path.join(args.output, "training_renders")
        os.makedirs(training_render_dir, exist_ok=True)
        
        viewpoint_stack = train_cameras.copy()
        progress_bar = tqdm(range(1, args.training_iterations + 1), desc="Training progress")
        
        # Select a fixed test camera for consistent intermediate rendering
        test_viewpoint = train_cameras[0] if len(train_cameras) > 0 else None
        
        for iteration in progress_bar:
            # Select random viewpoint
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Training step
            total_loss, pixel_loss = training_step(
                triangles, viewpoint_cam, pipe, background, None, iteration
            )
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss:.4f}',
                'L1': f'{pixel_loss:.4f}'
            })
            
            # Output intermediate render every 2000 iterations
            if iteration % 2000 == 0 or iteration == 1:
                print(f"\n[INFO] Saving intermediate render at iteration {iteration}")
                
                with torch.no_grad():
                    # Render the fixed test viewpoint
                    render_result = render(test_viewpoint, triangles, pipe, background)
                    rendering = render_result["render"]
                    
                    # Ground truth for comparison
                    gt = test_viewpoint.original_image[0:3, :, :]
                    
                    # Save intermediate render and ground truth
                    iter_render_path = os.path.join(training_render_dir, f"iter_{iteration:06d}_render.png")
                    iter_gt_path = os.path.join(training_render_dir, f"iter_{iteration:06d}_gt.png")
                    
                    torchvision.utils.save_image(rendering, iter_render_path)
                    torchvision.utils.save_image(gt, iter_gt_path)
                    
                    print(f"[OK] Saved intermediate renders:")
                    print(f"  Render: {iter_render_path}")
                    print(f"  GT: {iter_gt_path}")
                    
                    # Log current triangle properties
                    if hasattr(triangles, '_opacity'):
                        current_opacity = triangles.get_opacity
                        print(f"[ITER {iteration}] Opacity range: [{current_opacity.min():.4f}, {current_opacity.max():.4f}] (mean: {current_opacity.mean():.4f})")
                    
                    if hasattr(triangles, '_sigma'):
                        current_sigma = triangles.get_sigma
                        print(f"[ITER {iteration}] Sigma range: [{current_sigma.min():.4f}, {current_sigma.max():.4f}] (mean: {current_sigma.mean():.4f})")
        
        print("[OK] Training iterations completed")
        
        # Create output directories
        render_dir = os.path.join(args.output, "renders")
        gt_dir = os.path.join(args.output, "gt")
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        
        # Create simple test cameras
        from scene.cameras import Camera
        
        additional_test_cameras = []
        if len(train_cameras) > 0:
            # Camera parameters from first training camera
            first_cam = train_cameras[0]
            width = int(first_cam.image_width)
            height = int(first_cam.image_height)
            FoVx = first_cam.FoVx
            FoVy = first_cam.FoVy
            
            # Create a simple test camera at scene center
            center_pos = torch.tensor([0.0, 0.0, 1.0], device="cuda")
            R = torch.eye(3, device="cuda", dtype=torch.float32)
            
            test_camera = Camera(
                colmap_id=9999,
                R=R.cpu().numpy(),
                T=center_pos.cpu().numpy(), 
                FoVx=FoVx,
                FoVy=FoVy,
                image=torch.zeros((3, height, width)),
                gt_alpha_mask=None,
                image_name="center_view",
                uid=9999,
                data_device="cuda"
            )
            additional_test_cameras.append(test_camera)
        
        print(f"[INFO] Starting rendering process...")
        print(f"[INFO] Using {len(train_cameras)} training cameras + {len(additional_test_cameras)} test cameras")
        
        with torch.no_grad():
            # Render training cameras
            num_to_render = min(5, len(train_cameras))
            for idx, camera in enumerate(train_cameras[:num_to_render]):
                print(f"[INFO] Rendering training camera {idx+1}/{num_to_render}")
                
                # Render using Triangle Splatting
                render_result = render(camera, triangles, pipe, background)
                rendering = render_result["render"]
                
                # Ground truth
                gt = camera.original_image[0:3, :, :]
                
                # Save rendered image and ground truth
                render_path = os.path.join(render_dir, f"train_render_{idx:02d}.png")
                gt_path = os.path.join(gt_dir, f"train_gt_{idx:02d}.png")
                
                torchvision.utils.save_image(rendering, render_path)
                torchvision.utils.save_image(gt, gt_path)
                
                print(f"[OK] Saved: {render_path}")
            
            # Render additional test cameras
            for idx, camera in enumerate(additional_test_cameras):
                print(f"[INFO] Rendering test camera: {camera.image_name}")
                
                # Render using Triangle Splatting
                render_result = render(camera, triangles, pipe, background)
                rendering = render_result["render"]
                
                # Save rendered image
                render_path = os.path.join(render_dir, f"test_{camera.image_name}.png")
                torchvision.utils.save_image(rendering, render_path)
                
                print(f"[OK] Saved: {render_path}")
        
        print(f"[OK] Test rendering with training complete. Results saved to: {args.output}")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())