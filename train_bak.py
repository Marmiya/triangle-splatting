#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, equilateral_regularizer, l2_loss
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips


def training(
        dataset,   
        opt, 
        pipe,
        no_dome, 
        outdoor,
        testing_iterations,
        save_iterations,
        checkpoint, 
        debug_from,
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points, opt.set_sigma, no_dome)
    triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma, opt.lr_triangles_points_init)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    total_dead = 0

    opacity_now = True

    new_round = False
    removed_them = False

    large_scene = triangles.large

    if large_scene and outdoor:
        loss_fn = l2_loss
    else:
        loss_fn = l1_loss

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        triangles.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # largest distance from point to center of image
        triangle_area = render_pkg["density_factor"].detach()
        # largest distance from point after applying sigma to center of image
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        pixel_loss = loss_fn(image, gt_image)

        ##############################################################
        # WE ADD A LOSS FORCING LOW OPACITIES                        #
        ##############################################################
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # loss opacity
        loss_opacity = torch.abs(triangles.get_opacity).mean() * args.lambda_opacity

        # loss normal and distortion
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0 # 0.001
        rend_dist = render_pkg["rend_dist"]
        dist_loss = lambda_dist * (rend_dist).mean()
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean() 
        loss_size = loss_size * opt.lambda_size


        if iteration < opt.densify_until_iter:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_size
        else:
            loss = loss_image + loss_opacity + normal_loss + dist_loss

        loss.backward()
     
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in save_iterations:
                print("\n[ITER {}] Saving Triangles".format(iteration))
                scene.save(iteration)
            if iteration % 1000 == 0:
                total_dead = 0

            if iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0 and iteration > opt.densify_from_iter:
                
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()

                if iteration > 1000 and not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                    
                    if not outdoor:
                        mask_test = triangles.image_size > 1400
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                          

                total_dead += dead_mask.sum()

                if opt.proba_distr == 0:
                    oddGroup = True
                elif opt.proba_distr == 1:
                    oddGroup = False
                else:
                    if opacity_now:
                        oddGroup = opacity_now
                        opacity_now = False
                    else:
                        oddGroup = opacity_now
                        opacity_now = True

                removed_them = True
                new_round = False

                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=oddGroup, dead_mask=dead_mask)


            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()


                if not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                triangles.remove_final_points(dead_mask)
                removed_them = True
                new_round = False

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                
    print("Training is done")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.triangles.get_triangles_points.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.no_dome,
             args.outdoor,
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             )
    
    # All done
    print("\nTraining complete.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    from PIL import Image
    import torch.nn.functional as F
    import torchvision
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
            
            # Extract quaternion and translation from VCCSim data  
            rotation_quat = camera_info['rotation']  # 4-element quaternion [x, y, z, w]
            quat = np.array(rotation_quat).astype(np.float32)  # VCCSim quaternion format: [x, y, z, w]
            T = np.array(camera_info['translation']).astype(np.float32)  # Camera position in world coords
            
            # Convert quaternion to rotation matrix
            # VCCSim quaternion format: [x, y, z, w] (UE format)
            qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
            
            # Build camera-to-world rotation matrix from quaternion (same as 3DGS)
            R = np.array([
                [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
                [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ], dtype=np.float32)
            
            # Debug: Verify data for first camera
            if idx == 0:
                print(f"[DEBUG] Camera {idx} VCCSim data (3DGS compatible format):")
                print(f"  Quaternion [x,y,z,w]: {quat}")
                print(f"  R (camera-to-world):\n{R}")
                print(f"  T (translation vector): {T}")
                print(f"  FOV: {np.degrees(fov_x_rad):.2f}° x {np.degrees(fov_y_rad):.2f}°")
                print(f"  Image size: {width} x {height}")
                print(f"  Focal lengths: fx={focal_x:.2f}, fy={focal_y:.2f}")
            
            camera = Camera(
                colmap_id=idx,
                R=R,  # Camera-to-world rotation matrix (same as 3DGS expectation)
                T=T,  # Translation vector (same as 3DGS expectation)
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
    
    print(f"[DEBUG] Scene statistics:")
    print(f"  Camera positions shape: {camera_positions.shape}")
    print(f"  Scene center: {scene_center}")
    print(f"  Scene radius: {scene_radius}")
    print(f"  Camera position range:")
    print(f"    X: [{np.min(camera_positions[:, 0]):.2f}, {np.max(camera_positions[:, 0]):.2f}]")
    print(f"    Y: [{np.min(camera_positions[:, 1]):.2f}, {np.max(camera_positions[:, 1]):.2f}]")
    print(f"    Z: [{np.min(camera_positions[:, 2]):.2f}, {np.max(camera_positions[:, 2]):.2f}]")
    
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


def render_debug_images(scene, triangle_model: TriangleModel, pipe_params: PipelineParams, 
                       background: torch.Tensor, iteration: int, output_dir: str, device: torch.device):
    """Render debug images from the first 5 camera poses - using torchvision.utils.save_image"""
    
    try:
        # Create debug directory structure
        debug_dir = os.path.join(output_dir, "debug_renders")
        iteration_dir = os.path.join(debug_dir, f"iteration_{iteration:06d}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Get first 5 training cameras
        train_cameras = scene.getTrainCameras()
        num_cameras_to_render = min(5, len(train_cameras))
        
        print(f"[DEBUG] Rendering {num_cameras_to_render} debug images at iteration {iteration}")
        
        with torch.no_grad():
            for cam_idx in range(num_cameras_to_render):
                camera = train_cameras[cam_idx]
                
                print(f"[DEBUG] Camera {cam_idx} original image shape: {camera.original_image.shape}")
                
                # Render image
                render_pkg = render(camera, triangle_model, pipe_params, background)
                rendered_image = render_pkg["render"]
                
                print(f"[DEBUG] Rendered image shape: {rendered_image.shape}")
                print(f"[DEBUG] Rendered image range: [{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
                
                # Save rendered image using torchvision (matches original TS implementation)
                render_path = os.path.join(iteration_dir, f"camera_{cam_idx:02d}_render.png")
                torchvision.utils.save_image(rendered_image, render_path)
                
                # Save ground truth for comparison (use RGB channels only)
                if hasattr(camera, 'original_image'):
                    gt_image = camera.original_image[0:3, :, :]  # Take RGB channels [C, H, W]
                    gt_path = os.path.join(iteration_dir, f"camera_{cam_idx:02d}_gt.png")
                    torchvision.utils.save_image(gt_image, gt_path)
                    
                    print(f"[DEBUG] GT image shape: {gt_image.shape}")
                    print(f"[DEBUG] GT image range: [{gt_image.min():.3f}, {gt_image.max():.3f}]")
                
                print(f"[OK] Saved camera {cam_idx} renders")
                
        print(f"[DEBUG] Debug images saved to: {iteration_dir}")
        
    except Exception as e:
        print(f"[WARNING] Failed to render debug images at iteration {iteration}: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")


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
        
        initial_loss = None
        loss_not_improving_count = 0
        
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
            
            # Check for training issues early on
            if iteration <= 10:
                if initial_loss is None:
                    initial_loss = metrics['loss']
                    print(f"[DEBUG] Initial loss: {initial_loss:.6f}")
                elif metrics['loss'] > initial_loss * 2:
                    print(f"[WARNING] Loss exploding at iteration {iteration}: {metrics['loss']:.6f} vs initial {initial_loss:.6f}")
                elif np.isnan(metrics['loss']) or np.isinf(metrics['loss']):
                    print(f"[ERROR] Invalid loss at iteration {iteration}: {metrics['loss']}")
                    break
            
            # Update progress with triangle count
            triangle_count = triangle_model.get_triangles_points.shape[0]
            progress_bar.set_postfix({
                'Loss': f"{metrics['loss']:.6f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'Triangles': f"{triangle_count}"
            })
            
            # Render sample images every 200 iterations for debugging
            if iteration % 200 == 0:
                render_debug_images(scene, triangle_model, PipelineParams(ArgumentParser()), 
                                  background, iteration, config['output_directory'], device)
            
            # Log progress periodically  
            if iteration % 100 == 0 or iteration == max_iterations:
                log_message = f"Iteration {iteration}/{max_iterations} - Loss: {metrics['loss']:.6f}, L1: {metrics['l1_loss']:.6f}, PSNR: {metrics['psnr']:.2f}\n"
                
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
            
            # Periodic checkpoint saving to organized runtime directory
            if iteration > 0 and (iteration % 5000 == 0 or iteration == max_iterations - 1):
                print(f"Saving checkpoint at iteration {iteration}...")
                
                # Create runtime directory for checkpoints
                runtime_dir = os.path.join(output_dir, "runtime")
                os.makedirs(runtime_dir, exist_ok=True)
                
                # Save checkpoint PLY
                checkpoint_path = os.path.join(runtime_dir, f"checkpoint_{iteration:06d}.ply")
                triangle_model.save_ply(checkpoint_path)
                
                # Save intermediate result to outputs for inspection
                if iteration % 10000 == 0:  # Save major milestones to outputs too
                    outputs_dir = os.path.join(output_dir, "outputs")
                    os.makedirs(outputs_dir, exist_ok=True)
                    milestone_path = os.path.join(outputs_dir, f"model_iter_{iteration:06d}.ply")
                    triangle_model.save_ply(milestone_path)
                    print(f"[MILESTONE] Saved model at iteration {iteration} to outputs")
        
        # Save final model to organized outputs directory
        print("Saving final model...")
        outputs_dir = os.path.join(output_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        model_path = os.path.join(outputs_dir, "final_model.ply")
        triangle_model.save_ply(model_path)
        
        # Also save training metrics summary
        metrics_path = os.path.join(outputs_dir, "training_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"VCCSim Triangle Splatting Training Results\n")
            f.write(f"=========================================\n\n")
            f.write(f"Total iterations: {max_iterations}\n")
            f.write(f"Final loss: {loss_value:.6f}\n")
            if 'psnr_value' in locals():
                f.write(f"Final PSNR: {psnr_value:.2f} dB\n")
            f.write(f"Model saved to: {model_path}\n")
            f.write(f"Training completed successfully\n")
        
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