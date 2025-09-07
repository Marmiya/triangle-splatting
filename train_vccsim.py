#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCCSim Triangle Splatting Training Script

VCCSim custom training script that uses VCCSim data format instead of COLMAP
but maintains original Triangle Splatting training parameters and loop structure.

This script is called by VCCSimPanel's "Train VCCSim" button with:
- --config: JSON configuration file path
- --output: Output directory path
- --log: Log file path for progress monitoring
"""

import os
import sys
import json
import torch
import uuid
import argparse
from pathlib import Path
from datetime import datetime
from random import randint

# Add triangle splatting modules to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

try:
    from scene import Scene, TriangleModel
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from triangle_renderer import render
    from utils.general_utils import safe_state
    from utils.loss_utils import l1_loss, ssim, equilateral_regularizer, l2_loss
    from utils.image_utils import psnr
    from tqdm import tqdm
    
    # Optional tensorboard support
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_FOUND = True
    except ImportError:
        TENSORBOARD_FOUND = False
        
    # Optional lpips support
    try:
        import lpips
        LPIPS_FOUND = True
    except ImportError:
        LPIPS_FOUND = False
        
except ImportError as e:
    print(f"Error importing triangle splatting modules: {e}")
    sys.exit(1)


class VCCSimTrainingLogger:
    """Logger that outputs progress in format expected by VCCSim C++ manager"""
    
    def __init__(self, log_file_path):
        # Normalize path to use forward slashes and resolve relative paths
        self.log_file_path = os.path.normpath(os.path.abspath(log_file_path)).replace('\\', '/')
        self.ensure_log_dir()
        
    def ensure_log_dir(self):
        """Ensure log directory exists"""
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
    def log(self, message, iteration=None, loss=None, psnr_val=None):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format for C++ progress parsing
        if iteration is not None:
            if loss is not None:
                log_msg = f"[{timestamp}] Iteration {iteration}: Loss={loss:.6f}"
                if psnr_val is not None:
                    log_msg += f", PSNR={psnr_val:.2f}"
            else:
                log_msg = f"[{timestamp}] Iteration {iteration}: {message}"
        else:
            log_msg = f"[{timestamp}] {message}"
            
        print(log_msg)  # Console output
        
        # Write to log file for C++ monitoring
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")


def prepare_output_and_logger(dataset_args, log_file_path):
    """Setup output directories and logger"""
    if not dataset_args.model_path:
        unique_str = str(uuid.uuid4())
        dataset_args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset_args.model_path))
    os.makedirs(dataset_args.model_path, exist_ok=True)
    
    # Create renders directory for intermediate images
    renders_dir = os.path.join(dataset_args.model_path, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    
    # Setup logger
    logger = VCCSimTrainingLogger(log_file_path)
    
    # Setup tensorboard if available
    tb_writer = None
    if TENSORBOARD_FOUND:
        try:
            tb_writer = SummaryWriter(dataset_args.model_path)
        except:
            tb_writer = None
            
    return tb_writer, logger


def render_and_save_images(triangles, cameras, pipe, background, output_dir, iteration, logger):
    """Render and save first 5 training images at specified iterations"""
    import torchvision
    
    renders_dir = os.path.join(output_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, camera in enumerate(cameras):
            if idx >= 5:  # Only render first 5 images
                break
                
            # Render image
            render_result = render(camera, triangles, pipe, background)
            rendered_image = render_result["render"]
            
            # Ground truth image
            gt_image = camera.original_image[0:3, :, :]
            
            # Save rendered and ground truth images
            render_filename = f"iter_{iteration:06d}_cam_{idx:02d}_render.png"
            gt_filename = f"iter_{iteration:06d}_cam_{idx:02d}_gt.png"
            
            render_path = os.path.join(renders_dir, render_filename)
            gt_path = os.path.join(renders_dir, gt_filename)
            
            torchvision.utils.save_image(rendered_image, render_path)
            torchvision.utils.save_image(gt_image, gt_path)
            
        logger.log(f"Saved 5 rendered images at iteration {iteration}")


def training(dataset_args, opt, pipe, no_dome, outdoor, testing_iterations, save_iterations, 
            checkpoint_path, debug_from, logger, tb_writer):
    """Main training loop - maintains original Triangle Splatting structure"""
    
    first_iter = 0
    
    logger.log("Initializing Triangle Splatting training with VCCSim data")
    logger.log(f"Dataset source: {dataset_args.source_path}")
    logger.log(f"Output directory: {dataset_args.model_path}")
    logger.log(f"Max iterations: {opt.iterations}")
    
    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points, opt.set_sigma, no_dome)
    triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma, opt.lr_triangles_points_init)
    
    # Load checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.log(f"Loading checkpoint: {checkpoint_path}")
        (model_params, first_iter) = torch.load(checkpoint_path)
        triangles.restore(model_params, opt)

    # Background color
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Get cameras
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras() if hasattr(scene, 'getTestCameras') else []
    
    logger.log(f"Training cameras: {len(train_cameras)}")
    logger.log(f"Test cameras: {len(test_cameras)}")
    logger.log(f"Total triangles: {triangles.get_triangles_points.shape[0]}")

    # CUDA timing events
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = train_cameras.copy()
    number_of_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    # Disable tqdm progress bar to avoid dual logging formats
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Training state variables (from original)
    total_dead = 0
    opacity_now = True
    new_round = False
    removed_them = False
    large_scene = triangles.large

    # Loss function selection (from original)
    if large_scene and outdoor:
        loss_fn = l2_loss
        logger.log("Using L2 loss for large outdoor scene")
    else:
        loss_fn = l1_loss
        logger.log("Using L1 loss")

    # Optional LPIPS loss
    lpips_fn = None
    if LPIPS_FOUND:
        try:
            lpips_fn = lpips.LPIPS(net='vgg').cuda()
            logger.log("LPIPS loss initialized")
        except:
            logger.log("LPIPS initialization failed, skipping")

    logger.log("Starting training iterations...")
    
    for iteration in range(first_iter, opt.iterations + 1):
        
        iter_start.record()

        triangles.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Refill viewpoint stack if empty
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        # Pick a random camera
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render and save images at iteration 1 and every 1000 iterations
        if iteration == 1 or iteration % 1000 == 0:
            logger.log(f"Rendering first 5 images at iteration {iteration}")
            render_and_save_images(triangles, train_cameras[:5], pipe, background, 
                                 scene.model_path, iteration, logger)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Random background for training
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # Density and scaling tracking (from original)
        triangle_area = render_pkg["density_factor"].detach()
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        # Loss computation (matching original Triangle Splatting)
        gt_image = viewpoint_cam.original_image.cuda()
        pixel_loss = loss_fn(image, gt_image)
        
        # SSIM loss (from original)
        ssim_value = ssim(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)

        # Opacity regularization (from original)  
        loss_opacity = torch.abs(triangles.get_opacity).mean() * opt.lambda_opacity

        # Normal and distortion losses (from original Triangle Splatting)
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        rend_dist = render_pkg["rend_dist"]
        dist_loss = lambda_dist * (rend_dist).mean()
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        # Size regularization (from original Triangle Splatting)
        loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean()
        loss_size = loss_size * opt.lambda_size

        # Complete loss computation (matching original)
        if iteration < opt.densify_until_iter:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_size
        else:
            loss = loss_image + loss_opacity + normal_loss + dist_loss

        # Ensure loss is scalar before backward
        if loss.dim() > 0:
            loss = loss.mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress reporting
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # Log every 10 iterations
            if iteration % 10 == 0:
                psnr_val = psnr(image, gt_image).mean()
                logger.log("", iteration=iteration, loss=ema_loss_for_log, psnr_val=psnr_val.item())
                
                # Tensorboard logging
                if tb_writer is not None:
                    tb_writer.add_scalar('train_loss_patches/l1_loss', pixel_loss.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/psnr', psnr_val.item(), iteration)

            # Progress bar disabled to avoid dual logging
            # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            # progress_bar.update(1)
            
            # Optimizer step
            triangles.optimizer.step()
            triangles.optimizer.zero_grad(set_to_none=True)

            # Save checkpoints
            if iteration in save_iterations:
                logger.log(f"Saving checkpoint at iteration {iteration}")
                torch.save((triangles.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # Testing and validation
        if iteration in testing_iterations:
            logger.log(f"Running validation at iteration {iteration}")
            validation_loss = 0.0
            validation_psnr = 0.0
            test_count = min(5, len(test_cameras)) if test_cameras else min(5, len(train_cameras))
            test_cams = test_cameras[:test_count] if test_cameras else train_cameras[:test_count]
            
            with torch.no_grad():
                for idx, test_cam in enumerate(test_cams):
                    test_render = render(test_cam, triangles, pipe, background)["render"]
                    test_gt = test_cam.original_image.cuda()
                    
                    test_loss = l1_loss(test_render, test_gt)
                    test_psnr = psnr(test_render, test_gt).mean()
                    
                    validation_loss += test_loss.item()
                    validation_psnr += test_psnr.item()
                    
            validation_loss /= test_count
            validation_psnr /= test_count
            
            logger.log(f"Validation - Loss: {validation_loss:.6f}, PSNR: {validation_psnr:.2f}", iteration=iteration)
            
            if tb_writer is not None:
                tb_writer.add_scalar('validation/loss', validation_loss, iteration)
                tb_writer.add_scalar('validation/psnr', validation_psnr, iteration)

    # Final checkpoint
    logger.log("Training completed, saving final model")
    torch.save((triangles.capture(), iteration), scene.model_path + "/chkpnt_final.pth")
    
    if tb_writer is not None:
        tb_writer.close()
        
    logger.log(f"Training finished after {opt.iterations} iterations")
    logger.log(f"Final model saved to: {scene.model_path}")


def main():
    parser = argparse.ArgumentParser(description="VCCSim Triangle Splatting Training")
    parser.add_argument("--workspace", required=True, help="Training session workspace directory")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--debug_from", type=int, default=-1, help="Debug from iteration")
    parser.add_argument("--no_dome", action="store_true", help="Disable dome")
    parser.add_argument("--outdoor", default=True, action="store_true", help="Outdoor scene flag")
    
    args = parser.parse_args()
    
    # Derive paths from workspace - normalize all paths
    workspace_normalized = os.path.normpath(os.path.abspath(args.workspace))
    config_file = os.path.join(workspace_normalized, "config", "vccsim_training_config.json")
    # Use separate log file to avoid conflicts with C++ TriangleSplattingManager
    log_file = os.path.join(workspace_normalized, "python_training.log")
    output_dir = workspace_normalized
    
    print("=== VCCSim Triangle Splatting Training ===")
    print(f"Workspace (normalized): {workspace_normalized}")
    print(f"Config file: {config_file}")
    print(f"Log file: {log_file}")
    
    # Validate inputs
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file not found: {config_file}")
        sys.exit(1)
        
    # Load VCCSim configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config file: {e}")
        sys.exit(1)
    
    # Initialize system state
    safe_state(False)
    
    # Initialize CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create parameter objects with original Triangle Splatting defaults
        model_parser = argparse.ArgumentParser()
        model = ModelParams(model_parser, sentinel=True)
        opt_parser = argparse.ArgumentParser()
        opt = OptimizationParams(opt_parser)
        pipe_parser = argparse.ArgumentParser()
        pipe = PipelineParams(pipe_parser)
        
        # Create VCCSim scene arguments
        class VCCSimSceneArgs:
            def __init__(self, config, output_dir, config_file_path):
                # VCCSim data paths - source_path should point to the config directory
                # where both camera_info.json and vccsim_training_config.json are located
                self.source_path = os.path.dirname(config_file_path)  # Config directory
                self.images = config.get('image_directory', '')  # Image directory
                self.model_path = output_dir
                
                # Camera parameters
                self.sh_degree = 3
                self.white_background = False
                self.resolution = -1
                self.data_device = "cuda"
                self.eval = False
        
        scene_args = VCCSimSceneArgs(config, output_dir, config_file)
        
        # Extract parameters
        dataset_params = model.extract(scene_args)
        
        # Prepare output and logger
        tb_writer, logger = prepare_output_and_logger(dataset_params, log_file)
        
        # Training parameters - use original defaults
        testing_iterations = [7000, 30000]  # Original testing schedule
        save_iterations = [7000, 30000]     # Original save schedule
        
        logger.log("Starting VCCSim Triangle Splatting training")
        logger.log(f"Using VCCSim data from: {scene_args.source_path}")
        logger.log(f"Images from: {scene_args.images}")
        
        # Start training with original parameters
        training(
            dataset_params,
            opt,  # Original optimization parameters
            pipe,  # Original pipeline parameters  
            args.no_dome,
            args.outdoor,
            testing_iterations,
            save_iterations,
            args.checkpoint,
            args.debug_from,
            logger,
            tb_writer
        )
        
        logger.log("VCCSim training completed successfully")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Log error for C++ monitoring
        try:
            logger = VCCSimTrainingLogger(log_file)
            logger.log(f"[ERROR] {error_msg}")
        except:
            pass
            
        import traceback
        print(f"[DEBUG] Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    print("[OK] Training completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()