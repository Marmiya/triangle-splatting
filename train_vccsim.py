#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCCSim Triangle Splatting Training Script

VCCSim custom training script that uses VCCSim data format instead of COLMAP
but maintains original Triangle Splatting training parameters and loop structure.

This script is called by VCCSimPanel's "Train VCCSim" button with:
- --workspace: Training session workspace directory
"""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Import VCCSim logger utilities
from utils.vccsim_logger import VCCSimTrainingLogger, VCCSimLoggingConfig

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

def prepare_output_and_logger(args):
    """Setup output directories and logger - based on original train.py"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    
    # Write cfg_args file exactly like original train.py
    from argparse import Namespace
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # Setup logger with performance-optimized configuration
    log_file = os.path.join(args.model_path, "python_training.log")
    logger = VCCSimTrainingLogger(log_file)
    logging_config = VCCSimLoggingConfig.default_performance_optimized()
    
    # Setup tensorboard if available
    tb_writer = None
    if TENSORBOARD_FOUND:
        try:
            tb_writer = SummaryWriter(args.model_path)
        except:
            tb_writer = None
            
    return tb_writer, logger, logging_config


def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, logger):
    """Training report function based on original train.py but with VCCSim logger"""
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
                    
                    # Save PNG files directly for easy viewing (first 5 images of each config)
                    if idx < 5:
                        import torchvision
                        render_dir = os.path.join(scene.model_path, f"debug_renders_iter_{iteration}")
                        os.makedirs(render_dir, exist_ok=True)
                        
                        # Save rendered image
                        render_path = os.path.join(render_dir, f'{config["name"]}_view_{idx:02d}_render.png')
                        torchvision.utils.save_image(image, render_path)
                        
                        # Save ground truth (only for first iteration to avoid duplicates)
                        if iteration == testing_iterations[0]:
                            gt_path = os.path.join(render_dir, f'{config["name"]}_view_{idx:02d}_gt.png')
                            torchvision.utils.save_image(gt_image, gt_path)
                        
                        logger.log_debug_render_save(render_path)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    if lpips_fn:
                        lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                
                # Update PSNR cache for progress logging
                if config['name'] == 'train':
                    logger.update_psnr_cache(train_psnr=psnr_test.item())
                elif config['name'] == 'test':
                    logger.update_psnr_cache(test_psnr=psnr_test.item())
                
                # Use specialized logging method for evaluation results
                logger.log_evaluation_results(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.triangles.get_triangles_points.shape[0], iteration)
        torch.cuda.empty_cache()


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
        logger,
        logging_config,
        tb_writer
        ):
    """Main training function based on original train.py but with VCCSim logger"""
    
    first_iter = 0

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
    first_iter += 1
    
    # Variables for triangle count and speed tracking
    initial_triangle_count = triangles.get_triangles_points.shape[0]
    logger.log_initial_stats(initial_triangle_count)
    last_speed_report_time = datetime.now()
    last_speed_report_iter = first_iter

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
        loss_opacity = torch.abs(triangles.get_opacity).mean() * opt.lambda_opacity

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
            # Progress tracking and logging
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            current_triangle_count = triangles.get_triangles_points.shape[0]
            
            # Calculate speed for loss logging (every few iterations to avoid overhead)
            current_speed = None
            if iteration % 10 == 0:  # Calculate speed every 10 iterations
                current_time = datetime.now()
                time_diff = (current_time - last_speed_report_time).total_seconds()
                iter_diff = iteration - last_speed_report_iter
                if time_diff > 0 and iter_diff > 0:
                    current_speed = iter_diff / time_diff
                    last_speed_report_time = current_time
                    last_speed_report_iter = iteration
            
            # Log loss updates at high frequency with speed information
            if logging_config.should_log_loss(iteration):
                logger.log_loss_update(iteration, ema_loss_for_log, current_speed)
            
            # Log PSNR updates (when available and at configured frequency)
            if logging_config.should_log_psnr(iteration) and logger.latest_train_psnr is not None:
                logger.log_psnr_update(iteration, logger.latest_train_psnr)
            
            # Log triangle statistics (removed redundant triangle count log)
            if logging_config.should_log_triangle_stats(iteration):
                logger.log_triangle_stats(iteration, current_triangle_count)
            
            # Optional frequent PSNR calculation (configurable)
            if logging_config.should_calculate_frequent_psnr(iteration):
                torch.cuda.empty_cache()  # Clear memory before PSNR calculation
                
                # Quick PSNR calculation on a single training view (minimal impact)
                if len(scene.getTrainCameras()) > 0:
                    test_viewpoint = scene.getTrainCameras()[0]  # Use first training view
                    with torch.no_grad():
                        test_image = torch.clamp(render(test_viewpoint, triangles, pipe, background)["render"], 0.0, 1.0)
                        test_gt = torch.clamp(test_viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        current_psnr = psnr(test_image, test_gt).mean().item()
                        
                        # Update cached PSNR for progress logging
                        logger.update_psnr_cache(train_psnr=current_psnr)
                        logger.log(f"Quick PSNR check: {current_psnr:.2f}", iteration)
                
                torch.cuda.empty_cache()  # Clear memory after PSNR calculation

            # Log and save
            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), logger)
            if iteration in save_iterations:
                logger.log_checkpoint_save(iteration)
                scene.save(iteration)
            if iteration % 1000 == 0:
                total_dead = 0

            if iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0 and iteration > opt.densify_from_iter:
                
                triangle_count_before = triangles.get_triangles_points.shape[0]
                
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < opt.importance_threshold).squeeze(),(triangles.get_opacity <= opt.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < opt.importance_threshold).squeeze(),(triangles.get_opacity <= opt.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

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
                
                triangle_count_after = triangles.get_triangles_points.shape[0]
                dead_count = dead_mask.sum().item()
                logger.log_densification_stats(iteration, dead_count, triangle_count_before, triangle_count_after)


            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                triangle_count_before = triangles.get_triangles_points.shape[0]
                
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < opt.importance_threshold).squeeze(),(triangles.get_opacity <= opt.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < opt.importance_threshold).squeeze(),(triangles.get_opacity <= opt.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()


                if not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                triangles.remove_final_points(dead_mask)
                removed_them = True
                new_round = False
                
                triangle_count_after = triangles.get_triangles_points.shape[0]
                removed_count = triangle_count_before - triangle_count_after
                logger.log_pruning_stats(iteration, removed_count, triangle_count_before, triangle_count_after)

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                
    # Final triangle count report
    final_triangle_count = triangles.get_triangles_points.shape[0]
    logger.log_final_stats(final_triangle_count, initial_triangle_count)
    logger.log_completion()


if __name__ == "__main__":
    # Set up command line argument parser like original train.py but with VCCSim workspace
    parser = argparse.ArgumentParser(description="VCCSim Triangle Splatting Training")
    
    # Add VCCSim-specific workspace argument first  
    parser.add_argument("--workspace", required=True, help="Training session workspace directory")
    
    # Add standard Triangle Splatting parameter groups like original train.py
    lp = ModelParams(parser)
    op = OptimizationParams(parser) 
    pp = PipelineParams(parser)
    
    # Add remaining arguments exactly like original train.py
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000, 3000, 5000, 7000, 15000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--no_dome", action="store_true", default=True)
    parser.add_argument("--outdoor", action="store_true", default=False)
    
    # Parse arguments once like original train.py
    args = parser.parse_args(sys.argv[1:])
    
    # Derive VCCSim paths from workspace argument
    workspace_normalized = os.path.normpath(os.path.abspath(args.workspace))
    config_file = os.path.join(workspace_normalized, "config", "vccsim_training_config.json")
    
    # Validate config file exists
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
    
    # Override paths for VCCSim data format like original train.py modifies args
    args.source_path = os.path.dirname(config_file)  # Config directory
    args.images = config.get('image_directory', '')  # Image directory
    args.model_path = workspace_normalized  # Output goes to workspace

    # Apply max_iterations from config if specified
    if 'training' in config and 'max_iterations' in config['training']:
        args.iterations = config['training']['max_iterations']
        print(f"[CONFIG] Using max_iterations from config: {args.iterations}")

    args.save_iterations.append(args.iterations)  # Like original train.py

    print("Optimizing " + args.model_path)

    # Initialize LPIPS like original train.py
    global lpips_fn
    lpips_fn = None
    if LPIPS_FOUND:
        try:
            lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")
        except Exception as e:
            print(f"Failed to initialize LPIPS: {e}")
            pass

    # Initialize system state (RNG) like original train.py
    safe_state(args.quiet)
    
    # Prepare output and logger
    tb_writer, logger, logging_config = prepare_output_and_logger(args)

    # Configure and run training exactly like original train.py
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
             logger,
             logging_config,
             tb_writer,
             )