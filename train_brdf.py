#!/usr/bin/env python3
"""
Triangle-BRDF Training Script

Train Triangle Splatting with BRDF materials instead of Spherical Harmonics
Based on train_vccsim.py but with BRDF model and renderer

Usage:
    python train_brdf.py --workspace <path/to/workspace>

Or with custom config:
    python train_brdf.py --workspace <path> --iterations 30000
"""

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9+PTX'

import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import json
import torch
import uuid
import argparse
from pathlib import Path
from datetime import datetime
from random import randint

# Add to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import Triangle Splatting modules
from scene import Scene
from scene.triangle_brdf_model import TriangleBRDFModel  # Use BRDF model
from arguments import ModelParams, PipelineParams, OptimizationParams
from triangle_renderer.brdf_renderer import render_brdf
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, equilateral_regularizer, l2_loss
from utils.image_utils import psnr
from utils.vccsim_logger import VCCSimTrainingLogger, VCCSimLoggingConfig

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except:
    TENSORBOARD_FOUND = False

try:
    import lpips
    LPIPS_FOUND = True
except:
    LPIPS_FOUND = False


def prepare_output_and_logger(args):
    """Setup output directories and logger"""
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)

    # Write config
    from argparse import Namespace
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))

    # Setup logger
    log_file = os.path.join(args.model_path, "python_training.log")
    logger = VCCSimTrainingLogger(log_file)
    logging_config = VCCSimLoggingConfig.default_performance_optimized()

    # Setup tensorboard
    tb_writer = None
    if TENSORBOARD_FOUND:
        try:
            tb_writer = SummaryWriter(args.model_path)
        except:
            pass

    return tb_writer, logger, logging_config


def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed,
                   testing_iterations, scene, renderFunc, renderArgs, logger):
    """Training report with test view rendering"""
    if tb_writer:
        tb_writer.add_scalar('train_loss/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        configs = [
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                         for idx in range(5, 30, 5)]}
        ]

        for config in configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and idx < 5:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render",
                                           image[None], iteration)

                    if idx < 5:
                        import torchvision
                        render_dir = os.path.join(scene.model_path, f"debug_renders_iter_{iteration}")
                        os.makedirs(render_dir, exist_ok=True)

                        render_path = os.path.join(render_dir, f'{config["name"]}_view_{idx:02d}_render.png')
                        torchvision.utils.save_image(image, render_path)

                        if iteration == testing_iterations[0]:
                            gt_path = os.path.join(render_dir, f'{config["name"]}_view_{idx:02d}_gt.png')
                            torchvision.utils.save_image(gt_image, gt_path)

                        logger.log_debug_render_save(render_path)

                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                pixel_loss_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                if config['name'] == 'train':
                    logger.update_psnr_cache(train_psnr=psnr_test.item())
                elif config['name'] == 'test':
                    logger.update_psnr_cache(test_psnr=psnr_test.item())

                logger.log_evaluation_results(iteration, config['name'],
                                            pixel_loss_test, psnr_test, ssim_test, 0.0)

        torch.cuda.empty_cache()


def get_model_memory_stats(model):
    """Calculate detailed memory statistics for model parameters"""
    total_params = 0
    param_breakdown = {}

    for name, param in [
        ('base_color', model._base_color),
        ('roughness', model._roughness),
        ('metallic', model._metallic),
        ('opacity', model._opacity),
        ('triangles_points', model._triangles_points),
        ('sigma', model._sigma),
        ('mask', model._mask)
    ]:
        if param.numel() > 0:
            count = param.numel()
            size_mb = count * param.element_size() / (1024 * 1024)
            param_breakdown[name] = {
                'count': count,
                'size_mb': size_mb,
                'shape': tuple(param.shape)
            }
            total_params += count

    total_size_mb = sum(p['size_mb'] for p in param_breakdown.values())

    return {
        'total_params': total_params,
        'total_size_mb': total_size_mb,
        'breakdown': param_breakdown
    }


def training(dataset, opt, pipe, no_dome, outdoor, testing_iterations,
            save_iterations, checkpoint, debug_from, logger, logging_config, tb_writer):
    """Main BRDF training function"""

    first_iter = 0
    triangles = TriangleBRDFModel()

    # Create scene
    scene = Scene(dataset, triangles, opt.set_opacity, opt.triangle_size,
                 opt.nb_points, opt.set_sigma, no_dome)

    # Setup training
    triangles.training_setup(opt, opt.feature_lr, opt.opacity_lr,
                            opt.lr_sigma, opt.lr_triangles_points_init)

    # Print initial memory statistics
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"\n[MEMORY] Initial GPU memory: {initial_memory:.3f} GB")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    # Configure UE-compatible lighting (sunlight from above)
    light_dir_unnorm = torch.tensor([0., 0., -1.], device="cuda")
    pipe.light_direction = light_dir_unnorm / torch.norm(light_dir_unnorm)
    pipe.light_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    pipe.ambient_color = torch.tensor([0.2, 0.2, 0.2], device="cuda")

    print(f"\n[BRDF LIGHTING CONFIGURATION]")
    print(f"Light Direction: {pipe.light_direction.cpu().numpy()}")
    print(f"Light Color: {pipe.light_color.cpu().numpy()}")
    print(f"Ambient Color: {pipe.ambient_color.cpu().numpy()}")
    print(f"Note: Ambient reduced to 20% to allow BRDF effects to be visible")

    def render_brdf_wrapper(viewpoint, model, pipe_cfg, bg):
        return render_brdf(viewpoint, model, pipe_cfg, bg)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    first_iter += 1

    initial_triangle_count = triangles.get_triangles_points.shape[0]
    logger.log_initial_stats(initial_triangle_count)

    model_stats = get_model_memory_stats(triangles)
    model_memory = torch.cuda.memory_allocated() / (1024 ** 3)

    print(f"\n[BRDF MODEL INITIALIZATION]")
    print(f"Triangles: {initial_triangle_count:,}")
    print(f"Total parameters: {model_stats['total_params']:,} ({model_stats['total_size_mb']:.2f} MB)")
    print(f"Per-triangle: {model_stats['total_params'] / initial_triangle_count:.1f} params")
    print(f"GPU memory: {model_memory:.3f} GB (+{model_memory - initial_memory:.3f} GB)")

    triangles.triangle_area = torch.zeros(initial_triangle_count, device="cuda")
    triangles.image_size = torch.zeros(initial_triangle_count, device="cuda")
    triangles.importance_score = torch.zeros(initial_triangle_count, device="cuda")

    last_speed_report_time = datetime.now()
    last_speed_report_iter = first_iter

    total_dead = 0
    opacity_now = True
    new_round = False
    removed_them = False
    large_scene = triangles.large

    loss_fn = l2_loss if (large_scene and outdoor) else l1_loss

    print(f"\nStarting training: {initial_triangle_count} triangles, {opt.iterations} iterations")

    # ========== Training Loop ==========
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        triangles.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ========== Render with BRDF ==========
        render_pkg = render_brdf_wrapper(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        triangle_area = torch.zeros(triangles.get_triangles_points.shape[0], device="cuda")
        image_size = torch.zeros(triangles.get_triangles_points.shape[0], device="cuda")
        importance_score = torch.zeros(triangles.get_triangles_points.shape[0], device="cuda")

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        # Loss computation
        gt_image = viewpoint_cam.original_image.cuda()
        pixel_loss = loss_fn(image, gt_image)

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_opacity = torch.abs(triangles.get_opacity).mean() * opt.lambda_opacity

        loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean()
        loss_size = loss_size * opt.lambda_size

        if iteration < opt.densify_until_iter:
            loss = loss_image + loss_opacity + loss_size
        else:
            loss = loss_image + loss_opacity

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            current_triangle_count = triangles.get_triangles_points.shape[0]

            current_speed = None
            if iteration % 10 == 0:
                current_time = datetime.now()
                time_diff = (current_time - last_speed_report_time).total_seconds()
                iter_diff = iteration - last_speed_report_iter
                if time_diff > 0 and iter_diff > 0:
                    current_speed = iter_diff / time_diff
                    last_speed_report_time = current_time
                    last_speed_report_iter = iteration

            if logging_config.should_log_loss(iteration):
                logger.log_loss_update(iteration, ema_loss_for_log, current_speed)

            if logging_config.should_log_triangle_stats(iteration):
                logger.log_triangle_stats(iteration, current_triangle_count)

            # Memory monitoring at intervals
            if iteration % 1000 == 0:
                current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[MEMORY] Iter {iteration}: Current {current_memory:.3f} GB, Peak {peak_memory:.3f} GB")

            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn,
                          iter_start.elapsed_time(iter_end), testing_iterations,
                          scene, render_brdf_wrapper, (pipe, background), logger)

            if iteration in save_iterations:
                logger.log_checkpoint_save(iteration)
                scene.save(iteration)

            # ========== DENSIFICATION AND PRUNING DISABLED ==========
            # TODO: Redesign densification/pruning strategy for BRDF training
            # The standard Triangle Splatting strategy needs adaptation for BRDF parameters

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    # Final statistics
    final_triangle_count = triangles.get_triangles_points.shape[0]
    logger.log_final_stats(final_triangle_count, initial_triangle_count)

    # Final memory and model statistics
    final_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final_model_stats = get_model_memory_stats(triangles)

    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL STATISTICS")
    print("="*80)

    # Triangle statistics
    print(f"\nTriangle Count:")
    print(f"  Initial: {initial_triangle_count:,}")
    print(f"  Final:   {final_triangle_count:,}")
    print(f"  Change:  {final_triangle_count - initial_triangle_count:+,} ({(final_triangle_count/initial_triangle_count - 1)*100:+.1f}%)")

    # Memory statistics
    print(f"\nGPU Memory Usage:")
    print(f"  Final:   {final_memory:.3f} GB")
    print(f"  Peak:    {peak_memory:.3f} GB")
    print(f"  Model:   {final_model_stats['total_size_mb']:.2f} MB")

    # Parameter statistics
    print(f"\nModel Parameters:")
    print(f"  Total:   {final_model_stats['total_params']:,}")
    print(f"  Per-triangle: {final_model_stats['total_params'] / final_triangle_count:.1f}")

    print(f"\nBRDF Parameter Breakdown:")
    for name, info in final_model_stats['breakdown'].items():
        print(f"  {name:20s}: {info['count']:12,} params ({info['size_mb']:8.2f} MB) {info['shape']}")

    triangles.print_brdf_statistics()

    logger.log_completion()
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triangle-BRDF Training")

    parser.add_argument("--workspace", help="Training workspace directory", default="C:\\UEProjects\\VCCSimDev\\Saved\\RatSplatting\\RatSplatting\\prepared_session_20251031_201608")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                       default=[1, 1000, 3000, 5000, 7000, 15000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--no_dome", action="store_true", default=True)
    parser.add_argument("--outdoor", action="store_true", default=True)

    args = parser.parse_args(sys.argv[1:])

    # Load config
    workspace_normalized = os.path.normpath(os.path.abspath(args.workspace))
    config_file = os.path.join(workspace_normalized, "config", "vccsim_training_config.json")

    if not os.path.exists(config_file):
        print(f"[ERROR] Config file not found: {config_file}")
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)

    args.source_path = os.path.dirname(config_file)
    args.images = config.get('image_directory', '')
    args.model_path = workspace_normalized

    if 'training' in config and 'max_iterations' in config['training']:
        args.iterations = config['training']['max_iterations']

    args.save_iterations.append(args.iterations)

    print(f"[BRDF] Optimizing {args.model_path}")

    # Initialize LPIPS
    global lpips_fn
    lpips_fn = None
    if LPIPS_FOUND:
        try:
            lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")
        except:
            pass

    # Initialize system
    safe_state(args.quiet)

    # Prepare output and logger
    tb_writer, logger, logging_config = prepare_output_and_logger(args)

    # Run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
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
        tb_writer
    )

    print("\n[BRDF] Training script completed successfully!")
