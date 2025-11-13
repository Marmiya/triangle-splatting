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

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"\nTriangle Count: {initial_triangle_count:,}")
    print(f"Total Iterations: {opt.iterations}")
    print(f"Number of Views: {number_of_views}")
    print(f"\nDensification Parameters:")
    print(f"  densify_from_iter: {opt.densify_from_iter}")
    print(f"  densify_until_iter: {opt.densify_until_iter}")
    print(f"  densification_interval: {opt.densification_interval}")
    print(f"  importance_threshold: {opt.importance_threshold}")
    print(f"  opacity_dead: {opt.opacity_dead}")
    print(f"  max_shapes: {opt.max_shapes}")
    print(f"\nScene Configuration:")
    print(f"  large_scene: {large_scene}")
    print(f"  outdoor: {outdoor}")
    print(f"  loss_fn: {loss_fn.__name__}")
    print(f"{'='*80}\n")

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

        # Extract statistics from render package (CRITICAL: don't use zeros!)
        triangle_area = render_pkg["density_factor"].detach()
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        # Debug: Print statistics every 100 iterations
        if iteration % 100 == 0:
            print(f"\n[DEBUG ITER {iteration}] Render statistics:")
            print(f"  triangle_area    - min: {triangle_area.min().item():.6f}, max: {triangle_area.max().item():.6f}, mean: {triangle_area.mean().item():.6f}")
            print(f"  image_size       - min: {image_size.min().item():.6f}, max: {image_size.max().item():.6f}, mean: {image_size.mean().item():.6f}")
            print(f"  importance_score - min: {importance_score.min().item():.6f}, max: {importance_score.max().item():.6f}, mean: {importance_score.mean().item():.6f}")

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

            # Reset dead counter periodically
            if iteration % 1000 == 0:
                total_dead = 0

            # ========== DENSIFICATION (during early training) ==========
            if iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0 and iteration > opt.densify_from_iter:

                triangle_count_before = triangles.get_triangles_points.shape[0]

                # Print accumulated statistics
                print(f"\n[DEBUG PRE-DENSIFICATION ITER {iteration}]")
                print(f"  Accumulated statistics:")
                print(f"    triangles.triangle_area    - min: {triangles.triangle_area.min().item():.6f}, max: {triangles.triangle_area.max().item():.6f}, mean: {triangles.triangle_area.mean().item():.6f}")
                print(f"    triangles.image_size       - min: {triangles.image_size.min().item():.6f}, max: {triangles.image_size.max().item():.6f}, mean: {triangles.image_size.mean().item():.6f}")
                print(f"    triangles.importance_score - min: {triangles.importance_score.min().item():.6f}, max: {triangles.importance_score.max().item():.6f}, mean: {triangles.importance_score.mean().item():.6f}")
                print(f"    triangles.get_opacity      - min: {triangles.get_opacity.min().item():.6f}, max: {triangles.get_opacity.max().item():.6f}, mean: {triangles.get_opacity.mean().item():.6f}")

                # Calculate dead mask
                if number_of_views < 250:
                    dead_mask = torch.logical_or(
                        (triangles.importance_score < opt.importance_threshold).squeeze(),
                        (triangles.get_opacity <= opt.opacity_dead).squeeze()
                    )
                else:
                    if not new_round:
                        dead_mask = torch.logical_or(
                            (triangles.importance_score < opt.importance_threshold).squeeze(),
                            (triangles.get_opacity <= opt.opacity_dead).squeeze()
                        )
                    else:
                        dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

                # Additional pruning conditions after iteration 1000
                if iteration > 1000 and not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())

                    if not outdoor:
                        mask_test = triangles.image_size > 1400
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())

                total_dead += dead_mask.sum()

                # Debug: Print dead mask statistics
                print(f"\n[DEBUG DENSIFICATION ITER {iteration}]")
                print(f"  Triangle count before: {triangle_count_before}")
                print(f"  Dead triangles: {dead_mask.sum().item()}")
                print(f"  Dead ratio: {dead_mask.sum().item() / triangle_count_before * 100:.2f}%")
                print(f"  Criteria breakdown:")
                print(f"    - importance_score < {opt.importance_threshold}: {(triangles.importance_score < opt.importance_threshold).sum().item()}")
                print(f"    - opacity <= {opt.opacity_dead}: {(triangles.get_opacity <= opt.opacity_dead).sum().item()}")
                if iteration > 1000 and not new_round:
                    print(f"    - triangle_area < 2: {(triangles.triangle_area < 2).sum().item()}")
                    if not outdoor:
                        print(f"    - image_size > 1400: {(triangles.image_size > 1400).sum().item()}")
                print(f"  Accumulated dead count: {total_dead}")

                # Determine densification strategy
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

                # Add new triangles
                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=oddGroup, dead_mask=dead_mask)

                triangle_count_after = triangles.get_triangles_points.shape[0]
                dead_count = dead_mask.sum().item()
                logger.log_densification_stats(iteration, dead_count, triangle_count_before, triangle_count_after)

                print(f"  Triangle count after: {triangle_count_after}")
                print(f"  Net change: {triangle_count_after - triangle_count_before:+d}")

            # ========== PRUNING (after densification phase) ==========
            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                triangle_count_before = triangles.get_triangles_points.shape[0]

                # Print accumulated statistics
                print(f"\n[DEBUG PRE-PRUNING ITER {iteration}]")
                print(f"  Accumulated statistics:")
                print(f"    triangles.triangle_area    - min: {triangles.triangle_area.min().item():.6f}, max: {triangles.triangle_area.max().item():.6f}, mean: {triangles.triangle_area.mean().item():.6f}")
                print(f"    triangles.image_size       - min: {triangles.image_size.min().item():.6f}, max: {triangles.image_size.max().item():.6f}, mean: {triangles.image_size.mean().item():.6f}")
                print(f"    triangles.importance_score - min: {triangles.importance_score.min().item():.6f}, max: {triangles.importance_score.max().item():.6f}, mean: {triangles.importance_score.mean().item():.6f}")
                print(f"    triangles.get_opacity      - min: {triangles.get_opacity.min().item():.6f}, max: {triangles.get_opacity.max().item():.6f}, mean: {triangles.get_opacity.mean().item():.6f}")

                # Calculate dead mask
                if number_of_views < 250:
                    dead_mask = torch.logical_or(
                        (triangles.importance_score < opt.importance_threshold).squeeze(),
                        (triangles.get_opacity <= opt.opacity_dead).squeeze()
                    )
                else:
                    if not new_round:
                        dead_mask = torch.logical_or(
                            (triangles.importance_score < opt.importance_threshold).squeeze(),
                            (triangles.get_opacity <= opt.opacity_dead).squeeze()
                        )
                    else:
                        dead_mask = (triangles.get_opacity <= opt.opacity_dead).squeeze()

                # Additional pruning condition
                if not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())

                # Debug: Print pruning statistics
                print(f"\n[DEBUG PRUNING ITER {iteration}]")
                print(f"  Triangle count before: {triangle_count_before}")
                print(f"  Dead triangles: {dead_mask.sum().item()}")
                print(f"  Dead ratio: {dead_mask.sum().item() / triangle_count_before * 100:.2f}%")
                print(f"  Criteria breakdown:")
                print(f"    - importance_score < {opt.importance_threshold}: {(triangles.importance_score < opt.importance_threshold).sum().item()}")
                print(f"    - opacity <= {opt.opacity_dead}: {(triangles.get_opacity <= opt.opacity_dead).sum().item()}")
                if not new_round:
                    print(f"    - triangle_area < 2: {(triangles.triangle_area < 2).sum().item()}")

                # Remove dead triangles
                triangles.remove_final_points(dead_mask)
                removed_them = True
                new_round = False

                triangle_count_after = triangles.get_triangles_points.shape[0]
                removed_count = triangle_count_before - triangle_count_after
                logger.log_pruning_stats(iteration, removed_count, triangle_count_before, triangle_count_after)

                print(f"  Triangle count after: {triangle_count_after}")
                print(f"  Removed: {removed_count}")

            # ========== OPTIMIZER STEP ==========
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
