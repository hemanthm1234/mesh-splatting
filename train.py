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
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, VGGTParams, update_indoor
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import lpips
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Using fused SSIM for faster training.")
except:
    FUSED_SSIM_AVAILABLE = False
try:
    from diff_triangle_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Adam optimizer available")
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.render_utils import generate_path, create_videos



def training(
        dataset,   
        opt, 
        pipe,
        testing_iterations,
        checkpoint, 
        debug_from,
        scene_name,
        use_sparse_adam=False,
        wandb_name=None,
        vggt_args=None
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, wandb_name)
    
    if vggt_args is not None and WANDB_FOUND and wandb.run is not None:
        wandb.config.update({
            "vggt/vggt_mode": vggt_args.vggt_mode,
        })

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, vggt_args=vggt_args)

    if WANDB_FOUND:
        wandb.log({
            "iteration": first_iter
        }, step=first_iter)

    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_training_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # define the scheduler for sigma and opacity
    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter

    lambda_weight = opt.lambda_weight
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    
    need_delaunay = False

    run_restricted_delaunay = opt.densify_until_iter + 1000

    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)
    
    last_test_metrics = {}

    for iteration in range(first_iter, opt.iterations + 1):

        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False

        # Supersampling
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Sigma schedule
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = (iteration - sigma_start) / (total_iters - sigma_start)
            progress = min(progress, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda") # reset to 0 to ensure that everything is deleted with an importance score of 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, "normal_map", None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]
            seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]
            gt_normal = seg_ds_area.squeeze(0)  # -> [3, H0, W0]
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg["scaling"].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]

        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        pixel_count = render_pkg["triangle_was_rendered"].detach() # Not used but could be useful. Gives per triangle, the number of pixels it covered in the current render
        mask = pixel_count > triangles.pixel_count
        triangles.pixel_count[mask] = pixel_count[mask]

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
    

        # FINAL LOSS
        loss = loss_image

        # Opacity loss
        Lweight_pure = 0.0
        lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
        if lambda_weight > 0:
            mask_out = triangles.vertices.shape[0]
            vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            Lweight_pure = vertex_weights.mean()
            Lweight = lambda_weight * Lweight_pure
            loss += Lweight
        else:
            Lweight = 0

        # Vertex depth regularization
        Lvertex_depth_pure = 0.0
        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            depth_down = render_pkg["surf_depth"]
            vertex_depth_out = render_pkg["vertex_depth_out"]
            image_2D = render_pkg["image_2D"]
            vertex_rendered = render_pkg["vertex_rendered"]
            Lvertex_depth_pure = vertex_depth_loss_hr(
                vertex_depth_out,
                image_2D,
                vertex_rendered,
                depth_down,
                max_diff_threshold=opt.max_diff_threshold,
            )
            Lvertex_depth = lambda_vertex * Lvertex_depth_pure
            loss += Lvertex_depth
        else:
            Lvertex_depth = 0

        # Depth loss
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "invdepthmap", None) is not None:
            invDepth = 1.0 / (render_pkg["expected_depth"] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
        else:
            Ll1depth = 0

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        # Normal regularization (2DGS)
        Lnormal_pure = 0.0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            Lnormal_pure = normal_error.mean()
            Lnormal = lambda_normal * Lnormal_pure
            loss += Lnormal
        else:
            Lnormal = 0

        # supervised normal loss
        if gt_normal is not None:
            lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            normal_error = (1 - (rend_normal * gt_normal).sum(dim=0))[None]
            normal_loss_super = lambda_normals_super * (normal_error).mean()
            loss += normal_loss_super

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

            extra_metrics = {}
            if WANDB_FOUND:
                extra_metrics["system/gpu_vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                extra_metrics["train/loss_depth_l1"] = Ll1depth.item() if isinstance(Ll1depth, torch.Tensor) else Ll1depth
                extra_metrics["train/loss_vertex_depth_hr"] = Lvertex_depth.item() if isinstance(Lvertex_depth, torch.Tensor) else Lvertex_depth
                extra_metrics["train/loss_normal"] = Lnormal.item() if isinstance(Lnormal, torch.Tensor) else Lnormal

            test_metrics = training_report(tb_writer, scene_name, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), args.model_path, extra_metrics=extra_metrics)
            if test_metrics:
                last_test_metrics = test_metrics

            if iteration % 300 == 0:
                if WANDB_FOUND:
                    mask_out = triangles.vertices.shape[0]
                    vertex_weights = triangles.opacity_activation(triangles.get_vertex_weight[:mask_out])
                    opacity_mean = vertex_weights.mean().item()
                    opacity_max = vertex_weights.max().item()
                    opacity_min = vertex_weights.min().item()
                    opacity_saturation_pct = (vertex_weights > 0.99).float().mean().item() * 100.0
                    
                    tri_count = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0
                    vert_count = triangles.vertices.shape[0]
                    vertex_sharing_ratio = (tri_count * 3) / vert_count if vert_count > 0 else 0
                    
                    wandb.log({
                        "geometry/vertex_count": vert_count,
                        "geometry/triangle_count": tri_count,
                        "geometry/opacity_mean": opacity_mean,
                        "geometry/opacity_max": opacity_max,
                        "geometry/opacity_min": opacity_min,
                        "geometry/opacity_saturation_pct": opacity_saturation_pct,
                        "geometry/vertex_sharing_ratio": vertex_sharing_ratio,
                        "iteration": iteration
                    }, step=iteration)

            # Handle pruning operations
            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                
                # Building masks to delete triangles
                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                ) 
                min_weights = triangle_vertex_weights.min(dim=1).values

                mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                mask_importance  = (triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                mask_size        = (triangles.image_size > prune_size).squeeze()                 # delete if too big

                delete_mask = mask_opacity | mask_size

                if number_of_training_views < 500: # only delete if the number of views are below 500. Otherwise, we might delete too much
                    delete_mask = delete_mask | mask_importance
                
                if opt.context_adaptive_gradient_verification:
                    consistency = triangles.get_gradient_consistency()
                    if consistency is not None:
                        tri_consistency = consistency[triangles._triangle_indices].mean(dim=1)
                        if tri_consistency.numel() > 0:
                            # Only prune if consistency is below 0.1, AND don't prune more than 10% of the scene
                            thresh = torch.quantile(tri_consistency, 0.1).clamp(max=0.1)
                            mask_consistency_low = (tri_consistency < thresh).squeeze()
                            delete_mask = delete_mask | mask_consistency_low

                keep_mask   = ~delete_mask 

                if WANDB_FOUND:
                    tri_count_before_prune = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0

                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
                    
                if WANDB_FOUND:
                    tri_count_after_prune = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0
                    if tri_count_before_prune > 0:
                        prune_removed_pct = (tri_count_before_prune - tri_count_after_prune) / tri_count_before_prune * 100.0
                    else:
                        prune_removed_pct = 0.0
             
                # We prune vertices that are no longer used
                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], 
                                            dtype=torch.bool, 
                                            device=device)
                if triangles._triangle_indices.numel() > 0:
                    flat_indices = triangles._triangle_indices.flatten()
                    used_vertex_mask[flat_indices] = True
                
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                triangles._prune_vertices(vertex_mask)


                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                )  # [T,3]

                needs_densification = (iteration < opt.densify_until_iter and 
                                     iteration % opt.densification_interval == 0 and 
                                     iteration > opt.densify_from_iter)
                
                if needs_densification:
                    if WANDB_FOUND:
                        tri_count_before_densify = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0

                    effective_max_points = opt.max_points
                    if vggt_args is not None and vggt_args.vggt_mode:
                        effective_max_points = float('inf')  # Uncapped max points in VGGT mode

                    triangles.add_new_gs(iteration, cap_max=effective_max_points, splitt_large_triangles=splitt_large_triangles, context_adaptive_gradient_verification=opt.context_adaptive_gradient_verification)
                    
                    if WANDB_FOUND:
                        tri_count_after_densify = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0
                        if tri_count_before_densify > 0:
                            densify_added_pct = (tri_count_after_densify - tri_count_before_densify) / tri_count_before_densify * 100.0
                        else:
                            densify_added_pct = 0.0
                else:
                    if WANDB_FOUND:
                        densify_added_pct = 0.0
                        
                if WANDB_FOUND:
                    wandb.log({
                        "geometry/prune_removed_pct": prune_removed_pct,
                        "geometry/densify_added_pct": densify_added_pct,
                        "iteration": iteration
                    }, step=iteration)
   

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)

    # cleaning of triangles that we do not need
    viewpoint_stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
    while viewpoint_stack:
        viewpoint_cam = viewpoint_stack.pop(0)
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)

        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
    mask_importance  = (triangles.importance_score <= 0.5).squeeze() 
    triangles.prune_triangles(~mask_importance) # delete all the remaining triangles that do not have an influence

    device = triangles.vertices.device
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], 
                                dtype=torch.bool, 
                                device=device)
    if triangles._triangle_indices.numel() > 0:
        # Flatten indices and mark used vertices
        flat_indices = triangles._triangle_indices.flatten()
        used_vertex_mask[flat_indices] = True
    
    vertex_mask = used_vertex_mask
    triangles._prune_vertices(vertex_mask)

    if WANDB_FOUND:
        final_tri_count = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0
        final_vert_count = triangles.vertices.shape[0]
        final_vertex_sharing_ratio = (final_tri_count * 3) / final_vert_count if final_vert_count > 0 else 0
        
        mask_out = triangles.vertices.shape[0]
        if mask_out > 0:
            vertex_weights = triangles.opacity_activation(triangles.get_vertex_weight[:mask_out])
            opacity_mean = vertex_weights.mean().item()
            opacity_max = vertex_weights.max().item()
            opacity_min = vertex_weights.min().item()
            opacity_saturation_pct = (vertex_weights > 0.99).float().mean().item() * 100.0
        else:
            opacity_mean = opacity_max = opacity_min = opacity_saturation_pct = 0.0
        
        wandb.log({
            "geometry/final_vertex_count": final_vert_count,
            "geometry/final_triangle_count": final_tri_count,
        }, step=iteration)
        
        wandb.run.summary["final_triangle_count"] = final_tri_count
        wandb.run.summary["final_vertex_count"] = final_vert_count
        wandb.run.summary["geometry/vertex_count"] = final_vert_count
        wandb.run.summary["geometry/triangle_count"] = final_tri_count
        wandb.run.summary["final_vertex_sharing_ratio"] = final_vertex_sharing_ratio
        wandb.run.summary["final_opacity_mean"] = opacity_mean
        wandb.run.summary["final_opacity_max"] = opacity_max
        wandb.run.summary["final_opacity_min"] = opacity_min
        wandb.run.summary["final_opacity_saturation_pct"] = opacity_saturation_pct
        wandb.run.summary["max_vram_gb"] = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        if last_test_metrics:
            wandb.run.summary["final_psnr"] = last_test_metrics.get('psnr')
            wandb.run.summary["final_ssim"] = last_test_metrics.get('ssim')
            wandb.run.summary["final_lpips"] = last_test_metrics.get('lpips')
            wandb.run.summary["final_l1_loss"] = last_test_metrics.get('l1_loss')
            wandb.run.summary["final_fps"] = last_test_metrics.get('fps')

    scene.save(iteration)          
    print("Training is done")

def prepare_output_and_logger(args, wandb_name=None):    
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
        
    if WANDB_FOUND:
        wandb.init(entity="HiLite-4D", project="HiLite-4D-MeshSplatting", name=wandb_name, config=vars(args))
        # Save WandB run ID for later metrics logging
        with open(os.path.join(args.model_path, "wandb_id.txt"), "w") as f:
            f.write(wandb.run.id)
        # Explicitly tell wandb to use 'step' as the x-axis for all metrics
        wandb.define_metric("iteration")
        wandb.define_metric("train/*", step_metric="iteration")
        wandb.define_metric("test/*", step_metric="iteration")
        wandb.define_metric("geometry/*", step_metric="iteration")
    else:
        print("WandB not available: not logging to wandb")
        
    return tb_writer

def training_report(tb_writer, scene_name, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, model_path=".", extra_metrics=None):
    final_test_metrics = {}
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if WANDB_FOUND:
        log_dict = {
            'train/pixel_loss': pixel_loss.item(),
            'train/total_loss': loss.item(),
            'train/iter_time': elapsed,
            'iteration': iteration
        }
        if extra_metrics is not None:
            log_dict.update(extra_metrics)
        wandb.log(log_dict, step=iteration)

    # Report test and samples of training set
    if iteration % 300 == 0:
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
                
                wandb_test_images = []
                wandb_normal_maps = []
                
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    
                    render_pkg = renderFunc(viewpoint, scene.triangles, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # Save fixed progress video frames
                    if config['name'] == 'test' and idx < 5:
                        progress_dir = os.path.join(model_path, "progress", f"view_{idx}")
                        os.makedirs(progress_dir, exist_ok=True)
                        torchvision.utils.save_image(image, os.path.join(progress_dir, f"iter_{iteration:05d}.png"))
                        
                        if WANDB_FOUND:
                            rgb_vis = image.detach().cpu().permute(1, 2, 0).numpy()
                            error_map = torch.abs(image - gt_image).mean(dim=0).detach().cpu().numpy()
                            depth_vis = render_pkg["surf_depth"].detach().squeeze(0).cpu().numpy()
                            depth_vis = depth_vis / (depth_vis.max() + 1e-6)
                            
                            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                            axs[0].imshow(rgb_vis)
                            axs[0].set_title("RGB")
                            axs[0].axis("off")
                            axs[1].imshow(depth_vis, cmap='plasma')
                            axs[1].set_title("Depth")
                            axs[1].axis("off")
                            axs[2].imshow(error_map, cmap='magma')
                            axs[2].set_title("Error")
                            axs[2].axis("off")
                            fig.tight_layout()
                            
                            wandb_test_images.append(wandb.Image(fig, caption=f"View {idx}"))
                            plt.close(fig)
                            
                            if "rend_normal" in render_pkg:
                                normal_map = render_pkg["rend_normal"].detach().cpu().permute(1, 2, 0).numpy()
                                normal_vis = np.clip((normal_map + 1.0) / 2.0, 0.0, 1.0)
                                wandb_normal_maps.append(wandb.Image(normal_vis, caption=f"View {idx} Normal"))
                        
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
                fps = 1000.0 / total_time
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} FPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test, fps))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test.item() if hasattr(pixel_loss_test, 'item') else pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test.item() if hasattr(psnr_test, 'item') else psnr_test, iteration)
                    
                if WANDB_FOUND:
                    log_dict = {
                        f"{config['name']}/l1_loss": pixel_loss_test.item() if hasattr(pixel_loss_test, 'item') else pixel_loss_test,
                        f"{config['name']}/psnr": psnr_test.item() if hasattr(psnr_test, 'item') else psnr_test,
                        f"{config['name']}/ssim": ssim_test.item() if hasattr(ssim_test, 'item') else ssim_test,
                        f"{config['name']}/lpips": lpips_test.item() if hasattr(lpips_test, 'item') else lpips_test,
                        f"{config['name']}/fps": fps,
                        "iteration": iteration
                    }
                    if config['name'] == 'test' and len(wandb_test_images) > 0:
                        log_dict["media/test_images"] = wandb_test_images
                    if config['name'] == 'test' and len(wandb_normal_maps) > 0:
                        log_dict["media/normal_maps"] = wandb_normal_maps
                        
                    if config['name'] == 'test':
                        final_test_metrics['psnr'] = psnr_test.item() if hasattr(psnr_test, 'item') else psnr_test
                        final_test_metrics['ssim'] = ssim_test.item() if hasattr(ssim_test, 'item') else ssim_test
                        final_test_metrics['lpips'] = lpips_test.item() if hasattr(lpips_test, 'item') else lpips_test
                        final_test_metrics['l1_loss'] = pixel_loss_test.item() if hasattr(pixel_loss_test, 'item') else pixel_loss_test
                        final_test_metrics['fps'] = fps
                        
                    wandb.log(log_dict, step=iteration)

        torch.cuda.empty_cache()
    
    return final_test_metrics

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    vp = VGGTParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--wandb_name', default="Test", type=str)
    parser.add_argument('--scene_name', default="Garden", type=str)
    parser.add_argument("--use_sparse_adam", action="store_true", default=True)
    parser.add_argument("--indoor", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)
    vps = vp.extract(args)

    if args.indoor:
        ops = update_indoor(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps,
             ops,
             pps,
             args.test_iterations,
             args.start_checkpoint,
             args.debug_from,
             args.scene_name,
             use_sparse_adam=args.use_sparse_adam,
             wandb_name=args.wandb_name,
             vggt_args=vps
             )
    
    # All done
    print("\nTraining complete.")