# Experiment: vggt_omega_initialised

## Goal
To implement a modular add-on that utilizes the `vggt_omega` model to improve the MeshSplatting pipeline. 
> [!WARNING]
> **DEPRECATION NOTICE**: Experiment 1 (`geometry_only` / PCA alignment) was found to be **utter nonsense** because VGGT point clouds cannot be forcibly aligned into COLMAP coordinate frames via PCA. Attempting to mix VGGT points with COLMAP cameras creates extreme rendering corruption. **`geometry_only` is DEPRECATED and strictly prohibited.**
> **Only `full_pipeline` mode (`--vggt_mode full_pipeline`) must be used for all VGGT models.**

## Tasks
### Experiment 1: The Geometry Ablation [DEPRECATED] [❌]
- [❌] **geometry_only mode**: DEPRECATED. Mixing VGGT points with COLMAP cameras via PCA is invalid and utter nonsense.
- [❌] **Alignment**: PCA alignment removed. Never mix VGGT points with COLMAP cameras.

### Experiment 2: The COLMAP-Free Pipeline (VGGT Frame)
- [x] **Format Discovery:** Inspect the output format of VGGT Omega's cameras (e.g., standard COLMAP vs custom JSON).
- [x] **New Dataloader:** Write a custom reader `readVGGTSceneInfo` (or adapt existing) to parse VGGT extrinsics and intrinsics based on discovery.
- [x] **Depth Prior Realignment:** Create `make_depth_scale_vggt.py` using a Z-buffer splatting approach (rendering the dense cloud into frustums) to align Depth Anything V2 to the VGGT geometry, since standard 2D-3D matches won't exist.
- [ ] **Native Execution:** Train using VGGT points, VGGT cameras, and VGGT-aligned depth priors natively, bypassing COLMAP completely.
- [ ] **Evaluation:** Evaluate training speed and final mesh visual fidelity against the baseline (noting PSNR caveats due to different test cameras).

## Progress
- [x] Initialized experiment directory and tracking document.
- [x] Generated VGGT Omega points and cameras for the Tanks and Temples dataset (excluding Courthouse due to OOM).
- [x] Integrated `VGGTParams` and dataloader logic into `train.py`.
- [x] Computed depth scaling parameters for native VGGT execution.
- [x] Executed VGGT Full Pipeline training (`--vggt_mode full_pipeline`) for `Truck`, `Church`, `Caterpillar`, `Ignatius`, and `Meeting_room` (all encountered CUDA OOM due to massive ~50M-100M initial dense point cloud sizes).

## Results

### Experiment 2: VGGT Full Pipeline (`--vggt_mode full_pipeline`)

| Scene | Mode / Config | Final Vertices | Final Triangles | Peak GPU Memory | WandB Run | Status / Outcome | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Truck** | `full_pipeline` | - | - | **93.80 GiB** | [`amwls64p`](https://wandb.ai/HiLite-4D/HiLite-4D-MeshSplatting/runs/amwls64p) | ❌ **CUDA OOM** | Out of Memory during backward pass (`93.80 GiB` in use). |
| **Meeting_room** | `full_pipeline` | - | - | **65.95 GiB** | [`2e1jzk95`](https://wandb.ai/HiLite-4D/HiLite-4D-MeshSplatting/runs/2e1jzk95) | ❌ **CUDA OOM** | Out of Memory in `create_from_pcd` (78.41M points). |
| **Church** | `full_pipeline` | - | - | **89.78 GiB** | [`m9rkroxk`](https://wandb.ai/HiLite-4D/HiLite-4D-MeshSplatting/runs/m9rkroxk) | ❌ **CUDA OOM** | Out of Memory during backward pass (`89.78 GiB` in use). |
| **Caterpillar** | `full_pipeline` | - | - | **68.05 GiB** | [`ldg6wsj0`](https://wandb.ai/HiLite-4D/HiLite-4D-MeshSplatting/runs/ldg6wsj0) | ❌ **CUDA OOM** | Out of Memory in `create_from_pcd` (80.95M points). |
| **Ignatius** | `full_pipeline` | - | - | **93.90 GiB** | [`clc1ckxf`](https://wandb.ai/HiLite-4D/HiLite-4D-MeshSplatting/runs/clc1ckxf) | ❌ **CUDA OOM** | Out of Memory during backward pass (`93.90 GiB` in use). |
| **Barn** | `full_pipeline` | - | - | - | - | ⏳ Pending | VGGT_omega ply not proper |
| **Courthouse** | `full_pipeline` | - | - | - | - | ❌ **Skipped** | OOM during VGGT point cloud generation. |

### Configuration Modifications for Dense Point Cloud Initialization

Based on the analysis of the catastrophic pruning event with the dense VGGT initialization, the following hyperparameter adjustments were applied to stabilize training and prevent massive geometry loss:

| Parameter | Old Value (Baseline) | New Value | Rationale |
| :--- | :--- | :--- | :--- |
| `max_points` | `4,000,000` | Dynamic (`float('inf')` - Uncapped) | The ~10M point cloud produces ~30M initial vertices, which exceeded the 4M cap, completely blocking MCMC densification. The code in `train.py` was modified to set `effective_max_points = float('inf')` when `vggt_mode` is active, completely removing the max points ceiling for dense point clouds. |
| `start_pruning` | `4000` | `7000` | Delayed pruning gives the dense, overlapping triangles more time to adapt and develop sufficient opacity and importance scores before the thresholding begins. |
| `prune_triangles_threshold` | `0.235` | `0.1` | A lower threshold is more forgiving for dense initializations, where opacity is spread across many small overlapping triangles. |
| `set_weight` (Init Opacity) | `0.28` | `0.45` | Started triangles with higher opacity since the dense VGGT geometry is already a closer approximation of the surface than sparse SfM points. |
| `densify_from_iter` | `500` | `5000` | Shifted the densification window later into the training so the optimizer can first refine the positions of the dense initial points before spawning new ones. |
| `densify_until_iter` | `10000` | `10000` | Kept the end of the densification window the same. |
| `densification_interval` | `500` | `1000` | Reduced the frequency of densification to give the dense geometry more time to settle between splits/clones. |
| `start_opacity_floor` | `5000` | `3000` | Starts pushing opacities toward 1.0 earlier, helping more triangles survive the delayed pruning phase. |
