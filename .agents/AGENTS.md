# MeshSplatting AGENT

## Problem Statement

> ### `HiLite-4D`: High-Fidelity, Lightweight Surface Reconstruction for Dynamic Scenes

<div align="center">

![High Fidelity](https://img.shields.io/badge/High%20Fidelity-blue)
![4D SURFACE RECONSTRUCTION](https://img.shields.io/badge/4D%20SURFACE%20RECONSTRUCTION-orange)
![Efficient](https://img.shields.io/badge/Efficient-brightgreen)

</div>

#### 1. Introduction & Motivation
Modelling 4D dynamic scenes is of utmost importance in fields like AR/VR, digital twins, telepresence, and robotics. For these applications to be practical, the reconstructed assets must be both `computationally efficient` (capable of real-time rendering and fast optimization) and `high-fidelity` (yielding high fidelity surfaces geometry).

#### 2. The Core Problem (The Gap)
While 3D Gaussian Splatting (3DGS) achieves real-time rendering and high visual fidelity, its primitive-based representation (millions of anisotropic Gaussians requiring sorting and alpha blending) is fundamentally incompatible with standard mesh-based pipelines that power AR/VR and game engines (which rely on depth buffers and occlusion culling). Existing conversion methods are a-posteriori post-processing steps that degrade visual quality, while native mesh-based approaches often lack realism or take too long to train.

#### 3. Goal
Bridge the gap between Speed and Quality of surface reconstruction to enable practical usage of 4D assets on low compute devices by using **MeshSplatting**.

#### 4. Scope
Exploring various techniques for High Fidelity reconstruction like different regularisation losses, different surface representations, etc. Exploring efficiency techniques like pruning number of gaussians, different representations of motion, etc. Understanding current baselines and existing trade-off better.

#### 5. Current Sub-Goal (Main Focus)

<div style="background-color: rgba(65, 105, 225, 0.15); border-left: 5px solid #4169E1; padding: 15px; border-radius: 4px;">
  <h4 style="margin-top: 0; color: #4169E1; font-weight: bold;">🎯 Pruning and Efficiency for 3D Scenes</h4>
  <p style="margin-bottom: 0;">Our immediate, primary focus is on pruning and efficiency-related optimizations for Mesh Splatting. The objective is to drastically reduce the number of triangles and minimize the compute and time required to represent 3D scenes, while preserving high-fidelity surface reconstruction. Once these efficiency gains are successfully achieved and validated for static 3D scenes, we will extend the same principles to dynamic 4D scenes.</p>
</div>

---

## MeshSplatting Differentiable Rendering with Opaque Meshes - Theory & Methodology (The "What" and "Why")

This agent represents the **MeshSplatting** architecture, which directly optimizes a connected, opaque triangle mesh end-to-end without requiring post-hoc marching cubes for game engine compatibility. The training is a two-stage process:

1. **Stage 1: Triangle Soup Optimization.** The scene is initialized from SfM points as an unstructured "soup" of independent, semi-transparent triangles. The system optimizes the vertex positions, Spherical Harmonics (color), size, and opacity. This unconstrained stage allows rapid adaptation to geometry. It includes MCMC-like densification (midpoint subdivision) and opacity-based pruning.
2. **Stage 2: Mesh Creation & Refinement.** To convert the soup into a connected surface, the system executes **Restricted Delaunay Triangulation** (using `effrdel`) once (e.g., at iteration 5k+1000). This operation restores global connectivity, creating shared vertices among adjacent triangles.
3. **Fine-Tuning.** Optimization then continues on the *connected mesh*. Gradients are accumulated across shared vertices. Crucially, the opacity is aggressively driven to $1.0$ (fully opaque) using a scheduled sigmoid function, meaning the final representation is a purely opaque, standard colored triangle mesh natively compatible with engines like Unity or Unreal.

```mermaid
graph LR
  A["SFM Point Cloud"] -- (Triangle Splatting) --> B["Independent Triangles"]
  B -- (Restricted Delaunay Triangularisation) --> C["Connected Mesh"]
  C --(finetune)--> D["Opaque Connected Mesh"]
```

---

## Codebase Reality & Overview (The "How")

### Input Data Formats & Requirements

The training pipeline (`train.py`) accepts two primary dataset formats via the `--source_path` argument:

1. **COLMAP Datasets** (Real-world scenes):
   - **Structure**: 
     - `sparse/0/`: Must contain COLMAP camera and image parameters (`cameras.bin`/`.txt`, `images.bin`/`.txt`).
     - `images/`: The RGB input images.
   - **Initial Geometry**: Loads `sparse/0/points3D.ply` (or converts `.bin`/`.txt` to `.ply`). If the `aug` flag is used, it looks for `points3D_138views.ply`.
   - **Geometry Priors (Highly Recommended/Required for optimal results)**:
     - **Monocular Depth**: Looks for a `depth/` folder containing depth maps (e.g., from Depth Anything V2) and a scale/offset configuration file at `sparse/0/depth_params.json`. These are used for depth alignment (`Ll1depth` and `Lvertex_depth` losses).
     - **Normal Maps**: Looks for a `normals/` directory (alongside `images/`) containing PNG normal maps. Used for supervised normal loss.

2. **Blender Synthetic Datasets** (NeRF synthetic scenes):
   - **Structure**: Requires `transforms_train.json` and `transforms_test.json`.
   - **Images**: Referenced directly within the JSON files.
   - **Initial Geometry**: Looks for `points3d.ply`. If not found, it generates a random point cloud of 100k points within the scene's bounding box.

### Output Pathways

While the training loop (`train.py`) natively outputs a connected, game-engine-ready mesh (`point_cloud.ply`) thanks to Delaunay triangulation, some evaluation metrics strictly require a single-layer, perfectly watertight manifold surface. Thus, the repository contains two separate pathways for exporting geometry:

1. **`create_ply.py`**: The true MeshSplatting export. It takes the trained model (the Delaunay-connected splats) and exports a standard RGB `.ply` mesh directly for game engines. **This is the path used by the T&T baseline**.
2. **`mesh.py`**: An *academic evaluation fallback*. It renders unbounded depth maps from the trained model and fuses them using Open3D **TSDF Integration** and Marching Cubes (`fuse_unbounded.ply`). This is used for Chamfer Distance on the DTU dataset, but **it is not used in the T&T baseline**.

### Key Components:
- **`scene/triangle_model.py`**: Core logic. Manages initialization, properties (opacity, SH, scaling), and dynamic topology. It is here that `run_restricted_delaunay()` is called to merge the triangle soup.
- **`triangle_renderer/`**: The customized CUDA renderer (`diff-triangle-mesh-rasterization`). Evaluates differentiable rasterization for triangles.
- **`train.py` & `arguments/`**: The main training loop. Handles loss scheduling (L1, SSIM, `vertex_depth_loss_hr`, normal loss, depth alignment).
- **`submodules/effrdel`**: The C++ extension that computes the Restricted Delaunay Triangulation during Stage 2.
- **`segmentation/` & `annotate_points_boxes.py`**: Integrates SAM 2 for 2D object masking, allowing specific triangles to be cleanly selected and extracted directly in 3D since there is a 1:1 pixel-to-triangle mapping.

## Full Repository Structure & Context

Below is the definitive structure of the `mesh-splatting` repository, outlining every critical file and folder along with its contribution to the pipeline:

### 1. Root Level Scripts (Pipeline Orchestration)
- **`train.py`**: The main entry point for optimization. It orchestrates the 2-stage process: Triangle Soup initialization, Restricted Delaunay Triangulation (via `effrdel`), and fine-tuning.
- **`render.py`**: Script for novel view synthesis rendering using the optimized mesh splats.
- **`create_video.py`**: Generates a continuous camera trajectory video from the rendered frames.
- **`create_ply.py`**: The definitive script to export the optimized, Delaunay-connected splats into a standard RGB `.ply` mesh directly usable by game engines (Unity/Unreal).
- **`mesh.py`**: A fallback script utilizing TSDF integration and Marching Cubes (`open3d`) to extract a watertight surface (`fuse_unbounded.ply`). Used to satisfy academic metrics (like DTU Chamfer Distance).
- **`eval.py` & `full_eval.py` & `metrics.py`**: A suite of scripts for measuring geometric and image-space metrics. `eval.py` computes Chamfer distance for DTU, while `metrics.py` evaluates PSNR, SSIM, LPIPS for T&T.
- **`annotate_points_boxes.py`**: The entry point for generating 2D segmentation masks of specific objects using SAM2.

### 2. Core Modules
- **`scene/` (Geometry & Data)**
  - `triangle_model.py`: The heart of the representation. Manages the initialization of primitives, their attributes (opacity, spherical harmonics, scaling), and dynamic topology updates (MCMC densification, pruning, and the critical `effrdel` Delaunay invocation).
  - `dataset_readers.py` & `cameras.py`: Handles loading the COLMAP sparse point clouds, Blender synthetic datasets, and the monocular depth priors from Depth Anything V2.
- **`triangle_renderer/` (Rasterization Bridge)**
  - `__init__.py`: The PyTorch wrapper that interfaces with the custom CUDA rasterizer. It projects the 3D triangles to 2D screen space and evaluates the differentiable rasterization.
- **`utils/` (Helper Logic)**
  - `loss_utils.py`: Defines the photometric losses and the essential `vertex_depth_loss_hr` used for aligning splats with the monocular depth priors.
  - `mesh_utils.py`: Contains the `GaussianExtractor` class used by `mesh.py` for TSDF volumetric fusion and bounded/unbounded meshing.
  - `make_depth_scale.py`: Generates `depth_params.json` to correctly scale the imported depth priors.
- **`submodules/` (C++/CUDA Extensions)**
  - `diff-triangle-mesh-rasterization`: The core custom CUDA rasterizer, heavily modified to support the rendering of opaque triangle splats and depth calculation.
  - `effrdel`: A custom C++ extension wrapper around SciPy's Delaunay that executes the Restricted Delaunay Triangulation (RDT) during Stage 2 of training.
  - `simple-knn`: Used for fast K-Nearest Neighbors search during the initial splat scaling.
- **`segmentation/` (Object Extraction)**
  - `extract_images.py`, `sam_mask_generator_json.py`, `segment.py`, `run_single_object.py`: A highly specialized toolkit that maps 2D SAM2 masks to the 3D triangle mesh. Because each pixel maps to exactly one opaque triangle, these scripts can cleanly isolate, extract, or remove specific object sub-meshes directly.

---

## Workflows and Guidelines
- **Experiments**: All experiments should be stored in the `./my_expts` folder. Furthermore, each individual experiment folder inside `./my_expts` must contain an `experiment_details.md` file documenting the experiment.
- **Modular, Flag-Based Development**: 
  - **Modifying Core Files**: Any new experimental feature or modification to existing pipeline logic (e.g., `train.py`, `scene/triangle_model.py`) MUST be implemented behind a specific configuration flag or command-line argument.
    - **Do NOT** blindly overwrite or comment out existing logic.
    - Create a new, separate function or conditional branch for the experimental feature.
    - The new feature should only run when its corresponding flag is provided. If the flag is absent, the original, untouched logic must execute.
    - Ensure changes to the core codebase are minimalistic and modular so multiple experimental flags can be toggled independently without conflict.
  - **Experiment-Specific Files**: If you are generating entirely new files exclusively for an experiment (e.g., scripts inside `my_expts/<experiment_folder>/`), you have full freedom. Flagging is not required for these isolated files.
- **VGGT Dataset Handling Rule**: ALL VGGT `ply` files (e.g., VGGT 10M, 5M, or VGGT Colmap-downsampled) MUST be used with VGGT native camera poses via `--vggt_mode full_pipeline`. The legacy `--vggt_mode geometry_only` (PCA coordinate alignment between VGGT and COLMAP) is **utter nonsense and strictly deprecated**. There is absolutely NO conversion, NO PCA alignment, and NO ICP alignment required or permitted between VGGT and COLMAP coordinate systems. Always use `--vggt_mode full_pipeline` when visualizing, training, or working with ANY VGGT-based model to ensure native VGGT cameras are loaded.
- For completed tasks, put [✅] and for those tasks that needn't be done, put [❌] and for the tasks that need to be done, put [ ].
- **Dashboard Maintenance**: Whenever you complete a new major modification, experiment, or integration, you MUST proactively update both the high-level dashboard table in `AGENTS.md` and the detailed technical documentation in `my_expts/what_all_I_have_done.tex`.

---

## What all I have done?

> [!NOTE]
> For highly detailed technical documentation, architectural rationales, and algorithmic flowcharts for all completed tasks, please refer to the fully compiled PDF document generated from: [what_all_I_have_done.tex](file:///data1/hemanth/mesh-splatting/my_expts/what_all_I_have_done.tex).

| Type | Name / Description | Status / Outcome | Impact / Notes |
|---|---|---|---|
| Environment & Setup | **Dockerization & Portability** | [✅] Completed | Packaged the perfectly working local environment into a portable Docker blueprint to provide the flexibility to seamlessly deploy and run the codebase on any remote machine. |
| Experiment / Baseline | **Baseline Setup (T&T Truck, Barn, Caterpillar)** | [✅] Baseline Established | Ran baseline, documented Delaunay vs TSDF fallback, and created `create_annotated_video.py`. Confirmed Chamfer Distance is incompatible with T&T exported meshes. Verified final post-pruning vertex count (e.g. Barn ~2.7M) is the definitive benchmark metric. |
| Tooling / Integration | **WandB Logging Integration** | [✅] Completed | Configured `train.py` for full tracking, added `geometry/final_vertex_count` to main UI table, and created `.agents/skills/wandb/SKILL.md`. |
| Experiment | **Context-Adaptive Gradient Verification** | [ ] Pending execution | Implemented EMA gradient consistency pruning/splitting and added `--context_adaptive_gradient_verification`. |
| Experiment | **VGGT Omega Initialization** | [✅] Completed | Pipeline integrated for using `vggt_omega` dense point cloud and native cameras. |
