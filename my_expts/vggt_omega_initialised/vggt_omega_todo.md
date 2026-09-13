# VGGT Omega Initialization — Implementation Specification

> [!CAUTION]
> **DEPRECATION NOTICE: `geometry_only` / PCA Alignment is UTTER NONSENSE and DEPRECATED.**  
> Attempting to mix VGGT geometry with COLMAP cameras via PCA alignment is completely invalid and produces corrupted renders.
> **All VGGT datasets MUST be trained and rendered using native VGGT cameras via `--vggt_mode full_pipeline`.**

---

## PART 1 — EXPERIMENT 1: The Geometry Ablation [DEPRECATED]

## Design Decisions (Already Resolved)

| Decision | Answer |
|---|---|
| Where is the VGGT PLY stored? | `<source_path>/vggt_omega/points3D.ply` |
| Alignment strategy | Centroid + scale alignment (see §3) |
| Dense cloud handling | Pass directly to `create_from_pcd` — no subsampling |
| Argument group location | New `VGGTParams` class in `arguments/__init__.py` |
| Pipeline interception point | In `scene/__init__.py` — after `scene_info` is built, before `create_from_pcd` |
| Missing PLY behavior | Raise a hard `FileNotFoundError` and abort |
| Comparison scene | T&T Truck |
| Metrics | All standard WandB metrics (no changes to logging) |

---

## PHASE 0 — Prerequisites: Generate VGGT Omega PLY Files

> [!IMPORTANT]
> These steps must be completed BEFORE any code changes are run. The PLY files must exist at the expected paths.

### TODO 0.1 — Understand VGGT Omega Output Format
- [ ] Identify what the `vggt_omega` model outputs:
  - Does it produce a world-space point cloud, or a camera-space one?  
  - What coordinate convention does it use (Y-up vs Z-up, OpenCV vs OpenGL)?
  - Does it produce vertex colors (RGB)? Are they in [0,1] or [0,255]?
  - Does it produce per-point normals? (The `fetchPly()` function expects `nx, ny, nz` fields.)
- [ ] Document these findings in `experiment_details.md`.

### TODO 0.2 — Run VGGT Omega Inference on T&T Truck
- [ ] Run `vggt_omega` on the T&T Truck scene using its images/cameras.
- [ ] Save the resulting dense point cloud as:
  ```
  /data1/hemanth/datasets/tandt/truck/vggt_omega/points3D.ply
  ```
- [ ] Verify the PLY file is valid:
  - It must have `x, y, z` (float), `red, green, blue` (uint8), and `nx, ny, nz` (float) vertex properties.
  - If normals are missing, add zero-normals using a simple script (see §1.4).
  - Open in MeshLab and visually confirm the point cloud looks like the Truck scene.

### TODO 0.3 — Create PLY Validation/Conversion Utility Script
- [ ] Create `my_expts/vggt_omega_initialised/prepare_vggt_ply.py`:
  - **Input**: path to the raw VGGT Omega output file (any format).
  - **Checks**: verifies all required PLY fields exist.
  - **Conversion**: if normals are missing, pads with zeros. If colors are float [0,1], converts to uint8.
  - **Output**: writes a standards-compliant `points3D.ply` to `<source_path>/vggt_omega/`.
  - This script is idempotent — if the output already exists, it is a no-op (prints a message and exits).

---

## PHASE 1 — Argument System Changes

### TODO 1.1 — Add `VGGTParams` class to `arguments/__init__.py`

In [arguments/__init__.py](file:///data1/hemanth/mesh-splatting/arguments/__init__.py), after the `OptimizationParams` class definition (around line 149), add:

```python
class VGGTParams(ParamGroup):
    def __init__(self, parser):
        # Controls which experiment is running: 'geometry_only' (Exp 1), 'full_pipeline' (Exp 2), or '' (Disabled)
        self.vggt_mode = ""
        super().__init__(parser, "VGGT Omega Init Parameters")
```

**Why a separate class?**  
- Matches how `OptimizationParams`, `ModelParams`, `PipelineParams` are already structured.
- Makes it easy to `extract(args)` as a standalone group in `train.py`.
- Future VGGT-related flags (e.g., `--vggt_max_points`, `--vggt_alignment_mode`) can be added here without touching other param groups.

> [!NOTE]
> The PLY path is **not** a flag. It is automatically derived as:  
> `<source_path>/vggt_omega/points3D.ply` where `source_path` comes from `ModelParams`.  
> This avoids adding a redundant path argument and keeps convention consistent.

---

## PHASE 2 — Data Pipeline: PLY Loader

### TODO 2.1 — Add `load_vggt_omega_pcd()` to `scene/dataset_readers.py`

Add the following function at the bottom of [scene/dataset_readers.py](file:///data1/hemanth/mesh-splatting/scene/dataset_readers.py) (before `sceneLoadTypeCallbacks`):

```python
def load_vggt_omega_pcd(source_path: str) -> BasicPointCloud:
    """
    Loads the pre-generated VGGT Omega point cloud from the convention path:
        <source_path>/vggt_omega/points3D.ply
    Raises FileNotFoundError if the file does not exist.
    Returns a BasicPointCloud aligned to COLMAP world space.
    """
    vggt_ply_path = os.path.join(source_path, "vggt_omega", "points3D.ply")
    if not os.path.exists(vggt_ply_path):
        raise FileNotFoundError(
            f"[VGGT Omega Init] PLY file not found at: '{vggt_ply_path}'.\n"
            f"Please generate the VGGT Omega point cloud and place it at the expected path.\n"
            f"If you did not intend to use VGGT Omega init, remove the --vggt_mode == "geometry_only" flag."
        )
    print(f"[VGGT Omega Init] Loading VGGT Omega point cloud from: {vggt_ply_path}")
    pcd = fetchPly(vggt_ply_path)
    print(f"[VGGT Omega Init] Loaded {pcd.points.shape[0]:,} points from VGGT Omega PLY.")
    return pcd
```

**Why here and not in `scene/__init__.py`?**  
- All PLY loading logic lives in `dataset_readers.py` (`fetchPly`, `storePly`, `readColmapSceneInfo`). This function follows that convention.
- `scene/__init__.py` stays as the orchestrator that decides *which* PCD to use, while `dataset_readers.py` handles *how* to load it.

---

## PHASE 3 — Alignment Strategy

> [!IMPORTANT]
> This is the most critical and conceptually tricky part of the whole experiment. Read carefully.

### TODO 3.1 — Understand Why Alignment is Needed

VGGT Omega outputs point clouds in its own **model/camera coordinate frame** which may differ from the COLMAP world frame in:
- **Scale**: COLMAP uses an arbitrary scale (meters or relative units). VGGT Omega may use a different scale.
- **Translation**: Origins may not coincide.
- **Rotation**: VGGT may use a different up-vector (e.g., Y-up vs COLMAP's Z-forward convention).

Without alignment, initializing triangles from an unaligned VGGT cloud would produce triangles that are wildly offset from the camera frustums, causing zero photometric gradients and broken training.

#### TODO 3.2 — Implement PCA + Centroid + Scale Alignment
 
 **Decision: Use PCA (Principal Component Analysis) + centroid-matching + scale normalization.** 
 
 **Algorithm:**
 1. Load both the VGGT PLY (`vggt_pts`) and the COLMAP PLY (`colmap_pts`) as numpy arrays.
 2. **Translation**: Compute `vggt_centroid = vggt_pts.mean(axis=0)`, `colmap_centroid = colmap_pts.mean(axis=0)`. Apply `translated_pts = vggt_pts - vggt_centroid`.
 3. **Rotation (PCA)**: Compute the covariance matrix of `translated_pts` and `colmap_pts - colmap_centroid`. Find the eigenvectors (principal axes) for both. Compute the rotation matrix `R = V_colmap @ V_vggt.T`. Apply `rotated_pts = translated_pts @ R.T`. *(Note: Eigenvectors have a sign ambiguity (180-degree flips). If the object appears upside-down or mirrored, this ambiguity will need manual correction via a simple flag flip).*
 4. **Scale**: Compute `vggt_scale = np.linalg.norm(rotated_pts, axis=1).mean()`, `colmap_scale = np.linalg.norm(colmap_pts - colmap_centroid, axis=1).mean()`. Apply `scale_factor = colmap_scale / vggt_scale`. Apply `aligned_pts = rotated_pts * scale_factor + colmap_centroid`.
 5. Return a new `BasicPointCloud` with the aligned coordinates and the original colors.
 
 **Why PCA?**  
 - The user requested a "good method" like PCA for now. It attempts to align the dominant axes of the geometries without needing explicit point-to-point correspondences.

### TODO 3.3 — Add `align_vggt_to_colmap()` to `scene/dataset_readers.py`

```python
def align_vggt_to_colmap(
    vggt_pcd: BasicPointCloud,
    colmap_pcd: BasicPointCloud,
) -> BasicPointCloud:
    """
    Aligns the VGGT Omega point cloud to the COLMAP world frame using:
      - Centroid matching (translation)
      - RMS scale matching (scale)
    No rotation is applied (centroid + scale is sufficient for initialization).
    """
    vggt_pts = np.asarray(vggt_pcd.points)
    colmap_pts = np.asarray(colmap_pcd.points)

    vggt_centroid  = vggt_pts.mean(axis=0)
    colmap_centroid = colmap_pts.mean(axis=0)

    # Normalize both to zero-centroid, compute scale ratio
    vggt_centered  = vggt_pts - vggt_centroid
    colmap_centered = colmap_pts - colmap_centroid

    vggt_rms  = np.sqrt((vggt_centered ** 2).sum(axis=1).mean())
    colmap_rms = np.sqrt((colmap_centered ** 2).sum(axis=1).mean())

    if vggt_rms < 1e-8:
        print("[VGGT Omega Init] WARNING: VGGT point cloud has near-zero spread. Skipping scale alignment.")
        scale_factor = 1.0
    else:
        scale_factor = colmap_rms / vggt_rms

    aligned_pts = vggt_centered * scale_factor + colmap_centroid

    print(f"[VGGT Omega Init] Alignment: centroid shift={colmap_centroid - vggt_centroid}, "
          f"scale_factor={scale_factor:.4f}")

    return BasicPointCloud(
        points=aligned_pts.astype(np.float32),
        colors=np.asarray(vggt_pcd.colors).astype(np.float32),
        normals=np.asarray(vggt_pcd.normals).astype(np.float32),
    )
```

---

## PHASE 4 — Scene Initialization Interception (`scene/__init__.py`)

### TODO 4.1 — Modify `Scene.__init__()` to optionally swap the PCD

In [scene/__init__.py](file:///data1/hemanth/mesh-splatting/scene/__init__.py):

**Step 1:** Update the function signature to accept VGGT args:
```python
def __init__(self, args: ModelParams, triangles: TriangleModel, init_opacity, set_sigma,
             load_iteration=None, shuffle=True, resolution_scales=[1.0],
             segment=False, ratio_threshold=0.75,
             vggt_args=None):   # <-- ADD THIS
```

**Step 2:** After the line `if not self.loaded_iter:` block and the `scene_info` is built, add a new conditional block to replace `scene_info.point_cloud`:

```python
# ── VGGT Omega Init (optional) ──────────────────────────────────────
if not self.loaded_iter and vggt_args is not None and vggt_args.vggt_mode == "geometry_only":
    from scene.dataset_readers import load_vggt_omega_pcd, align_vggt_to_colmap
    vggt_pcd = load_vggt_omega_pcd(args.source_path)
    colmap_pcd = scene_info.point_cloud  # the original COLMAP PCD, kept for alignment reference
    aligned_vggt_pcd = align_vggt_to_colmap(vggt_pcd, colmap_pcd)
    # Replace point_cloud in scene_info with the aligned VGGT cloud
    scene_info = scene_info._replace(point_cloud=aligned_vggt_pcd)
    print(f"[VGGT Omega Init] Replaced COLMAP PCD ({colmap_pcd.points.shape[0]:,} pts) "
          f"with VGGT Omega PCD ({aligned_vggt_pcd.points.shape[0]:,} pts).")
# ────────────────────────────────────────────────────────────────────
```

**Why here and not in `dataset_readers.py`?**  
- `scene/__init__.py` is the orchestrator. It already decides whether to call `create_from_pcd` or `load_parameters`. It's the natural place to decide *which* PCD to pass to `create_from_pcd`.
- `dataset_readers.py` stays responsible for *reading files*, not for *orchestrating the initialization strategy*.
- **Once aligned**, the VGGT PCD is passed directly into the existing `create_from_pcd()` call — **zero changes needed to `triangle_model.py`**. This is the key to keeping the core pipeline invariant.

> [!NOTE]
> `scene_info` is a `NamedTuple`. Use `._replace(point_cloud=...)` to create a modified copy without mutating the original.

---

## PHASE 5 — `train.py` Integration

### TODO 5.1 — Import and parse `VGGTParams` in `train.py`

In [train.py](file:///data1/hemanth/mesh-splatting/train.py):

**Step 1:** Import the new param class:
```python
from arguments import ModelParams, PipelineParams, OptimizationParams, VGGTParams, update_indoor
```

**Step 2:** Add parser registration in `__main__` (around line 690):
```python
vp = VGGTParams(parser)
```

**Step 3:** Extract the args after `lp.extract(args)`:
```python
vps = vp.extract(args)
```

**Step 4:** Update the `training()` function signature to accept `vggt_args`:
```python
def training(dataset, opt, pipe, testing_iterations, checkpoint, debug_from,
             scene_name, use_sparse_adam=False, wandb_name=None,
             vggt_args=None):    # <-- ADD THIS
```

**Step 5:** Pass `vggt_args` when constructing `Scene`:
```python
scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma, vggt_args=vggt_args)
```

**Step 6:** Pass `vggt_args` in the `training()` call at the bottom of `__main__`:
```python
training(lps, ops, pps,
         args.test_iterations,
         args.start_checkpoint,
         args.debug_from,
         args.scene_name,
         use_sparse_adam=args.use_sparse_adam,
         wandb_name=args.wandb_name,
         vggt_args=vps)          # <-- ADD THIS
```

### TODO 5.2 — Log VGGT Omega init status to WandB

In `train.py`, inside the `if WANDB_FOUND:` block near the top of `training()`:
```python
if vggt_args is not None and WANDB_FOUND:
    wandb.config.update({
        "vggt/vggt_mode == "geometry_only"": vggt_args.vggt_mode == "geometry_only",
    })
```

This ensures every WandB run records whether VGGT Omega init was used, making the comparison dashboard clear.

---

## PHASE 6 — Caching: Avoid Re-Running Alignment Each Time

> [!NOTE]
> The alignment itself is fast (pure numpy, runs in milliseconds). The PLY file is already pre-generated by the user. There is nothing to cache — `load_vggt_omega_pcd()` just calls `fetchPly()` which reads the pre-existing file. The "idempotency" requirement from the design discussion is satisfied by requiring the file to exist before training.

No additional caching logic is needed.

---

## PHASE 7 — Experiment Run: Baseline vs VGGT Omega

### TODO 7.1 — Run Baseline (COLMAP) on T&T Truck

```bash
python train.py \
  --source_path /data1/hemanth/datasets/tandt/truck \
  --model_path my_expts/vggt_omega_initialised/output/truck_colmap_baseline \
  --wandb_name vggt_expt_truck_colmap \
  --scene_name truck \
  --eval
```

- Record the WandB run ID.
- Let it train to full 30k iterations.

### TODO 7.2 — Run VGGT Omega Init on T&T Truck

```bash
python train.py \
  --source_path /data1/hemanth/datasets/tandt/truck \
  --model_path my_expts/vggt_omega_initialised/output/truck_vggt_omega \
  --wandb_name vggt_expt_truck_vggt_omega \
  --scene_name truck \
  --eval \
  --vggt_mode == "geometry_only"
```

- Record the WandB run ID.
- Let it train to full 30k iterations.

### TODO 7.3 — Export Meshes for Visual Comparison

For each run, export the trained mesh:
```bash
python create_ply.py \
  --model_path my_expts/vggt_omega_initialised/output/truck_colmap_baseline

python create_ply.py \
  --model_path my_expts/vggt_omega_initialised/output/truck_vggt_omega
```

Open both `.ply` files in MeshLab and take screenshots for qualitative comparison.

---

## PHASE 8 — Documentation

### TODO 8.1 — Update `experiment_details.md`

Mark completed tasks in [experiment_details.md](file:///data1/hemanth/mesh-splatting/my_expts/vggt_omega_initialised/experiment_details.md) using `[x]` as steps are completed.  
Add a **Results** section with:
- Links to WandB runs.
- PSNR/SSIM/LPIPS/vertex count comparison table.
- Qualitative mesh screenshots.
- Observation: did VGGT Omega init converge faster? Did it produce a better mesh?

### TODO 8.2 — Update `AGENTS.md` dashboard

In [.agents/AGENTS.md](file:///data1/hemanth/mesh-splatting/.agents/AGENTS.md), change the VGGT Omega row from `[ ] Pending execution` → `[✅] Completed` and add the outcome.

### TODO 8.3 — Update `my_expts/what_all_I_have_done/05_vggt_omega_initialization.tex`

In [05_vggt_omega_initialization.tex](file:///data1/hemanth/mesh-splatting/my_expts/what_all_I_have_done/05_vggt_omega_initialization.tex):
- Change `[Pending Execution]` → `[Completed]`.
- Add a `\subsection{Results}` table with all the metrics.
- Add the alignment algorithm as a `\subsection{Alignment Algorithm}` section.

---

## PHASE 9 — Code Quality Checks

### TODO 9.1 — Verify Original Pipeline is Unaffected

After implementation, run a smoke test **without** the flag to confirm nothing broke:
```bash
python train.py \
  --source_path /data1/hemanth/datasets/tandt/truck \
  --model_path /tmp/smoke_test_no_vggt \
  --iterations 500 \
  --scene_name truck
```

This should run cleanly for 500 iterations with no errors. The `VGGTParams.vggt_mode == "geometry_only"` defaults to `False`, so all VGGT branches are skipped.

### TODO 9.2 — Verify Error Handling

Test that a missing PLY raises the expected error:
```bash
python train.py \
  --source_path /data1/hemanth/datasets/tandt/truck \
  --model_path /tmp/smoke_test_vggt_missing \
  --iterations 1 \
  --vggt_mode == "geometry_only"
  # (with no file at tandt/truck/vggt_omega/points3D.ply)
```

Expected: training aborts immediately with `FileNotFoundError` and a clear error message.

---

## Summary: Files to Modify

| File | What Changes |
|---|---|
| [`arguments/__init__.py`](file:///data1/hemanth/mesh-splatting/arguments/__init__.py) | Add `VGGTParams` class |
| [`scene/dataset_readers.py`](file:///data1/hemanth/mesh-splatting/scene/dataset_readers.py) | Add `load_vggt_omega_pcd()` and `align_vggt_to_colmap()` |
| [`scene/__init__.py`](file:///data1/hemanth/mesh-splatting/scene/__init__.py) | Accept `vggt_args`, intercept PCD after `scene_info` |
| [`train.py`](file:///data1/hemanth/mesh-splatting/train.py) | Import `VGGTParams`, parse it, pass to `Scene` and `training()` |
| [`scene/triangle_model.py`](file:///data1/hemanth/mesh-splatting/scene/triangle_model.py) | **No changes** — `create_from_pcd` is fully reused |

## New Files to Create

| File | Purpose |
|---|---|
| `my_expts/vggt_omega_initialised/prepare_vggt_ply.py` | Utility to validate/convert raw VGGT output → standards-compliant PLY |

---

## Implementation Order

```
Phase 0 (Prerequisites) → Phase 1 (Arguments) → Phase 2 (Loader) 
    → Phase 3 (Alignment) → Phase 4 (Scene init) → Phase 5 (train.py)
    → Phase 9.1 (smoke test no-flag) → Phase 9.2 (error test)
    → Phase 7 (Run experiments) → Phase 8 (Documentation)
```

---

## PART 2 — EXPERIMENT 2: The COLMAP-Free Pipeline

> **Goal:** Test if MeshSplatting can survive completely without COLMAP by relying purely on a feed-forward model (VGGT) for both geometry and cameras in its own native coordinate frame.

### TODO 10.1 — The Custom Dataloader
- [ ] Write a new function `readVGGTSceneInfo()` in `scene/dataset_readers.py`.
- [ ] Parse the specific JSON/TXT format that VGGT uses to output its intrinsics and extrinsics.
- [ ] Replicate the `llffhold=8` split logic to define train/test sets based on the VGGT poses.

### TODO 10.2 — The Depth Alignment Problem
- [ ] Create a new script `make_depth_scale_vggt.py` (or modify the existing one).
- [ ] The script must calculate a linear scale and shift to align the raw monocular depth maps (`Depth Anything V2`) to the **VGGT dense point cloud**, rather than the COLMAP sparse point cloud.
- [ ] Save the output to `vggt_omega/depth_params.json`.

### TODO 10.3 — Evaluation Caveats
- [ ] **Important Note:** When evaluating the Test Set for Experiment 2, we will be using VGGT's predicted test cameras. When evaluating the Baseline, we use COLMAP's test cameras. 
- [ ] Because the test cameras are slightly different, the PSNR metrics won't be perfectly comparable to Experiment 1. The real metric of success for Experiment 2 will be **visual mesh quality** and **end-to-end processing speed** (raw images → mesh).
