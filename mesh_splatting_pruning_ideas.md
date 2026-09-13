# Comprehensive Report: Pruning and Efficiency Ideas for MeshSplatting

## Executive Summary & Agent Validation

| Idea | Thoughts & Validation | Doability | Potential Impact | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **1.1 Context-Adaptive Gradient Verification** | MeshSplatting currently splits triangles based on area and probabilistic `importance_score` sampling (`add_new_gs`). It ignores gradient momentum. Tracking positional gradient EMA is highly feasible and prevents splitting oscillating floaters. | High | High (Training speed & less floaters) | **Immediate Action**. Add EMA tracking to optimizer step. |
| **1.2 Saliency-Guided Mesh Decimation (QEM)** | We currently only prune triangles via opacity/size thresholds. After the `effrdel` phase (iter > 6k) creates a connected mesh, applying QEM to collapse edges in flat areas is very promising. Differentiable collapse is hard, but discrete QEM steps are easy. | Medium | High (Final mesh size) | **Do next**. Apply standard QEM periodically after Delaunay. |
| **1.3 Z-Order Spatial Pooling** | Morton-coded pooling is a fast heuristic, but QEM (Idea 1.2) is geometrically safer for explicit meshes. | High | Medium | **Skip for now**, prioritize QEM. |
| **2.1 First-Order SH + MLP** | `TriangleModel` currently stores full degree 3 SH (huge memory). Dropping SH to degree 1 (or 0) and adding a tiny MLP decoder in the CUDA rasterizer is a proven memory optimization. | Medium | High (VRAM reduction) | **High Priority**. Reduces memory bandwidth drastically. |
| **2.2 Inter-Frame Rendering Cache** | Highly effective for smooth inference/rendering. Not useful for random-view training. | Medium | High (Inference FPS) | **Post-training task**. Good for real-time viewer. |
| **3.1 Voxel-Tethered Dual Scaffolding** | We already use `vertex_depth_loss_hr` to guide depth. An explicit voxel grid adds too much complexity. | Low | Low (Redundant) | **Discard**. Stick to depth priors. |

---

## 1. Advanced Densification and Pruning Strategies
*Currently, MeshSplatting uses heuristic-based pruning (opacity, bounding box size, and importance/blending score) and random/area-based densification (`_sample_alives` and `topk` area). These can be upgraded for better efficiency.*

### 1.1 Context-Adaptive Gradient Verification (Inspired by CAdam)
* **The Concept:** Standard densification relies on spatial size or opacity. Instead, track the temporal momentum (Exponential Moving Average) of the positional gradients of the vertices.
* **Adaptation to MeshSplatting:** In `scene/triangle_model.py`, the `add_new_gs` method blindly splits large and "important" triangles. If a vertex's gradient direction is highly inconsistent (oscillating), it indicates a "floater" or noise. We can safely prune triangles connected to such vertices. Conversely, edges connecting vertices with strong, consistent directional momentum should be prioritized for splitting.
* **Expected Gain:** Up to 80% reduction in unnecessary triangle splits during training, drastically reducing the final primitive count.
* **Validation:** **Highly Valid.** This is a direct upgrade to the `add_new_gs` method. We can tap into the optimizer's state to retrieve gradient momentum without adding heavy compute overhead.

### 1.2 Geometry and Saliency-Guided Mesh Decimation (Inspired by G²ARD-GS & Saliency-Guided Merging)
* **The Concept:** Not all regions of a scene require high triangle density. Flat surfaces (walls, roads) can be represented by large triangles, while high-frequency textures require dense micro-triangles.
* **Adaptation to MeshSplatting:** MeshSplatting transitions from independent triangles to a connected mesh using Restricted Delaunay Triangulation (`effrdel`) around iteration 6000. Post-Delaunay, we can perform periodic discrete edge collapses using Quadric Error Metrics (QEM). By projecting 2D saliency maps, we ensure edge collapse only occurs in low-texture, flat regions.
* **Expected Gain:** Massive reduction in triangle count for structural backgrounds, leading to faster inference and smaller PLY file sizes.
* **Validation:** **Valid but tricky.** Implementing differentiable edge collapse is very complex. However, applying a discrete QEM reduction step (e.g., using `trimesh` or `open3d`) every few thousand iterations after the Delaunay phase is highly doable and will yield massive compression.

### 1.3 Z-Order Spatial Pooling (Inspired by Z-Order Transformer)
* **The Concept:** Serialize 3D coordinates into a 1D sequence using a Z-order (Morton) curve to group spatially proximal primitives.
* **Adaptation to MeshSplatting:** Sort triangle barycenters via Morton coding. Triangles sharing the same Z-order prefix are inherently close. During training, apply a lightweight pooling operation that merges co-planar triangles within the same spatial bucket into a single, larger triangle.
* **Expected Gain:** Faster heuristic spatial decimation.
* **Validation:** **Low Priority.** While fast, Morton pooling is blind to precise topology compared to QEM. For a connected mesh (which MeshSplatting aims for post-Delaunay), QEM is mathematically superior for preserving sharp features.

---

## 2. Rendering and Memory Optimizations
*MeshSplatting currently stores high-degree Spherical Harmonics (SH) per vertex and relies on standard CUDA rasterization loops for every frame.*

### 2.1 First-Order SH with Monte Carlo Energy Aggregation (Inspired by Flux-GS)
* **The Concept:** 3rd-degree SH requires storing 48 floats per vertex (16 coefficients × 3 colors), consuming immense VRAM and memory bandwidth.
* **Adaptation to MeshSplatting:** Drop the SH degree to 1 (only 12 floats per vertex) or even degree 0. To recover view-dependent effects, implement a lightweight feature decoder (a tiny MLP) in the CUDA rasterizer. The MLP takes the view direction and the baseline vertex features to predict the final color. 
* **Expected Gain:** ~75% reduction in vertex attribute memory size and significantly faster rasterization memory reads.
* **Validation:** **Highly Valid.** This is a proven technique (used in Scaffold-GS and RadSplat). Given that MeshSplatting forces fully opaque triangles eventually, memory bandwidth is a major bottleneck. Replacing explicit SH with an MLP is a highly effective trade-off.

### 2.2 Inter-Frame Rendering Cache (Inspired by CaT-GS)
* **The Concept:** In large-scale scenes or continuous video rendering, camera viewpoints move smoothly. Depth-sorting and frustum culling millions of triangles per frame is highly redundant.
* **Adaptation to MeshSplatting:** Implement speculative multi-frame caching in `triangle_renderer`. For a "keyframe", calculate the expanded frustum bounds and perform depth sorting. For the next `N` sub-frames, completely skip the sorting and culling steps, reusing the keyframe's triangle list.
* **Expected Gain:** Up to 5-10x speedup in real-time inference FPS for continuous camera trajectories.
* **Validation:** **Valid for Inference.** This will strictly accelerate `render.py` and real-time viewers. During training (`train.py`), cameras are sampled randomly, so inter-frame caching is ineffective unless sequential training is adopted.

---

## 3. Geometric Regularization for Cleaner Topology
*Although MeshSplatting uses depth priors and normal losses, floaters and internal unseen geometry still consume compute.*

### 3.1 Voxel-Tethered Dual Scaffolding (Inspired by Gaussian-Voxel Duet)
* **The Concept:** Bind explicit primitives to a sparse implicit grid to enforce absolute surface boundaries.
* **Adaptation to MeshSplatting:** Introduce a sparse voxel octree that covers the scene to learn a coarse SDF. Apply a "tethering loss" forcing vertices to stay near the zero-level set.
* **Expected Gain:** Faster convergence for clean surfaces.
* **Validation:** **Not Recommended.** MeshSplatting already heavily relies on monocular depth priors (`vertex_depth_loss_hr`) to guide geometry. Adding an explicit voxel grid introduces massive overhead and complexity that overlaps with the existing depth supervision.

---

## 4. Extensions to 4D and Dynamic Scenes (Future Scope)
*Once static scenes are optimized, extending these efficiency principles to 4D requires careful handling of motion.*

### 4.1 Flow Splatting for Differentiable Kinematics (Inspired by Flow Splatting)
* **The Concept:** Supervising 3D motion purely through photometric loss is highly inefficient and requires dense cameras.
* **Adaptation to MeshSplatting:** Render a 3D velocity field into a 2D screen-space optical flow map. Supervise this using a pre-trained 2D optical flow network (e.g., RAFT). This forces triangles to follow true physical motion.
* **Validation:** **Highly Valid.** This ensures triangles move coherently instead of deleting and spawning new ones to simulate motion.

### 4.2 Decoupled Topology Registration (Inspired by DecoupleGS & MVFusion)
* **The Concept:** Do not re-render or deform the entire scene for dynamic movements.
* **Adaptation to MeshSplatting:** Segregate the mesh into a `Static Background Mesh` and multiple `Dynamic Foreground Meshes`. Only foreground meshes undergo temporal deformation updates via an MLP.
* **Validation:** **Highly Valid.** Essential for efficient 4D, as most of the scene remains static.

---

## Conclusion and Recommended Next Steps
Based on our codebase analysis, the immediate priorities should be:

1. **Phase 1 (Low Effort, High Reward):** Implement **Context-Adaptive Gradient Verification (1.1)** in `train.py` and `scene/triangle_model.py`. Modify `add_new_gs` to track the EMA of vertex gradients, ensuring we only split triangles with structural meaning.
2. **Phase 2 (Architecture Change):** Implement **First-Order SH + MLP (2.1)**. Modify `TriangleModel` to initialize SH with lower degree and update the custom CUDA rasterizer to include the MLP inference.
3. **Phase 3 (Medium Effort):** Implement **Saliency-Guided Decimation (1.2)** by executing a discrete QEM collapse algorithm on the mesh at regular intervals *after* `run_restricted_delaunay` triggers in `train.py`.
