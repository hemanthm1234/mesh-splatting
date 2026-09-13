---
name: wandb
description: Comprehensive guide detailing all metrics, scalars, and configurations logged to Weights & Biases (WandB) in the MeshSplatting training pipeline.
---

# WandB Logging in MeshSplatting

This skill outlines every single metric and configuration currently logged to Weights & Biases (WandB) during the training of the `MeshSplatting` model. Use this guide to understand what is being tracked, when it is tracked, and how to add or modify logged metrics.

## 1. Initialization and Setup

WandB is initialized inside `train.py` just before the main training loop starts:
- **Project Name:** `meshsplatting-baseline`
- **Run Name:** Taken from the `--wandb_name` command-line argument.
- **Config:** The full dictionary of command-line arguments `vars(args)`.

### X-Axis Metric Setup
WandB is explicitly configured to use `iteration` as the x-axis step metric for all graphs.
```python
wandb.define_metric("iteration")
wandb.define_metric("train/*", step_metric="iteration")
wandb.define_metric("test/*", step_metric="iteration")
wandb.define_metric("geometry/*", step_metric="iteration")
```

---

## 2. Metrics Logged

The training pipeline logs several categories of metrics at different frequencies.

### A. Initialization (Iteration 0)
At the very beginning of the training process, the initial iteration is logged to establish the starting point.
- **`iteration`**: `first_iter` (usually 0, unless resuming from a checkpoint).

### B. Every Iteration
Inside `training_report()`, the following metrics are recorded at *every single iteration* to track the real-time loss and speed of the optimization.
- **`train/pixel_loss`**: The raw L1 loss between the rendered image and the ground truth image (`pixel_loss.item()`).
- **`train/total_loss`**: The total combined loss (L1 + SSIM) used for backpropagation (`loss.item()`).
- **`train/iter_time`**: The time taken (in milliseconds) to compute the forward and backward pass for the current iteration (`elapsed`).
- **`iteration`**: The current iteration number.

### C. Every 300 Iterations (Geometry Stats)
Every 300 iterations, the main training loop in `train.py` logs the structural complexity of the scene geometry. This is crucial for tracking the MCMC densification and pruning process.
- **`geometry/vertex_count`**: The total number of vertices currently in the triangle mesh (`triangles.vertices.shape[0]`).
- **`geometry/triangle_count`**: The total number of triangles currently in the mesh (`triangles._triangle_indices.shape[0]`). If triangle indices do not exist yet, this logs `0`.
- **`iteration`**: The current iteration number.

### D. Every 300 Iterations (Validation & Image Metrics)
Every 300 iterations, the pipeline pauses to evaluate the model on specific camera viewpoints from both the `train` and `test` splits. For each split (referred to as `{split}` below), it renders the images, compares them to ground truth, and logs the average metrics:
- **`{split}/l1_loss`**: The average L1 pixel loss across all evaluated cameras in the split.
- **`{split}/psnr`**: Peak Signal-to-Noise Ratio (PSNR), measuring image reconstruction quality (higher is better).
- **`{split}/ssim`**: Structural Similarity Index Measure (SSIM) (higher is better).
- **`{split}/lpips`**: Learned Perceptual Image Patch Similarity (LPIPS) (lower is better).
- **`{split}/fps`**: The average rendering speed in Frames Per Second, calculated as `1000.0 / average_time_per_image_ms`.
- **`iteration`**: The current iteration number.

### E. End of Training (Final Geometry)
After the training completes and the final pruning operation (removing triangles with low opacity/importance) is executed, the final geometry metrics are logged. Note: These are logged without the `iteration` metric explicitly in the dict, but use `step=iteration`.
- **`geometry/final_vertex_count`**: The final number of vertices in the exported mesh.
- **`geometry/final_triangle_count`**: The final number of triangles in the exported mesh.

---

## 3. Notes on Plots and Media
- **Scalars & Line Charts**: All metrics listed above are logged as scalars. WandB automatically generates line charts for these scalars over time, using `iteration` as the x-axis.
- **Images & Media**: During the evaluation step (every 300 iterations), rendered test images (`media/test_images`) and normal maps (`media/normal_maps`) are natively logged to WandB if they are present in the test batch. This allows for visual inspection of the model's progress over time.

## How to use this skill
If you (the agent) are asked to add new logging metrics (e.g., pruning statistics, memory usage, regularization loss components), refer to the categories above to determine the appropriate logging frequency and grouping (`train/`, `test/`, `geometry/`).
