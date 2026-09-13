"""
store_results.py — Scene-aware result aggregator for the baseline MeshSplatting experiment.

For every *_baseline directory found under output/, this script:
  1. Reads image quality metrics from per_view.json (PSNR, SSIM, LPIPS).
  2. Reads mesh stats from mesh_stats.json (vertices, faces, file size).
  3. Reads T&T geometric metrics from tnt_eval_results/tandt_metrics.json (F-score, precision, recall).
  4. Aggregates everything into /data1/hemanth/results/papers/MeshSplatting.json.
  5. Resumes each scene's WandB run (via wandb_id.txt) and logs the geometric metrics to the run summary.
"""

import os
import json
import glob
import csv

EXP_DIR = "/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
OUTPUT_BASE = os.path.join(EXP_DIR, "output")
CENTRAL_JSON = "/data1/hemanth/results/papers/MeshSplatting.json"


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def extract_image_metrics(scene_dir):
    """Read PSNR/SSIM/LPIPS from per_view.json, averaged across all test views."""
    per_view_path = os.path.join(scene_dir, "per_view.json")
    per_view = load_json(per_view_path)
    if not per_view:
        return None, None, None
    # Structure: {"ours_30000": {"SSIM": {img: val, ...}, "PSNR": {...}, "LPIPS": {...}}}
    for key, metrics in per_view.items():
        if isinstance(metrics, dict) and "PSNR" in metrics:
            psnr_vals = list(metrics["PSNR"].values())
            ssim_vals = list(metrics["SSIM"].values())
            lpips_vals = list(metrics["LPIPS"].values())
            psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else None
            ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else None
            lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else None
            return round(psnr, 4), round(ssim, 4), round(lpips, 4)
    return None, None, None


def main():
    # Load VRAM data from the downloaded CSV
    vram_map = {}
    csv_path = os.path.join(EXP_DIR, "wandb_summary_table.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Name" in row and "max_vram_gb" in row and row["max_vram_gb"]:
                    vram_map[row["Name"]] = float(row["max_vram_gb"])

    # Auto-discover all *_baseline scene directories
    scene_dirs = sorted(glob.glob(os.path.join(OUTPUT_BASE, "*_baseline")))
    if not scene_dirs:
        print(f"[!] No *_baseline directories found under {OUTPUT_BASE}")
        return

    print(f"Found {len(scene_dirs)} scene(s): {[os.path.basename(d) for d in scene_dirs]}\n")

    # Load the existing central JSON (to append, not overwrite)
    os.makedirs(os.path.dirname(CENTRAL_JSON), exist_ok=True)
    central_data = load_json(CENTRAL_JSON)
    if "baseline_MeshSplatting" not in central_data:
        central_data["baseline_MeshSplatting"] = {
            "paper_title": "MeshSplatting: Differentiable Rendering with Opaque Meshes",
            "method_name": "MeshSplatting",
            "experiment": "Baseline (30k iterations, T&T scenes)",
            "scenes": {}
        }

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir).replace("_baseline", "")
        scene_key = scene_name.lower() + "_1"
        vram_max = vram_map.get(scene_key)
        
        print(f"--- Processing: {scene_name} ---")

        # 1. Image quality metrics
        psnr, ssim, lpips = extract_image_metrics(scene_dir)
        if psnr is not None:
            print(f"  Image: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}")
        else:
            print(f"  [!] per_view.json not found or empty — skipping image metrics.")

        # 2. Mesh stats
        mesh_stats = load_json(os.path.join(scene_dir, "mesh_stats.json"))
        num_vertices = mesh_stats.get("num_vertices")
        num_faces = mesh_stats.get("num_faces")
        file_size_mb = mesh_stats.get("file_size_mb")
        if num_vertices:
            print(f"  Mesh: {num_vertices:,} vertices, {num_faces:,} faces, {file_size_mb} MB")
        else:
            print(f"  [!] mesh_stats.json not found.")

        # 3. T&T geometric metrics
        tnt_path = os.path.join(scene_dir, "tnt_eval_results", "tandt_metrics.json")
        tnt = load_json(tnt_path)
        f_score = tnt.get("f_score")
        precision = tnt.get("precision")
        recall = tnt.get("recall")
        tau = tnt.get("distance_tau")
        if f_score is not None:
            print(f"  T&T: F-score={f_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f} @ tau={tau}")
        else:
            print(f"  [!] tnt_eval_results/tandt_metrics.json not found — T&T eval not yet run for this scene.")

        # 4. Aggregate into central data structure
        central_data["baseline_MeshSplatting"]["scenes"][scene_name] = {
            "image_quality": {
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": lpips,
                "num_test_views": len(load_json(os.path.join(scene_dir, "per_view.json")).get(
                    list(load_json(os.path.join(scene_dir, "per_view.json")).keys())[0], {}).get("PSNR", {})
                ) if load_json(os.path.join(scene_dir, "per_view.json")) else None
            },
            "surface_quality": {
                "F_score": f_score,
                "Precision": precision,
                "Recall": recall,
                "distance_tau": tau,
                "num_vertices": num_vertices,
                "num_faces": num_faces,
                "file_size_mb": file_size_mb,
            },
        }

        # 5. Log geometric metrics to WandB by resuming the original training run
        wandb_id_path = os.path.join(scene_dir, "wandb_id.txt")
        if not os.path.exists(wandb_id_path):
            print(f"  [!] wandb_id.txt not found — cannot log to WandB for {scene_name}.")
            print()
            continue

        with open(wandb_id_path, 'r') as wf:
            run_id = wf.read().strip()

        try:
            import wandb
            print(f"  Resuming WandB run {run_id} to log evaluation metrics...")
            run = wandb.init(
                entity="HiLite-4D",
                project="HiLite-4D-MeshSplatting",
                id=run_id,
                resume="must",
            )

            # Log image quality metrics to summary (re-confirms what was logged during training)
            if psnr is not None:
                run.summary["eval/psnr"] = psnr
                run.summary["eval/ssim"] = ssim
                run.summary["eval/lpips"] = lpips
                
                # Clean formatted columns
                run.summary["psnr_final"] = psnr
                run.summary["ssim_final"] = ssim
                run.summary["lpips_final"] = lpips

            # Log geometric quality metrics to summary
            if f_score is not None:
                run.summary["eval/tnt_f_score"] = f_score
                run.summary["eval/tnt_precision"] = precision
                run.summary["eval/tnt_recall"] = recall
                run.summary["eval/tnt_distance_tau"] = tau
                
                # Clean formatted columns
                run.summary["f_score_tnt_eval"] = f_score
                run.summary["precision_tnt_eval"] = precision
                run.summary["recall_tnt_eval"] = recall
                
                print(f"  [WandB] Logged T&T metrics to summary: F-score={f_score:.4f}")
            else:
                print(f"  [WandB] Skipped T&T metrics (not available for this scene).")

            # Log mesh stats to summary
            if num_vertices is not None:
                run.summary["eval/mesh_vertices"] = num_vertices
                run.summary["eval/mesh_faces"] = num_faces
                run.summary["eval/mesh_size_mb"] = file_size_mb
                
                # Clean formatted columns
                run.summary["vertex_count_final"] = num_vertices
                run.summary["triangle_count_final"] = num_faces
                
            # Grab VRAM from existing wandb summary if available
            if vram_max is not None:
                run.summary["vram_gb_max"] = vram_max
                print(f"  [WandB] Logged VRAM max: {vram_max:.2f} GB")
                
            wandb.finish()
            print(f"  [WandB] Run {run_id} updated and closed.")

        except ImportError:
            print("  [!] WandB not installed. Skipping remote logging.")
        except Exception as e:
            print(f"  [!] Error logging to WandB for {scene_name}: {e}")

        print()

    # Write the updated central JSON
    with open(CENTRAL_JSON, 'w') as f:
        json.dump(central_data, f, indent=4)

    print(f"\n=== Results written to {CENTRAL_JSON} ===")
    print("\n=== Summary Table ===")
    print(f"{'Scene':<20} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'F-Score':>9} {'Vertices':>12} {'Faces':>12}")
    print("-" * 85)
    for scene_name, data in central_data["baseline_MeshSplatting"]["scenes"].items():
        iq = data.get("image_quality", {})
        sq = data.get("surface_quality", {})
        psnr_s = f"{iq['PSNR']:.4f}" if iq.get("PSNR") else "N/A"
        ssim_s = f"{iq['SSIM']:.4f}" if iq.get("SSIM") else "N/A"
        lpips_s = f"{iq['LPIPS']:.4f}" if iq.get("LPIPS") else "N/A"
        fscore_s = f"{sq['F_score']:.4f}" if sq.get("F_score") else "N/A"
        verts_s = f"{sq['num_vertices']:,}" if sq.get("num_vertices") else "N/A"
        faces_s = f"{sq['num_faces']:,}" if sq.get("num_faces") else "N/A"
        print(f"{scene_name:<20} {psnr_s:>7} {ssim_s:>7} {lpips_s:>7} {fscore_s:>9} {verts_s:>12} {faces_s:>12}")


if __name__ == "__main__":
    main()
