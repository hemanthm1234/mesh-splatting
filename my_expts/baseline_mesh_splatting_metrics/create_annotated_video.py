import cv2
import os
import glob
import argparse
import pandas as pd
import wandb

def fetch_metrics():
    print("Fetching metrics from WandB API...")
    try:
        api = wandb.Api()
        # Specify the entity and project name that was used in train.py ("meshsplatting-baseline")
        # WandB API uses format "entity/project"
        runs = api.runs(path="hemanth-wandb-iit-madras/meshsplatting-baseline", filters={"display_name": "mesh_splatting_baseline_truck"})
        if not runs:
            print("Error: Could not find WandB run 'mesh_splatting_baseline_truck'")
            return None
            
        run = runs[0]
        print(f"Found run: {run.name} ({run.id})")
        
        # We need the full history for all evaluation steps
        # Use a large samples value to prevent downsampling
        history = run.history(samples=2000)
        
        metrics_by_iter = {}
        for index, row in history.iterrows():
            iteration = row.get('iteration')
            if pd.isna(iteration):
                continue
            iteration = int(iteration)
            
            if iteration not in metrics_by_iter:
                metrics_by_iter[iteration] = {}
                
            if 'geometry/triangle_count' in row and not pd.isna(row['geometry/triangle_count']):
                metrics_by_iter[iteration]['triangles'] = int(row['geometry/triangle_count'])
            if 'geometry/vertex_count' in row and not pd.isna(row['geometry/vertex_count']):
                metrics_by_iter[iteration]['vertices'] = int(row['geometry/vertex_count'])
            if 'test/psnr' in row and not pd.isna(row['test/psnr']):
                metrics_by_iter[iteration]['psnr'] = float(row['test/psnr'])
            if 'test/ssim' in row and not pd.isna(row['test/ssim']):
                metrics_by_iter[iteration]['ssim'] = float(row['test/ssim'])
            if 'test/lpips' in row and not pd.isna(row['test/lpips']):
                metrics_by_iter[iteration]['lpips'] = float(row['test/lpips'])
                
        return metrics_by_iter
    except Exception as e:
        print(f"Error fetching from WandB: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_dir", type=str, default="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics/output/Truck_baseline/progress")
    parser.add_argument("--output_dir", type=str, default="/data1/hemanth/mesh-splatting/my_expts/visualization_videos/progress")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    metrics = fetch_metrics()
    if not metrics:
        print("Failed to get metrics, proceeding with empty metrics.")
        metrics = {}
        
    progress_dir = args.progress_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(progress_dir)))
    scene_prefix = parent_dir if parent_dir and parent_dir != "output" else "Truck_baseline"
    
    if not os.path.exists(progress_dir):
        print(f"Error: Progress directory {progress_dir} not found.")
        return

    views = sorted([d for d in os.listdir(progress_dir) if os.path.isdir(os.path.join(progress_dir, d))])
    
    for view in views:
        view_path = os.path.join(progress_dir, view)
        images = sorted(glob.glob(os.path.join(view_path, "*.png")))
        
        if not images:
            print(f"No images found in {view_path}")
            continue
            
        print(f"Rendering annotated video for {view}...")
        
        # Read first image to get dimensions
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        
        video_path = os.path.join(output_dir, f"{scene_prefix}_annotated_progress_{view}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        
        for image_path in images:
            img = cv2.imread(image_path)
            
            basename = os.path.basename(image_path)
            try:
                iteration = int(basename.replace("iter_", "").replace(".png", ""))
            except:
                iteration = -1
                
            def get_closest_metric(metric_name):
                available = [it for it, m in metrics.items() if metric_name in m]
                if not available:
                    return 'N/A'
                closest_it = min(available, key=lambda x: abs(x - iteration))
                return metrics[closest_it][metric_name]
            
            # Setup text drawing
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            color = (255, 255, 255) # White
            bg_color = (0, 0, 0) # Black
            
            def draw_text_with_bg(img, text, pos, align="left"):
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                if align == "center":
                    x = pos[0] - tw // 2
                elif align == "right":
                    x = pos[0] - tw
                else:
                    x = pos[0]
                y = pos[1]
                
                pad = 5
                # Draw black background rectangle for visibility
                cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), bg_color, -1)
                # Draw white text
                cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

            # --- TOP SECTION ---
            t_count = get_closest_metric('triangles')
            draw_text_with_bg(img, f"Triangles: {t_count}", (20, 40), align="left")
            
            draw_text_with_bg(img, f"Iter: {iteration}", (width // 2, 40), align="center")
            
            v_count = get_closest_metric('vertices')
            draw_text_with_bg(img, f"Vertices: {v_count}", (width - 20, 40), align="right")
            
            # --- BOTTOM SECTION ---
            psnr = get_closest_metric('psnr')
            if isinstance(psnr, float): psnr = f"{psnr:.2f}"
            draw_text_with_bg(img, f"PSNR: {psnr}", (20, height - 30), align="left")
            
            ssim = get_closest_metric('ssim')
            if isinstance(ssim, float): ssim = f"{ssim:.3f}"
            draw_text_with_bg(img, f"SSIM: {ssim}", (width // 2, height - 30), align="center")
            
            lpips = get_closest_metric('lpips')
            if isinstance(lpips, float): lpips = f"{lpips:.3f}"
            draw_text_with_bg(img, f"LPIPS: {lpips}", (width - 20, height - 30), align="right")
            
            video.write(img)
            
        video.release()
        print(f"Saved to {video_path}")

if __name__ == "__main__":
    main()
