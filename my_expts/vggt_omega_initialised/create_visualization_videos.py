import argparse
import os
import subprocess
import concurrent.futures
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapy as media

# Base paths
PROJECT_ROOT = "/data1/hemanth/mesh-splatting"
BASE_DATASET_DIR = "/data1/hemanth/datasets/tandt"
CENTRAL_VIS_DIR = os.path.join(PROJECT_ROOT, "my_expts", "visualization_videos")
INDIVIDUAL_VIS_DIR = os.path.join(CENTRAL_VIS_DIR, "individual")
GRIDS_VIS_DIR = os.path.join(CENTRAL_VIS_DIR, "grids")

os.makedirs(INDIVIDUAL_VIS_DIR, exist_ok=True)
os.makedirs(GRIDS_VIS_DIR, exist_ok=True)

# Define configurations and their base paths
CONFIGS = {
    "Baseline": os.path.join(PROJECT_ROOT, "my_expts", "baseline_mesh_splatting_metrics", "output", "{}_baseline"),
    "VGGT 10M": os.path.join(PROJECT_ROOT, "my_expts", "vggt_omega_initialised", "output_full_10M", "{}"),
    "VGGT Colmap": os.path.join(PROJECT_ROOT, "my_expts", "vggt_omega_initialised", "output_colmap_base", "{}"),
    "VGGT Colmap NoDP": os.path.join(PROJECT_ROOT, "my_expts", "vggt_omega_initialised", "output_colmap_noDP", "{}")
}

# The videos we want to create
TARGET_VIDEOS = {
    "truck_visualization.mp4": {
        "layout": "2x2",
        "inputs": [
            {"scene": "Truck", "config": "Baseline", "label": "COLMAP Init (Baseline)"},
            {"scene": "Truck", "config": "VGGT 10M", "label": "VGGT 10M Init"},
            {"scene": "Truck", "config": "VGGT Colmap", "label": "VGGT COLMAP Init"},
            {"scene": "Truck", "config": "VGGT Colmap NoDP", "label": "VGGT COLMAP Init (NoDP)"}
        ]
    },
    "ignatius_visualization.mp4": {
        "layout": "3_centered_top",
        "inputs": [
            {"scene": "Ignatius", "config": "Baseline", "label": "COLMAP Init (Baseline)"},
            {"scene": "Ignatius", "config": "VGGT 10M", "label": "VGGT 10M Init"},
            {"scene": "Ignatius", "config": "VGGT Colmap", "label": "VGGT COLMAP Init"}
        ]
    },
    "barn_caterpillar_visualization.mp4": {
        "layout": "2x2",
        "inputs": [
            {"scene": "Barn", "config": "Baseline", "label": "Barn COLMAP Init (Baseline)"},
            {"scene": "Barn", "config": "VGGT 10M", "label": "Barn VGGT 10M Init"},
            {"scene": "Caterpillar", "config": "Baseline", "label": "Caterpillar COLMAP Init (Baseline)"},
            {"scene": "Caterpillar", "config": "VGGT 10M", "label": "Caterpillar VGGT 10M Init"}
        ]
    }
}

def render_video_if_missing(scene, config_name, fps, rebuild=False, verbose=True):
    """Renders the video for a given scene and config if it doesn't already exist."""
    base_path_template = CONFIGS[config_name]
    model_path = base_path_template.format(scene)
    config_safe = config_name.replace(" ", "_")
    target_video_name = f"{scene}_{config_safe}_360_30k.mp4"
    central_video_path = os.path.join(INDIVIDUAL_VIS_DIR, target_video_name)
    local_video_path = os.path.join(model_path, "rendered_video_30000.mp4")

    if os.path.exists(central_video_path) and not rebuild:
        if verbose:
            print(f"✅ Individual video already exists: {central_video_path}")
        return central_video_path

    if not os.path.exists(os.path.join(model_path, "point_cloud", "iteration_30000")):
        raise FileNotFoundError(f"Model for {scene} ({config_name}) has not finished training! Missing iteration_30000 at {model_path}")

    print(f"⏳ Generating individual video for {scene} ({config_name}) -> {central_video_path} (FPS: {fps})...")
    dataset_path = os.path.join(BASE_DATASET_DIR, scene if config_name != "Baseline" else scene.lower())
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DATASET_DIR, scene)

    cmd = [
        "python", "create_video.py",
        "-s", dataset_path,
        "-m", model_path,
        "--iteration", "30000",
        "--save_as", central_video_path,
        "--sync_mode",
        "--fps", str(fps)
    ]
    if "VGGT" in config_name:
        cmd.extend(["--vggt_mode", "full_pipeline"])

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Finished generating individual video for {scene} ({config_name})")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed rendering for {scene} ({config_name}):\n{e.stderr.decode('utf-8')}")
        raise e
        
    return central_video_path


def add_label_to_frame(frame_bgr, text):
    """Draws a styled text label top-center over a frame with a semi-transparent box."""
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    w, h = img_pil.size
    x = (w - text_w) // 2
    y = 12
    padding = 6
    
    box_coords = [x - padding, y - padding, x + text_w + padding, y + text_h + padding]
    draw.rectangle(box_coords, fill=(0, 0, 0, 160), outline=(255, 255, 255, 180), width=1)
    
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    frame_rgb = np.array(img_pil)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def create_grid_video(output_name, layout, inputs, fps, rebuild=False):
    output_path = os.path.join(GRIDS_VIS_DIR, output_name)
    if os.path.exists(output_path) and not rebuild:
        print(f"✅ Grid video already exists: {output_path}")
        return

    print(f"\n🎥 Generating grid video: {output_path}...")
    
    caps = []
    for inp in inputs:
        video_path = render_video_if_missing(inp["scene"], inp["config"], fps=fps, rebuild=rebuild, verbose=False)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Missing sub-part video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")
        caps.append(cap)
        
    # Read first frame to get base dimensions and frame count
    ret, first_frame = caps[0].read()
    if not ret:
        raise RuntimeError("Failed to read first frame from video")
        
    orig_h, orig_w = first_frame.shape[:2]
    half_h, half_w = orig_h // 2, orig_w // 2
    # Ensure fps is set to what we want for the grid video as well
    out_fps = fps
    
    # Reset all captures back to beginning
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (orig_w, orig_h))
    
    try:
        while True:
            sub_frames = []
            all_ret = True
            for i, cap in enumerate(caps):
                ret, frame_bgr = cap.read()
                if not ret:
                    all_ret = False
                    break
                # Scale down to 0.5x
                frame_resized = cv2.resize(frame_bgr, (half_w, half_h), interpolation=cv2.INTER_AREA)
                # Add label directly
                frame_labeled = add_label_to_frame(frame_resized, inputs[i]["label"])
                sub_frames.append(frame_labeled)
                
            if not all_ret or len(sub_frames) < len(inputs):
                break
                
            # Layout composition
            if layout == "2x2":
                top_row = np.hstack([sub_frames[0], sub_frames[1]])
                bot_row = np.hstack([sub_frames[2], sub_frames[3]])
                canvas = np.vstack([top_row, bot_row])
            elif layout == "3_centered_top":
                canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                x_offset = (orig_w - half_w) // 2
                canvas[0:half_h, x_offset:x_offset + half_w] = sub_frames[0]
                canvas[half_h:orig_h, 0:half_w] = sub_frames[1]
                canvas[half_h:orig_h, half_w:orig_w] = sub_frames[2]
            else:
                raise ValueError(f"Unknown layout: {layout}")
                
            writer.write(canvas)
            
        writer.release()
        print(f"✅ Successfully created {output_name}")
    finally:
        for cap in caps:
            cap.release()

def main():
    parser = argparse.ArgumentParser(description="Create grid visualization videos.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output videos.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild videos even if they exist (reuses frames).")
    args = parser.parse_args()

    print(f"\n🎥 Starting Visualization Generation for {len(TARGET_VIDEOS)} Grid Videos (FPS: {args.fps}, Rebuild: {args.rebuild})...")
    
    for vid_name, data in TARGET_VIDEOS.items():
        print(f"\n=======================================================")
        print(f"🎬 Processing Composite Video: {vid_name}")
        print(f"=======================================================")
        
        inputs = data["inputs"]
        
        # Render sub-parts sequentially
        for inp in inputs:
            scene = inp["scene"]
            config = inp["config"]
            try:
                render_video_if_missing(scene, config, fps=args.fps, rebuild=args.rebuild, verbose=True)
            except Exception as e:
                print(f"❌ Error occurred during rendering {scene} ({config}): {e}")
                return # Abort if rendering fails
                    
        # Composite this grid video
        create_grid_video(vid_name, data["layout"], inputs, fps=args.fps, rebuild=args.rebuild)
        
    print(f"\n🎉 All visualization videos successfully saved under {CENTRAL_VIS_DIR}!")

if __name__ == "__main__":
    main()

