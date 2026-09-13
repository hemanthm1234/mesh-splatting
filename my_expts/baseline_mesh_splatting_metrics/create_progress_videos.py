#!/usr/bin/env python3
"""
create_progress_videos.py

Generates MP4 progress videos for training views across iterations using OpenCV (cv2).
Includes a sleek top horizontal progress bar with milestone tick marks (0, 10k, 20k, 30k)
and a smooth animated indicator/fill bar representing iteration progress.

Usage:
  # Process a single progress directory:
  python create_progress_videos.py --path my_expts/baseline_mesh_splatting_metrics/output/Truck_baseline/progress

  # Process all progress directories under output/:
  python create_progress_videos.py --path my_expts/baseline_mesh_splatting_metrics/output --fps 10
"""

import argparse
import glob
import os
import re
import sys
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create MP4 progress videos for training views using OpenCV with an animated top progress bar."
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True,
        help="Path to a progress folder (containing view_* dirs) or a parent folder containing scene outputs."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="/data1/hemanth/mesh-splatting/my_expts/visualization_videos/progress",
        help="Target directory to save generated progress MP4 videos."
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frame rate of generated videos (default: 10)."
    )
    parser.add_argument(
        "--codec", type=str, default="mp4v",
        help="FourCC codec for MP4 video writing (default: 'mp4v')."
    )
    parser.add_argument(
        "--max-iter", type=int, default=30000,
        help="Maximum iteration count for the progress slider scale (default: 30000)."
    )
    parser.add_argument(
        "--no-bar", action="store_true",
        help="Disable the top progress bar overlay."
    )
    return parser.parse_args()


def get_iteration_num(filename):
    """Extract integer iteration number from filename (e.g., iter_00300.png -> 300)."""
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 0


def draw_top_progress_bar(frame, current_iter, max_iter=30000):
    """Draws a sleek top horizontal progress bar with milestone markings and fill indicator."""
    height, width, _ = frame.shape

    # Layout geometry relative to image width/height
    bar_margin_x = int(width * 0.12)       # 12% margin left & right
    bar_width = width - (2 * bar_margin_x)
    bar_y = int(height * 0.045)             # Vertical center of bar (4.5% down)
    bar_thickness = max(6, int(height * 0.008))

    box_y1 = max(5, bar_y - int(height * 0.025))
    box_y2 = min(height - 5, bar_y + int(height * 0.045))
    box_x1 = bar_margin_x - 25
    box_x2 = width - bar_margin_x + 25

    # 1. Draw semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 18, 18), -1)
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 60), 1)
    alpha = 0.70
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Draw track background bar (dark grey)
    track_y1 = bar_y - (bar_thickness // 2)
    track_y2 = bar_y + (bar_thickness // 2)
    cv2.rectangle(frame, (bar_margin_x, track_y1), (bar_margin_x + bar_width, track_y2), (50, 50, 50), -1)

    # 3. Calculate current fill fraction
    progress_frac = min(1.0, max(0.0, current_iter / float(max_iter)))
    fill_w = int(bar_width * progress_frac)

    # 4. Draw filled active progress bar (Bright Cyan / Light Blue: BGR -> (235, 170, 40))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_margin_x, track_y1), (bar_margin_x + fill_w, track_y2), (235, 170, 40), -1)

    # 5. Draw milestone ticks & text labels (0, 10k, 20k, 30k)
    milestones = [
        (0, "0"),
        (10000, "10k"),
        (20000, "20k"),
        (max_iter, f"{max_iter // 1000}k")
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, width / 2000.0)
    text_thickness = 1

    tick_half_h = max(6, int(bar_thickness * 1.2))

    for m_val, label in milestones:
        m_frac = m_val / float(max_iter)
        m_x = bar_margin_x + int(bar_width * m_frac)

        # Tick line
        tick_color = (255, 255, 255) if current_iter >= m_val else (140, 140, 140)
        cv2.line(frame, (m_x, bar_y - tick_half_h), (m_x, bar_y + tick_half_h), tick_color, 2)

        # Milestone text below tick
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        text_x = m_x - (tw // 2)
        text_y = bar_y + tick_half_h + th + 6
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, tick_color, text_thickness, cv2.LINE_AA)

    # 6. Draw moving thumb / indicator knob at current position
    thumb_x = bar_margin_x + fill_w
    thumb_radius = max(6, int(bar_thickness * 1.1))
    cv2.circle(frame, (thumb_x, bar_y), thumb_radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (thumb_x, bar_y), thumb_radius, (235, 170, 40), 2, cv2.LINE_AA)

    return frame


def process_progress_dir(progress_dir, fps, codec, max_iter=30000, show_bar=True, output_dir=None):
    """Generates MP4 videos for each view_* folder inside progress_dir."""
    print(f"\n==================================================")
    print(f"Processing progress directory: {progress_dir}")
    print(f"==================================================")

    view_dirs = sorted(glob.glob(os.path.join(progress_dir, "view_*")))
    if not view_dirs:
        print(f"[!] No view_* subdirectories found in {progress_dir}")
        return

    # Derive scene name from parent path (e.g. output/Truck_baseline/progress -> Truck_baseline)
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(progress_dir)))
    if not parent_dir or parent_dir == "output":
        scene_prefix = os.path.basename(os.path.abspath(progress_dir))
    else:
        scene_prefix = parent_dir

    if output_dir:
        target_dir = output_dir
    else:
        target_dir = progress_dir

    os.makedirs(target_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    for view_dir in view_dirs:
        if not os.path.isdir(view_dir):
            continue

        view_name = os.path.basename(view_dir)
        img_paths = sorted(
            glob.glob(os.path.join(view_dir, "*.png")) + glob.glob(os.path.join(view_dir, "*.jpg")),
            key=get_iteration_num
        )

        if not img_paths:
            print(f"  [!] Skipping {view_name}: No image frames found.")
            continue

        output_video_path = os.path.join(target_dir, f"{scene_prefix}_{view_name}.mp4")

        first_frame = cv2.imread(img_paths[0])
        if first_frame is None:
            print(f"  [!] Skipping {view_name}: Failed to read image {img_paths[0]}")
            continue

        height, width, _ = first_frame.shape
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"  [!] Failed to open VideoWriter for {output_video_path}")
            continue

        written_frames = 0
        for img_path in img_paths:
            frame = cv2.imread(img_path)
            if frame is not None:
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))

                if show_bar:
                    iter_num = get_iteration_num(img_path)
                    frame = draw_top_progress_bar(frame, iter_num, max_iter=max_iter)

                writer.write(frame)
                written_frames += 1

        writer.release()
        print(f"  ✅ Saved: {output_video_path} ({written_frames} frames @ {fps} FPS, {width}x{height})")


def find_all_progress_dirs(root_path):
    """Finds all progress directories under root_path."""
    root_path = os.path.abspath(root_path)

    if glob.glob(os.path.join(root_path, "view_*")):
        return [root_path]

    if os.path.basename(root_path) == "progress":
        return [root_path]

    progress_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == "progress" or any(d.startswith("view_") for d in dirnames):
            if glob.glob(os.path.join(dirpath, "view_*")):
                progress_dirs.append(dirpath)

    return sorted(list(set(progress_dirs)))


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"[!] Error: Provided path '{args.path}' does not exist.")
        sys.exit(1)

    progress_dirs = find_all_progress_dirs(args.path)

    if not progress_dirs:
        print(f"[!] No progress directories found under '{args.path}'.")
        sys.exit(1)

    print(f"Found {len(progress_dirs)} progress directory/directories to process.")
    for pdir in progress_dirs:
        process_progress_dir(
            pdir, args.fps, args.codec, max_iter=args.max_iter, show_bar=not args.no_bar, output_dir=args.output_dir
        )

    print("\n🎉 All progress video generations complete!")


if __name__ == "__main__":
    main()
