import cv2
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_dir", type=str, required=True, help="Directory containing view_* folders")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    args = parser.parse_args()

    progress_dir = args.progress_dir
    output_dir = os.path.dirname(progress_dir)
    
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
            
        print(f"Rendering video for {view}...")
        
        # Read first image to get dimensions
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        
        # Define the codec and create VideoWriter object
        video_path = os.path.join(output_dir, f"progress_{view}.mp4")
        # Use mp4v codec for mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        
        for image_path in images:
            video.write(cv2.imread(image_path))
            
        cv2.destroyAllWindows()
        video.release()
        
        print(f"Saved to {video_path}")

if __name__ == "__main__":
    main()
