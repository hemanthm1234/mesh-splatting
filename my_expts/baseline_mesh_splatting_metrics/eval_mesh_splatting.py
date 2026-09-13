import argparse
import os
import subprocess
import open3d as o3d
import sys

def main():
    parser = argparse.ArgumentParser(description="Evaluate MeshSplatting natively on Tanks and Temples")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to T&T evaluation dataset folder (e.g. .../tandt_eval_data/Truck)")
    parser.add_argument("--ply-path", type=str, required=True, help="Path to the output mesh (e.g. native_mesh_30000.ply)")
    parser.add_argument("--tnt-eval-dir", type=str, default="/data1/hemanth/tandt_eval", help="Path to TanksAndTemples evaluation repository")
    parser.add_argument("--num-samples", type=int, default=10000000, help="Number of points to sample from the mesh surface")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for evaluation results")
    args = parser.parse_args()

    scene = os.path.basename(os.path.normpath(args.dataset_dir))
    
    print(f"--- MeshSplatting Evaluation Wrapper for {scene} ---")
    print(f"Loading mesh from: {args.ply_path}")
    
    # 1. Load the mesh
    mesh = o3d.io.read_triangle_mesh(args.ply_path)
    if not mesh.has_triangles():
        print("ERROR: The provided PLY file does not contain triangles. Is it already a point cloud?")
        sys.exit(1)
        
    print(f"Mesh loaded successfully with {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices.")
    
    # 2. Densely sample points from the surface
    print(f"Sampling {args.num_samples} points uniformly from the mesh surface...")
    dense_pc = mesh.sample_points_uniformly(number_of_points=args.num_samples)
    print(f"Sampled point cloud has {len(dense_pc.points)} points.")
    
    # 3. Save the dense point cloud to a temporary file
    temp_ply_path = args.ply_path.replace(".ply", "_dense_pc.ply")
    print(f"Saving dense point cloud to: {temp_ply_path}")
    o3d.io.write_point_cloud(temp_ply_path, dense_pc)
    
    # 4. Construct arguments for the official evaluation script
    run_script = os.path.join(args.tnt_eval_dir, "python_toolbox", "evaluation", "run.py")
    traj_path = os.path.join(args.dataset_dir, f"{scene}_COLMAP_SfM.log")
    
    cmd = [
        sys.executable, run_script,
        "--dataset-dir", args.dataset_dir,
        "--traj-path", traj_path,
        "--ply-path", temp_ply_path
    ]
    
    if args.out_dir:
        cmd.extend(["--out-dir", args.out_dir])
        
    print("\n--- Executing Official Tanks & Temples Evaluation ---")
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # We must run the subprocess from the evaluation directory so it finds config.py
    eval_dir = os.path.join(args.tnt_eval_dir, "python_toolbox", "evaluation")
    
    try:
        subprocess.run(cmd, check=True, cwd=eval_dir)
    except subprocess.CalledProcessError as e:
        print(f"\nEvaluation script failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    
    print("\nEvaluation completed successfully!")
    print(f"You can find the results in the evaluation output directory.")

if __name__ == "__main__":
    main()
