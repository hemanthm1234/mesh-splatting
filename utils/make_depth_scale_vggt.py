from joblib import Parallel, delayed
import numpy as np
import argparse
import cv2
import json
import os
from plyfile import PlyData

def process_single_image(idx, img_name, E, K, pts_xyzs, depths_dir):
    print(f"Processing {img_name}...")
    # Transform points to camera space
    pts_cam = pts_xyzs @ E[:3, :3].T + E[:3, 3]
    
    # Filter points in front of camera
    valid_z = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid_z]
    
    n_remove = len(img_name.split('.')[-1]) + 1
    key_no_ext = img_name[:-n_remove]
    
    if len(pts_cam) == 0:
        return key_no_ext, {"scale": 0.0, "offset": 0.0}
        
    # Project to 2D
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (x / z) * fx + cx
    v = (y / z) * fy + cy
    
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)
    
    depth_png_path = os.path.join(depths_dir, f"{key_no_ext}.png")
    if not os.path.exists(depth_png_path):
        return key_no_ext, {"scale": 0.0, "offset": 0.0}
        
    invmonodepthmap = cv2.imread(depth_png_path, cv2.IMREAD_UNCHANGED)
    if invmonodepthmap is None:
        return key_no_ext, {"scale": 0.0, "offset": 0.0}
        
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]
        
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    H, W = invmonodepthmap.shape
    
    # Dynamically infer the prediction resolution from the principal point
    pred_W = cx * 2.0
    pred_H = cy * 2.0
    
    u = (u / pred_W) * W
    v = (v / pred_H) * H
    
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]
    
    if len(u) == 0:
        return key_no_ext, {"scale": 0.0, "offset": 0.0}
        
    # Z-buffer splatting: keep min z for each pixel
    flat_indices = v.astype(np.int64) * W + u.astype(np.int64)
    sort_idx = np.argsort(z)
    flat_indices = flat_indices[sort_idx]
    z_sorted = z[sort_idx]
    
    unique_indices, unique_first = np.unique(flat_indices, return_index=True)
    z_min = z_sorted[unique_first]
    
    v_unique = unique_indices // W
    u_unique = unique_indices % W
    
    invcolmapdepth = 1.0 / z_min
    invmonodepth = invmonodepthmap[v_unique, u_unique]
    
    if len(invcolmapdepth) > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))
        
        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        
        if s_mono > 1e-6:
            scale = float(s_colmap / s_mono)
            offset = float(t_colmap - t_mono * scale)
        else:
            scale, offset = 0.0, 0.0
    else:
        scale, offset = 0.0, 0.0
        
    return key_no_ext, {"scale": scale, "offset": offset}

def compute_depth_params(args):
    vggt_dir = os.path.join(args.base_dir, "vggt_omega")
    ply_path = os.path.join(vggt_dir, "vggt_omega_10M.ply")
    cameras_path = os.path.join(vggt_dir, "cameras.json")
    
    print(f"Loading point cloud for {args.base_dir}...")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    pts_xyzs = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    print("Loading cameras...")
    with open(cameras_path, 'r') as f:
        cam_data = json.load(f)
        
    image_names = cam_data['image_names']
    extrinsics = np.array(cam_data['extrinsics'])
    intrinsics = np.array(cam_data['intrinsics'])
    
    print(f"Dispatching {len(image_names)} images to parallel workers...")
    
    # Process all images concurrently across all CPU cores
    # Using backend='threading' since numpy matrix ops release the GIL
    # Restricted to n_jobs=16 to prevent Out Of Memory (OOM) errors from creating too many large intermediate arrays
    results = Parallel(n_jobs=16, backend="threading")(
        delayed(process_single_image)(
            idx, img_name, extrinsics[idx], intrinsics[idx], pts_xyzs, args.depths_dir
        ) for idx, img_name in enumerate(image_names)
    )
    
    depth_params = {k: v for k, v in results}
    all_scales = np.array([v["scale"] for v in depth_params.values() if "scale" in v])
    if len(all_scales) > 0 and (all_scales > 0).sum():
        med_scale = float(np.median(all_scales[all_scales > 0]))
    else:
        med_scale = 0.0
    for k in depth_params:
        depth_params[k]["med_scale"] = med_scale
    
    with open(os.path.join(vggt_dir, "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)
    print(f"✅ depth_params.json saved in {vggt_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--depths_dir', required=True)
    args = parser.parse_args()
    compute_depth_params(args)
