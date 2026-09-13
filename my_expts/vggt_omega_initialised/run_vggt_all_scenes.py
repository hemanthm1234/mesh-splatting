import sys
import os
import torch
import numpy as np
from glob import glob
import json
import time

sys.path.append("/data1/hemanth/vggt-omega")
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from visual_util import _images_to_rgb

def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )

def process_scene(scene_name, model):
    scene_dir = os.path.join("/data1/hemanth/datasets/tandt", scene_name)
    image_dir = os.path.join(scene_dir, "images")
    out_dir = os.path.join(scene_dir, "vggt_omega")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*40}")
    print(f"🚀 Processing Scene: {scene_name}")
    print(f"{'='*40}")

    image_paths = sorted(glob(os.path.join(image_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Loading {len(image_paths)} images...")
    t0 = time.time()
    images = load_and_preprocess_images(image_paths, image_resolution=512).to("cuda")

    print(f"Running inference (this may take a minute)...")
    with torch.inference_mode():
        predictions = model(images)
    print(f"✅ Inference complete in {time.time() - t0:.1f} seconds")

    print("Extracting cameras...")
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    extrinsics_np = extrinsics[0].cpu().numpy()
    N = extrinsics_np.shape[0]
    extrinsics_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
    extrinsics_4x4[:, :3, :4] = extrinsics_np
    extrinsics_4x4[:, 3, 3] = 1.0
    extrinsics_np = extrinsics_4x4
    intrinsics_np = intrinsics[0].cpu().numpy()
    depth_np = predictions["depth"][0].cpu().numpy()

    print(f"Extracted {N} cameras.")

    print("Unprojecting point cloud...")
    t1 = time.time()
    points = unproject_depth_map_to_point_map(depth_np, extrinsics_np, intrinsics_np)
    conf = predictions["depth_conf"][0].cpu().numpy()
    images_rgb = predictions["images"][0].cpu().numpy()

    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images_rgb).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    conf_thres = 20.0
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    vertices = vertices[mask]
    colors = colors[mask]
    normals = np.zeros_like(vertices)
    print(f"✅ Point cloud filtering complete in {time.time() - t1:.1f} seconds")

    ply_path = os.path.join(out_dir, "vggt_omega_10M.ply")
    print(f"Exporting {len(vertices):,} points to {ply_path} ...")
    
    with open(ply_path, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        vertex_data = np.empty(len(vertices), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex_data['x'] = vertices[:, 0]
        vertex_data['y'] = vertices[:, 1]
        vertex_data['z'] = vertices[:, 2]
        vertex_data['nx'] = normals[:, 0]
        vertex_data['ny'] = normals[:, 1]
        vertex_data['nz'] = normals[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]
        f.write(vertex_data.tobytes())
    print("✅ PLY file saved.")

    camera_data = {
        "image_names": [os.path.basename(p) for p in image_paths],
        "extrinsics": extrinsics_np.tolist(),
        "intrinsics": intrinsics_np.tolist()
    }
    with open(os.path.join(out_dir, "cameras.json"), 'w') as f:
        json.dump(camera_data, f)
    print("✅ cameras.json saved.")

if __name__ == "__main__":
    scenes = ["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
    # scenes = ["Courthouse"]
    
    checkpoint_path = "/data1/hemanth/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
    print(f"Loading VGGT Omega model from {checkpoint_path}...")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    
    for scene in scenes:
        # Note: Meetingroom is directory "Meeting_room"
        if scene == "Meetingroom":
            scene_name = "Meeting_room"
        else:
            scene_name = scene
            
        try:
            process_scene(scene_name, model)
        except Exception as e:
            print(f"❌ Error processing scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
