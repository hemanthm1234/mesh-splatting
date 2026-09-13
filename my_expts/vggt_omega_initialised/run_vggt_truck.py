import sys
import os
import torch
import numpy as np
from glob import glob
import json

sys.path.append("/data1/hemanth/vggt-omega")
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from visual_util import _images_to_rgb

# Load images
scene_dir = "/data1/hemanth/datasets/tandt/Truck"
image_dir = os.path.join(scene_dir, "images")
out_dir = os.path.join(scene_dir, "vggt_omega")
os.makedirs(out_dir, exist_ok=True)

image_paths = sorted(glob(os.path.join(image_dir, "*.*")))
# Let's filter out directories or non-images if any
image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Loading {len(image_paths)} images...")
# images will be shape (B, 3, H, W)
images = load_and_preprocess_images(image_paths, image_resolution=512).to("cuda")

checkpoint_path = "/data1/hemanth/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
print("Loading model...")
model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))

print("Running inference...")
with torch.inference_mode():
    predictions = model(images)

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

print("Exporting cameras...")
extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

extrinsics_np = extrinsics[0].cpu().numpy()
# Pad from [N, 3, 4] to [N, 4, 4]
N = extrinsics_np.shape[0]
extrinsics_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
extrinsics_4x4[:, :3, :4] = extrinsics_np
extrinsics_4x4[:, 3, 3] = 1.0
extrinsics_np = extrinsics_4x4

intrinsics_np = intrinsics[0].cpu().numpy()
depth_np = predictions["depth"][0].cpu().numpy()

print(f"Extracted {N} cameras.")

print("Processing point cloud...")
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

print(f"Exporting {len(vertices)} points to PLY...")
# Write PLY with nx, ny, nz
ply_path = os.path.join(out_dir, "vggt_omega_10M.ply")
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
    
    # Structure array for binary write
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

# Save to json
camera_data = {
    "image_names": [os.path.basename(p) for p in image_paths],
    "extrinsics": extrinsics_np.tolist(),
    "intrinsics": intrinsics_np.tolist()
}
with open(os.path.join(out_dir, "cameras.json"), 'w') as f:
    json.dump(camera_data, f)

print("Done!")
