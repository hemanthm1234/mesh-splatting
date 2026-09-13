import os
import json
import numpy as np
from PIL import Image
import sys

def get_vggt_loader_code():
    return """
def readVGGTSceneInfo(path, images, eval, llffhold=8, aug=False):
    vggt_dir = os.path.join(path, "vggt_omega")
    cameras_file = os.path.join(vggt_dir, "cameras.json")
    with open(cameras_file, 'r') as f:
        cam_data = json.load(f)
        
    image_names = cam_data['image_names']
    extrinsics = np.array(cam_data['extrinsics'])
    intrinsics = np.array(cam_data['intrinsics'])
    
    depths_params = None
    depth_params_file = os.path.join(vggt_dir, "depth_params.json")
    if os.path.exists(depth_params_file):
        with open(depth_params_file, 'r') as f:
            depths_params = json.load(f)
            
    cam_infos = []
    images_folder = os.path.join(path, images)
    depths_folder = os.path.join(path, "depth")
    
    for idx, (img_name, ext, intr) in enumerate(zip(image_names, extrinsics, intrinsics)):
        R = np.transpose(ext[:3, :3])
        T = ext[:3, 3]
        
        fx = intr[0, 0]
        fy = intr[1, 1]
        
        image_path = os.path.join(images_folder, img_name)
        image = Image.open(image_path)
        width, height = image.size
        
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        
        key_no_ext = img_name.split('.')[0]
        depth_params = depths_params.get(key_no_ext) if depths_params else None
        depth_path = os.path.join(depths_folder, f"{key_no_ext}.png") if os.path.isdir(depths_folder) else ""
        
        normal_dir  = images_folder.replace("images", "normals")
        normal_path = os.path.join(normal_dir, key_no_ext + ".png")
        normal = None
        if os.path.exists(normal_path):
            normal_image = Image.open(normal_path).convert("RGB")
            normal_np = np.array(normal_image).astype(np.float32) / 255.0
            normal = (normal_np * 2.0) - 1.0

        cam_info = CameraInfo(
            uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=key_no_ext, width=width, height=height,
            normal_map=normal, depth_params=depth_params, depth_path=depth_path
        )
        cam_infos.append(cam_info)
        
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(vggt_dir, "vggt_omega_10M.ply")
    pcd = fetchPly(ply_path) if os.path.exists(ply_path) else None
    
    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)
"""

with open("/data1/hemanth/mesh-splatting/scene/dataset_readers.py", "r") as f:
    code = f.read()

import re
code = re.sub(
    r"sceneLoadTypeCallbacks = {", 
    get_vggt_loader_code().strip() + "\n\nsceneLoadTypeCallbacks = {", 
    code
)

code = re.sub(
    r'"Colmap": readColmapSceneInfo,',
    r'"Colmap": readColmapSceneInfo,\n    "VGGT": readVGGTSceneInfo,',
    code
)

with open("/data1/hemanth/mesh-splatting/scene/dataset_readers.py", "w") as f:
    f.write(code)

