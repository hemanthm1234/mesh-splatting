#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.triangle_model import BasicPointCloud
import torch
import torchvision.transforms as transforms
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import re

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    normal_map: np.array = None 
    depth_params: dict = None  
    depth_path: str = ""     
    

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def resize_to_multiple(tensor, multiple=28):
    B, C, H, W = tensor.shape
    new_H = (H // multiple) * multiple
    new_W = (W // multiple) * multiple
    return torch.nn.functional.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder):

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width  = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image      = Image.open(image_path)

        # figure out the key without extension, e.g. "00012"
        n_remove = len(extr.name.split('.')[-1]) + 1
        key_no_ext = extr.name[:-n_remove]

        # grab per-view depth params (now guaranteed to have med_scale)
        depth_params = None
        if depths_params is not None and key_no_ext in depths_params:
            depth_params = depths_params[key_no_ext]
        else:
            if depths_params is not None:
                print("\n", key, "not found in depths_params")

        # depth png path
        if os.path.isdir(depths_folder):
            depth_path = os.path.join(depths_folder, f"{key_no_ext}.png")
        else:
            depth_path = ""

        # normal map (unchanged)
        normal_dir  = images_folder.replace("images", "normals")
        os.makedirs(normal_dir, exist_ok=True)
        normal_path = os.path.join(normal_dir, image_name + ".png")
        normal = None
        if os.path.exists(normal_path):
            normal_image = Image.open(normal_path).convert("RGB")
            normal_np = np.array(normal_image).astype(np.float32) / 255.0
            normal = (normal_np * 2.0) - 1.0
       
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            normal_map=normal,
            depth_params=depth_params,
            depth_path=depth_path,
        )
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, aug=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # Load depth scale and offset information
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if os.path.exists(depth_params_file):
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
    depths_folder = os.path.join(path, "depth")

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=depths_folder,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if aug:
        print("Using augmented PCD")

    ply_str = "points3D_138views" if aug else "points3D"

    ply_path = os.path.join(path, f"sparse/0/{ply_str}.ply")
    bin_path = os.path.join(path, f"sparse/0/{ply_str}.bin")
    txt_path = os.path.join(path, f"sparse/0/{ply_str}.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def load_vggt_omega_pcd(source_path: str, vggt_ply_name: str = "vggt_omega_10M.ply") -> BasicPointCloud:
    """
    Loads the pre-generated VGGT Omega point cloud from the convention path:
        <source_path>/vggt_omega/<vggt_ply_name>
    Raises FileNotFoundError if the file does not exist.
    Returns a BasicPointCloud aligned to COLMAP world space.
    """
    vggt_ply_path = os.path.join(source_path, "vggt_omega", vggt_ply_name)
    if not os.path.exists(vggt_ply_path):
        raise FileNotFoundError(
            f"[VGGT Omega Init] PLY file not found at: '{vggt_ply_path}'.\n"
            f"Please ensure the file exists."
        )
    print(f"[VGGT Omega Init] Loading VGGT Omega point cloud from: {vggt_ply_path}")
    pcd = fetchPly(vggt_ply_path)
    print(f"[VGGT Omega Init] Loaded {pcd.points.shape[0]:,} points from VGGT Omega PLY.")
    return pcd

def align_vggt_to_colmap(
    vggt_pcd: BasicPointCloud,
    colmap_pcd: BasicPointCloud,
) -> BasicPointCloud:
    """
    [DEPRECATED - DO NOT USE]
    WARNING: Attempting to align VGGT point clouds to COLMAP coordinate space via PCA/ICP is utter nonsense.
    ALL VGGT models must be rendered and trained using native VGGT camera poses via '--vggt_mode full_pipeline'.
    Do NOT call this function under any circumstances.
    """
    raise RuntimeError(
        "[DEPRECATED] align_vggt_to_colmap() is invalid and must NEVER be called! "
        "VGGT datasets MUST ALWAYS be used with native VGGT cameras via '--vggt_mode full_pipeline'."
    )
    vggt_pts = np.asarray(vggt_pcd.points)
    colmap_pts = np.asarray(colmap_pcd.points)

    # 1. Translation
    vggt_centroid  = vggt_pts.mean(axis=0)
    colmap_centroid = colmap_pts.mean(axis=0)

    vggt_centered  = vggt_pts - vggt_centroid
    colmap_centered = colmap_pts - colmap_centroid

    # 2. Rotation (PCA)
    cov_vggt = np.cov(vggt_centered, rowvar=False)
    cov_colmap = np.cov(colmap_centered, rowvar=False)

    U_vggt, _, _ = np.linalg.svd(cov_vggt)
    U_colmap, _, _ = np.linalg.svd(cov_colmap)

    # Project to principal axes to compute skewness
    vggt_proj = vggt_centered @ U_vggt
    colmap_proj = colmap_centered @ U_colmap

    vggt_skew = np.mean(vggt_proj**3, axis=0)
    colmap_skew = np.mean(colmap_proj**3, axis=0)

    # Resolve sign ambiguity using 3rd moment (skewness)
    for i in range(3):
        # If skewness signs don't match, flip the eigenvector
        if np.sign(vggt_skew[i]) != np.sign(colmap_skew[i]) and abs(vggt_skew[i]) > 1e-8:
            U_vggt[:, i] *= -1

    # R maps VGGT axes to COLMAP axes
    R = U_colmap @ U_vggt.T
    
    # Ensure R is a proper rotation matrix (no reflection)
    if np.linalg.det(R) < 0:
        # Fallback reflection correction if skewness was too noisy on the smallest axis
        U_colmap[:, -1] *= -1
        R = U_colmap @ U_vggt.T

    vggt_rotated = vggt_centered @ R.T

    # 3. Scale
    vggt_rms  = np.sqrt((vggt_rotated ** 2).sum(axis=1).mean())
    colmap_rms = np.sqrt((colmap_centered ** 2).sum(axis=1).mean())

    if vggt_rms < 1e-8:
        scale_factor = 1.0
    else:
        scale_factor = colmap_rms / vggt_rms

    aligned_pts = vggt_rotated * scale_factor + colmap_centroid

    print(f"[VGGT Omega Init] Alignment: centroid shift={colmap_centroid - vggt_centroid}, "
          f"scale_factor={scale_factor:.4f}")
    print(f"[VGGT Omega Init] Applied Rotation Matrix:\n{R}")

    return BasicPointCloud(
        points=aligned_pts.astype(np.float32),
        colors=np.asarray(vggt_pcd.colors).astype(np.float32),
        normals=np.asarray(vggt_pcd.normals).astype(np.float32),
    )

def readVGGTSceneInfo(path, images, eval, llffhold=8, aug=False, vggt_ply_name="vggt_omega_10M.ply"):
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
        all_scales = np.array([depths_params[key]["scale"] for key in depths_params if "scale" in depths_params[key]])
        if len(all_scales) > 0 and (all_scales > 0).sum():
            med_scale = np.median(all_scales[all_scales > 0])
        else:
            med_scale = 0
        for key in depths_params:
            depths_params[key]["med_scale"] = med_scale
            
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
        
        cx = intr[0, 2]
        cy = intr[1, 2]
        
        pred_width = cx * 2.0
        pred_height = cy * 2.0
        
        FovX = focal2fov(fx, pred_width)
        FovY = focal2fov(fy, pred_height)
        
        n_remove = len(img_name.split('.')[-1]) + 1
        key_no_ext = img_name[:-n_remove]
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
    ply_path = os.path.join(vggt_dir, vggt_ply_name)
    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"[VGGT Omega Init] PLY file not found at: '{ply_path}'.\n"
            f"Please ensure the file exists."
        )
    print(f"[VGGT Omega Init] Loading VGGT Omega point cloud from: {ply_path}")
    pcd = fetchPly(ply_path)
    
    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "VGGT": readVGGTSceneInfo,
    "Blender" : readNerfSyntheticInfo
}