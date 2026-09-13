import os
import json
import numpy as np

def validate_scene(scene_dir):
    print(f"--- Validating Scene: {scene_dir} ---")
    vggt_dir = os.path.join(scene_dir, "vggt_omega")
    ply_path = os.path.join(vggt_dir, "vggt_omega_10M.ply")
    cameras_path = os.path.join(vggt_dir, "cameras.json")
    
    if not os.path.exists(ply_path):
        print(f"❌ Missing {ply_path}")
        return False
    if not os.path.exists(cameras_path):
        print(f"❌ Missing {cameras_path}")
        return False
        
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        fields = [p.name for p in vertices.properties]
        required = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        missing = [req for req in required if req not in fields]
        if missing:
            print(f"❌ PLY file missing fields: {missing}")
            return False
        else:
            print(f"✅ PLY file has all required fields: {fields}")
            print(f"✅ Total points: {len(vertices.data):,}")
    except ImportError:
        print("⚠️ plyfile module not found, skipping deep PLY validation. (pip install plyfile)")
    except Exception as e:
        print(f"❌ Error reading PLY file: {e}")
        return False
        
    try:
        with open(cameras_path, 'r') as f:
            cam_data = json.load(f)
        if not all(k in cam_data for k in ['image_names', 'extrinsics', 'intrinsics']):
            print(f"❌ cameras.json missing required keys.")
            return False
        num_cams = len(cam_data['image_names'])
        print(f"✅ cameras.json contains {num_cams} cameras.")
        
        ex = np.array(cam_data['extrinsics'])
        if ex.shape != (num_cams, 4, 4):
            print(f"❌ Extrinsics shape mismatch. Expected ({num_cams}, 4, 4), got {ex.shape}")
            return False
        
        intr = np.array(cam_data['intrinsics'])
        if intr.shape != (num_cams, 3, 3):
            print(f"❌ Intrinsics shape mismatch. Expected ({num_cams}, 3, 3), got {intr.shape}")
            return False
            
        print("✅ cameras.json matrix shapes are correct.")
    except Exception as e:
        print(f"❌ Error reading cameras.json: {e}")
        return False

    print("🎉 Validation passed!\n")
    return True

if __name__ == "__main__":
    validate_scene("/data1/hemanth/datasets/tandt/Truck")
