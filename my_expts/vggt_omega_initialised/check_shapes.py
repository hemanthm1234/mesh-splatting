import sys, os, torch
sys.path.append("/data1/hemanth/vggt-omega")
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

scene_dir = "/data1/hemanth/datasets/tandt/Truck"
image_dir = os.path.join(scene_dir, "images")
from glob import glob
image_paths = sorted(glob(os.path.join(image_dir, "*.*")))[:2] # just 2 images
images = load_and_preprocess_images(image_paths, image_resolution=512).to("cuda")
print(f"images shape: {images.shape}")

model = VGGTOmega().to("cuda").eval()
with torch.inference_mode():
    predictions = model(images)

for k, v in predictions.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")
