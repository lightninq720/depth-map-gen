from functools import partial

import matplotlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from depth_pro_2 import DepthPro2, load_rgb, upright
from depth_pro_2.model import to_normalized_tensor
from depth_pro_2.network.fov import FOVEstimator
from pathlib import Path


HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "images" / "inputs"
OUT_DIR = HERE / "images" / "output"
EXTS = {".jpg", ".jpeg"}

model = DepthPro2.from_pretrained("apple/DepthPro2-L", precision=torch.bfloat16) # Change to model you want (S, B, L, H)

weights = hf_hub_download("apple/DepthPro2-focal", "model.safetensors")
model.fov = FOVEstimator(torch.bfloat16)
model.fov.load_state_dict(load_file(weights))
model = model.to("cuda").eval()
transform = partial(to_normalized_tensor, device=torch.device("cuda"))



def gen_depth_im(depth, out_path="output.jpg", colour=True):
    depth = depth.detach().float().cpu().squeeze().numpy()

    inv = 1.0 / np.clip(depth, 1e-4, None)
    lo, hi = np.percentile(inv, [1, 99])
    inv = np.clip((inv - lo) / (hi - lo + 1e-8), 0, 1)

    if colour:
        rgb = (matplotlib.colormaps["turbo"](inv)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_path, quality=95)
    else:
        Image.fromarray((inv * 255).astype(np.uint8)).save(out_path, quality=95)


def get_depth_map(file, out_dir=OUT_DIR):
    file = Path(file)
    out_path = Path(out_dir) / f"{file.stem}_depth.jpg"


    image, f_px = load_rgb(str(file))
    with torch.no_grad():
        depth = model.infer(transform(image), f_px=f_px)["depth"]
    gen_depth_im(depth, out_path=out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in IN_DIR.iterdir() if f.suffix.lower() in EXTS)
    print(f"Found {len(files)} images")

    for file in files:
        try:
            get_depth_map(file)
        except Exception as e:
            print(f"failed on {file.name}: {e}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()