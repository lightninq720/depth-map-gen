# Depth Map Generator

Generate colorized depth-map images from JPG and JPEG photos using Apple's
DepthPro2 model.

## Requirements

- Python with the packages listed in [`requirements.txt`](requirements.txt)
- A CUDA-capable NVIDIA GPU. The script runs the model on CUDA and uses
  `bfloat16`.

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment, then install the packages:

```bash
python -m pip install -r requirements.txt
```

## Usage

By default, the script reads images from `images/inputs` and saves results to
`images/output`, relative to `main.py`:

```bash
python main.py
```

Both directories can be overridden independently:

```bash
python main.py --in_dir "path/to/input/images"
python main.py --out_dir "path/to/output"
python main.py --in_dir "path/to/input/images" --out_dir "path/to/output"
```

Depth maps are colourized by default. Use `--no-colour` to save grayscale
depth maps instead, or `--colour` to explicitly enable colour:

```bash
python main.py --no-colour
python main.py --in_dir "path/to/input/images" --out_dir "path/to/output" --no-colour
```

The input directory must exist. The output directory is created automatically.
Supported input extensions are `.jpg` and `.jpeg` (case-insensitive). Each
result is saved as `<input-name>_depth.jpg`.

On the first run, the DepthPro2 model and focal-length weights are downloaded
from Hugging Face.