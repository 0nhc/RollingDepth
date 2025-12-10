Here is the README formatted exactly like the example you provided, tailored for your RollingDepth Flask server implementation.

# RollingDepth on MSR Server

This repository provides a simple Flask-based HTTP API wrapper around the original RollingDepth demo. Instead of running scripts locally, you can deploy the model as a network service and call it from any machine.

## What it does

The server exposes a single HTTP endpoint that performs video depth estimation using the RollingDepth framework.

### Endpoint

`POST /process_video`

### Input (multipart/form-data)

**Required files:**

  * `video` - RGB video file (e.g., .mp4, .avi, .mov)

**Optional parameters:**

  * `preset` - Inference configuration preset. Options: `fast`, `fast1024`, `full`, `paper`. (default: `paper`)

### Processing Pipeline

The server performs the following steps:

1.  **Video Upload & Validation**

      - Receives and saves the video file to a temporary unique directory
      - Validates file integrity

2.  **Configuration & Pre-processing**

      - Applies the selected preset configuration (Resolution, Dilation rates, Refinement steps)
      - Resizes input frames according to the preset resolution (e.g., 768px or 1024px)

3.  **Rolling Depth Estimation**

      - **Latent Video Diffusion:** Encodes video into VAE latent space
      - **Rolling Window Denoising:** Predicts depth using a diffusion model with temporal attention across rolling windows
      - **Coarse-to-Fine Refinement:** (If using `paper` or `full` preset) Iteratively refines depth details

4.  **Output Formatting**

      - Decodes prediction back to pixel space
      - Formats the output as a NumPy array `[Frames, Height, Width]`
      - Cleans up temporary files from the server

### Output

Returns a .npy file as an attachment containing the raw depth values:

  * Shape: `(N, H, W)` where N is the frame count
  * Data type: Floating point depth values

### Use Cases

  * **Video Depth Estimation** - Generate consistent depth maps for video sequences
  * **3D Reconstruction** - Use depth maps for downstream 3D scene reconstruction
  * **Visual Effects** - Depth-aware video editing and compositing

## Installation

Follow the following steps to install RollingDepth on the MSR server.

```sh
git clone https://github.com/prs-eth/RollingDepth.git
cd RollingDepth
micromamba create -n rollingdepth python=3.10
micromamba activate rollingdepth
micromamba install conda-forge::cuda-toolkit=12.1 conda-forge::ffmpeg
pip install -r requirements.txt
bash script/install_diffusers_dev.sh
```

## Quick Start

```sh
# Run these commands on the Lamb or Sheep server.
python server.py
```

```sh
# Run this script on your own laptop.
python client.py
```

# 🛹 RollingDepth: Video Depth without Video Models

**CVPR 2025**

[![Website](doc/badges/badge-website.svg)](https://rollingdepth.github.io)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/prs-eth/rollingdepth)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](http://arxiv.org/abs/2411.19189)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/prs-eth/rollingdepth-v1-0)

This repository represents the official implementation of the paper titled "Video Depth without Video Models".

[Bingxin Ke](http://www.kebingxin.com/)<sup>1</sup>,
[Dominik Narnhofer](https://scholar.google.com/citations?user=tFx8AhkAAAAJ&hl=en)<sup>1</sup>,
[Shengyu Huang](https://shengyuh.github.io/)<sup>1</sup>,
[Lei Ke](https://www.kelei.site/)<sup>2</sup>,
[Torben Peters](https://scholar.google.com/citations?user=F2C3I9EAAAAJ&hl=de)<sup>1</sup>,
[Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)<sup>2</sup>,
[Anton Obukhov](https://www.obukhov.ai/)<sup>1</sup>,
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en)<sup>1</sup>


<sup>1</sup>ETH Zurich, 
<sup>2</sup>Carnegie Mellon University



## 📢 News
2025-02-26: Paper is accepted to CVPR 2025. <br>
2024-12-02: Paper is on arXiv.<br>
2024-11-28: Inference code is released.<br>



## 🛠️ Setup
The inference code was tested on: Debian 12, Python 3.12.7 (venv), CUDA 12.4, GeForce RTX 3090

### 📦 Repository
```bash
git clone https://github.com/prs-eth/RollingDepth.git
cd RollingDepth
```

### 🐍 Python environment
Create python environment:
```bash
# with venv
python -m venv venv/rollingdepth
source venv/rollingdepth/bin/activate

# or with conda
conda create --name rollingdepth python=3.12
conda activate rollingdepth
```

### 💻 Dependencies
Install dependicies: 
```bash
pip install -r requirements.txt
bash script/install_diffusers_dev.sh  # Install modified diffusers with cross-frame self-attention
```
We use [pyav](https://github.com/PyAV-Org/PyAV) for video I/O, which relies on [ffmpeg](https://www.ffmpeg.org/) (tested with version 5.1.6-0+deb12u1).

To see the modification in diffusers, search for comments "Modified in RollingDepth".

## 🏃 Test on your videos
All scripts are designed to run from the project root directory.

### 📷 Prepare input videos
1. Use sample videos:
    ```bash
    bash script/download_sample_data.sh
    ```
    These example videos are to be used only as debug/demo input together with the code and should not be distributed outside of the repo.

1. Or place your videos in a directory, for example, under `data/samples`.

### 🚀 Run with presets
```bash
python run_video.py \
    -i data/samples \
    -o output/samples_fast \
    -p fast \
    --verbose
```
- `-p` or `--preset`: preset options
    - `fast` for **fast inference**, with dilations [1, 25] (flexible), fp16, without refinement, at max. resolution 768.
    - `fast1024` for **fast inference at resolution 1024**
    - `full` for **better details**, with dilations [1, 10, 25] (flexible), fp16, with 10 refinement steps, at max. resolution 1024.
    - `paper` for **reproducing paper numbers**, with (fixed) dilations [1, 10, 25], fp32, with 10 refinement steps, at max. resolution 768.
- `-i` or `--input-video`: path to input data, can be a single video file, a text file with video paths, or a directory of videos.
- `-o` or `--output-dir`: output directory.

#### Passing these inference arguments will overwrite the preset settings:
- `--res` or `--processing-resolution`: the maximum resolution (in pixels) at which image processing will be performed. If set to 0, processes at the original input image resolution.
- `--refine-step`: number of refinement iterations to improve accuracy and details. Set to 0 to disable refinement.
- `--snip-len` or `--snippet-lengths`: number of frames to analyze in each snippet.
- `-d` or `--dilations`: spacing between frames for temporal analysis, could have multiple values e.g. `-d 1 10 25`.

#### Clip sub-sequence to be processed:
- `--from` or `--start-frame`: the starting frame index for processing, default to 0.
- `--frames` or `--frame-count`: number of frames to process after the starting frame. Set to 0 (default) to process until the end of the video.

#### Output settings
- `--fps` or `--output-fps`: frame rate (FPS) for the output video. Set to 0 (default) to match the input video's frame rate.
- `--restore-res` or `--restore-resolution`: whether to restore the output to the original input resolution after processing, Default: False.
- `--save-sbs` or `--save-side-by-side`: whether to save side-by-side videos of RGB and colored depth. Default: True.
- `--save-npy`: whether to save depth maps as .npy files. Default: True.
- `--save-snippets`: whether to save initial snippets. Default: False

#### Other argumenets
- Please run `python run_video.py --help` to get details for other arguments.
- For low GPU memory footage:  pass ` --max-vae-bs 1 --unload-snippet true` and use a smaller resolution, e.g. `--res 512`

## ⬇ Checkpoint cache
By default, the [checkpoint](https://huggingface.co/prs-eth/rollingdepth-v1-0) is stored in the Hugging Face cache. The HF_HOME environment variable defines its location and can be overridden, e.g.:

```
export HF_HOME=$(pwd)/cache
```

Alternatively, use the following script to download the checkpoint weights locally and specify checkpoint path by `-c checkpoint/rollingdepth-v1-0 `

```bash
bash script/download_weight.sh
```


## 🦿 Evaluation on test datasets
Coming soon


## 🎓 Citation
```bibtex
@InProceedings{ke2024rollingdepth,
    title={Video Depth without Video Models}, 
    author={Bingxin Ke and Dominik Narnhofer and Shengyu Huang and Lei Ke and Torben Peters and Katerina Fragkiadaki and Anton Obukhov and Konrad Schindler},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```


## 🙏 Acknowledgments
We thank Yue Pan, Shuchang Liu, Nando Metzger, and Nikolai Kalischek for fruitful discussions. 
 
We are grateful to [redmond.ai](https://redmond.ai/) (robin@redmond.ai) for providing GPU resources.

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The model is licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.
