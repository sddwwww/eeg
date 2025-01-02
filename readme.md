

This repository contains the official implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653).

## Dataset
download [brains_data](https://drive.google.com/file/d/1NCK3RvyaQ9jKn4aJHSlBq9LFzmRYbxfH/view?usp=sharing)

download [clipfinetune_model.pkl](https://drive.google.com/file/d/1oRTQY4sxftyfqOMcUZkSdnbz2AylK2ey/view?usp=drive_link)

## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
```



## Usage

eeg-to-3D:
```bash

# 1
python main.py --config configs/eeg.yaml  save_path=name

# 2
python main2.py --config configs/eeg.yaml  save_path=name

```

