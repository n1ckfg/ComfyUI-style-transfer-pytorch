# ComfyUI Style Transfer (Pytorch)

This is a ComfyUI node version of the [style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch) repository.

## Features
- Multi-scale (coarse-to-fine) stylization.
- Wasserstein-2 style loss.
- EMA averaging of iterates.
- Supports multiple style images (as a batch).

## Parameters
- **content**: The content image.
- **style**: The style image (can be a batch).
- **content_weight**: How much to keep the content features (default: 0.015).
- **style_weight**: How much to apply the style (default: 1.0).
- **tv_weight**: Smoothness weight (default: 2.0).
- **min_scale**: Starting resolution (default: 128).
- **end_scale**: Final resolution (default: 512).
- **iterations**: Iterations per scale (default: 500).
- **initial_iterations**: Iterations for the first scale (default: 1000).
- **step_size**: Learning rate (default: 0.02).
- **avg_decay**: EMA decay rate (default: 0.99).
- **init**: Initial image ("content", "gray", "uniform", "style_stats").
- **pooling**: VGG pooling mode ("max", "average", "l2").
- **optimizer**: "adam" or "lbfgs".
- **style_scale_fac**: Relative scale of style to content.
- **style_size**: Fixed scale for style (0 for automatic).

## Installation
Clone this repository into your `ComfyUI/custom_nodes` directory.
Dependencies (should be in ComfyUI already):
- torch
- torchvision
- numpy
- Pillow
