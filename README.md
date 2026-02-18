# vibe-infer

GPU inference from scratch using WebGPU. No frameworks, no libraries. Just raw compute shaders doing matrix math.

Draw a digit, classify it on your GPU in real-time: **[Live Demo](https://vtemian.github.io/vibe-infer/)**

## What this is

A from-scratch implementation of neural network inference running entirely on the GPU via WebGPU compute shaders. A 2-layer MNIST classifier (784 -> 128 -> 10) that:

- Loads pre-trained weights as raw float32 binary
- Runs matrix multiplication, ReLU activation, and softmax — all as WGSL compute shaders
- Chains 4 GPU compute passes in a single command encoder
- Classifies handwritten digits at ~97.5% accuracy

## How it was built

Pair-programmed as a learning exercise. The entire session is documented here: **[Claude Code Session](https://claudebin.com/threads/jmdbMowNTz)**

Built incrementally across 8 lessons — from "get a GPU device" to "draw and classify digits in real-time." Every compute shader was written by hand to understand what `model.predict()` actually does under the hood.

## Files

- `index.html` — the entire app (drawing canvas, GPU setup, shaders, inference pipeline)
- `train_export.py` — trains a 2-layer model on MNIST and exports weights as raw binary
- `weights/` — pre-trained weight files (w1, b1, w2, b2) + test images
- `LESSONS.md` — the 8-lesson plan followed during the build

## Run locally

```
python -m http.server 8000
```

Open `http://localhost:8000` in Chrome (WebGPU required).

## Retrain weights

```
pip install torch torchvision
python train_export.py
```
