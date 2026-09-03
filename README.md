# Physically Embodied Gaussian Splatting

<div align="left" style="display: left; align-items: center; justify-content: center; gap: 20px;">
    <img src="static/logo.jpeg" alt="Embodied Gaussians Logo" width="400px">
</div>

[Project Page](https://embodied-gaussians.github.io/) | [Paper](https://openreview.net/pdf?id=AEq0onGrN2)

## Overview

Embodied Gaussians introduces a novel dual "Gaussian-Particle" representation that bridges the gap between physical simulation and visual perception for robotics. Our approach:

- 🎯 **Unifies** geometric, physical, and visual world representations
- 🔮 Enables **predictive simulation** of future states
- 🔄 Allows **online correction** from visual observations
- 🌐 Integrates with an **XPBD physics** system
- 🎨 Renders high-quality images through **3D Gaussian splatting**

## SUPER extension and selected H3 baseline

This fork adds a stereo surgical-tissue pipeline with tetrahedral XPBD,
AllTracker/depth trajectory correction, a post-trajectory RGB residual, and
causal online material identification. The currently selected reproducible
material model is the global H3 baseline:

- Reconstruction uses one non-overlapping H3 at the end of each legal 7:1
  training block.
- Future prediction estimates material parameters only in the first 80% and
  performs no visual or material update after frame 1152.
- The optimizer updates global distance stiffness and velocity damping while
  shape and volume remain fixed.
- Experimental 3-of-4 recovery, accumulated local stiffness, and Gaussian
  appearance learning are disabled in the selected runner.

Run the selected three-method evaluation with:

```bash
bash scripts/run_super_reproduce_old_h3_scale_only_three_way.sh \
  outputs/<new-output-directory>
```

The verified H3 results, equations, exact parameters, commit frames, and
reproduction contract are documented in
[H3刚度正式记录.md](H3刚度正式记录.md). The full development record is in
[PROGRESS.md](PROGRESS.md).

`data/`, `outputs/`, model checkpoints, and host-specific CUDA/VirtualGL
runtime files are intentionally excluded from Git. AllTracker and
FoundationStereo are pinned as Git submodules; their checkpoints must be
obtained according to the corresponding upstream project instructions.

## Demo
<div align="left" style="display: left; align-items: center; justify-content: center; gap: 20px;">
    <img src="static/embodied_demo.gif" alt="Embodied Gaussians Demo" width="640">
</div>

## Abstract

For robots to robustly understand and interact with the physical world, it is highly beneficial to have a comprehensive representation -- modelling geometry, physics, and visual observations -- that informs perception, planning, and control algorithms. We propose a novel dual "Gaussian-Particle" representation that models the physical world while (i) enabling predictive simulation of future states and (ii) allowing online correction from visual observations in a dynamic world.

Our representation comprises particles that capture the geometrical aspect of objects in the world and can be used alongside a particle-based physics system to anticipate physically plausible future states. Attached to these particles are 3D Gaussians that render images from any viewpoint through a splatting process thus capturing the visual state. By comparing the predicted and observed images, our approach generates "visual forces" that correct the particle positions while respecting known physical constraints.

By integrating predictive physical modeling with continuous visually-derived corrections, our unified representation reasons about the present and future while synchronizing with reality. We validate our approach on 2D and 3D tracking tasks as well as photometric reconstruction quality.

## Implementation Notes

This repository provides a reference implementation of Embodied Gaussians with some differences from the paper:

- 🔷 **Rigid Bodies Only**: Currently, this implementation only supports rigid body dynamics. The shape matching functionality described in the paper is not included.
- 🔨 **Simplified Physics**: Due to the rigid body constraint, the physics simulation is more straightforward but less flexible than the full implementation described in the paper.

## Getting Started

### Installation

1. First, install [pixi](https://pixi.sh/latest/#installation)

2. Then build the dependencies:
```bash
pixi r build
```

### Running the Demo

To run the included demo:
```bash
pixi r demo
```


## Scene Building

These scripts require Realsense cameras directly connected to your device. Note: Offline image processing is not currently supported.

### 1. Ground Plane Detection
First, detect the ground plane by running:
```bash
python scripts/find_ground.py temp/ground_plane.json --extrinsics scripts/example_extrinsics.json --visualize
```
You will be prompted to segment the ground in the interface. The script will then calculate the ground points and plane parameters.

<div align="left">
    <img src="static/ground_detection_example.png" alt="Ground Detection" width="320">
</div>

### 2. Generate Ground Gaussians
Convert the detected ground plane into Gaussian representations:
```bash
python scripts/build_body_from_pointcloud.py temp/ground_body.json --extrinsics scripts/example_extrinsics.json --points scripts/example_ground_plane.npy --visualize
```

### 3. Object Generation
Generate embodied Gaussian representations of objects using multiple viewpoints (more viewpoints yield better results):

1. Run the scene building script:
```bash
python scripts/build_simple_body.py objects/tblock.json \
    --extrinsics scripts/example_extrinsics.json \
    --ground scripts/example_ground_plane.json \
    --visualize
```

2. For each camera viewpoint:
   - A segmentation GUI will appear
   - Click to select the target object
   - Press `Escape` when satisfied with the selection
   - Repeat for all viewpoints

3. The script will generate a JSON file containing both particle and Gaussian representations of your object.

<div align="left">
    <img src="static/scene_builder_segmentation.png" alt="Segmentation Window" width="640">
</div>

You can visualize the object with
```bash
python scripts/visualize_object.py scripts/example_object.json
```


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{
    abouchakra-embodiedgaussians,
    title={Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics},
    author={Jad Abou-Chakra and Krishan Rana and Feras Dayoub and Niko Suenderhauf},
    booktitle={8th Annual Conference on Robot Learning},
    year={2024},
    url={https://openreview.net/forum?id=AEq0onGrN2}
}
```

## More Information

For videos and additional information, visit our [project page](https://embodied-gaussians.github.io/).

## Disclaimer
This software is provided as a research prototype and is not production-quality software. Please note that the code may contain missing features, bugs and errors. RAI Institute does not offer maintenance or support for this software.
