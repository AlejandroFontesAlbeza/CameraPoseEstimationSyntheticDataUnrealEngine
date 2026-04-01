## Synthetic Tennis Vision Pipeline (Unreal Engine + Computer Vision)
End-to-end pipeline demonstrating how synthetic data generated in Unreal Engine can be used to train Neural Networks arquitectures to solve a real problem. For this project we applied this workflow for a tennis match segmenting the court lines for homography estimation, and camera pose reconstruction for AR and broadcast applications.

<p align="center">
  <img src="rsc/inference_output.gif" width="45%" />
  <img src="rsc/inference_output.gif" width="45%" />
  <img src="rsc/inference_output.gif" width="45%" />
</p>

---

### Introduction

Camera calibration in sports broadcast and virtual production is traditionally a manual, time-consuming and error-prone process. Accurately estimating camera parameters is essential for rendering augmented reality (AR) graphics aligned with the real world.

This project explores an alternative approach based on synthetic data generation and automatic annotation.

The goal is to demonstrate that:

- Synthetic data generated in a 3D engine can replace manual dataset labeling
- Computer vision models trained on synthetic data can generalize to real-world scenarios
- Camera calibration can be automated using learned visual features

To achieve this, the project implements a full pipeline that:

- Generates synthetic tennis court data using Unreal Engine 5
- Automatically produces perfectly labeled segmentation masks
- Trains a neural network for court line segmentation
- Computes homography and estimates camera pose
- Integrates results back into Unreal Engine for AR visualization

*This is not intended as a production-ready system, but as a technical demonstration of synthetic data pipelines applied to real-world broadcast problems.*

---
### Repository Structure

```
.
├── readmes/
│   ├── data-generation-unreal-engine.md
│   └── computer-vision-pipeline.md
├── src/
├── models/
├── outputs/
└── README.md
```
---
### Documentation

To keep the main README concise and readable, the pipeline is documented in two dedicated sections:

- **Synthetic Data Generation (Unreal Engine)**
→ [README.md](readmes/data-generation-unreal-engine.md)
Covers scene setup, rendering pipeline, and automatic annotation.

- **Camera Pose Estimation Pipeline**
→ [README.md](readmes/computer-vision-pipeline.md)

*Although both parts were developed iteratively, the documentation is structured in a logical order: data generation → model training → inference pipeline.*

---
### Setup

**Clone the repository**
```bash
git clone https://github.com/AlejandroFontesAlbeza/synthetic-tennis-vision.git
cd synthetic-tennis-vision
```

