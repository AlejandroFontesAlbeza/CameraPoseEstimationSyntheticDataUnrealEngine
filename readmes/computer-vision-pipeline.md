# Computer Vision Pipeline: Training, Inference & Camera Pose Estimation

This module focuses on the computer vision and deep learning components of the project, building directly on top of the synthetic dataset generated using Unreal Engine.

In the previous stage, a fully automated pipeline was developed to generate images and perfectly aligned segmentation masks of a tennis court. This synthetic dataset serves as the foundation for training a segmentation model without relying on manual annotation.

The objective of this module is to demonstrate that a model trained exclusively on synthetic data can learn meaningful visual features and generalize to real-world scenarios. To achieve this, the pipeline covers three main stages:

Training a segmentation model using the generated dataset
Performing inference on unseen images
Extracting geometric information from the predictions to estimate camera pose

The final goal is not only to obtain accurate segmentation results, but to leverage those results for downstream tasks such as court understanding and camera calibration.

This README describes the full computer vision pipeline, from dataset consumption and model training to real-time inference and geometric reasoning.

<p align="center">
  <img src="../rsc/frame0019.png" width="45%" />
  <img src="../rsc/inference_example.png" width="45%" />
</p>

---

## Technical Overview


This module covers the computer vision pipeline that transforms the synthetic dataset into a functional system capable of understanding the geometry of a tennis court from visual input.

At a high level, the pipeline starts with the synthetic dataset generated in Unreal Engine, consisting of RGB images and their corresponding segmentation masks. These data are used to train a deep learning model for court line segmentation.

Once trained, the model is integrated into an inference pipeline where it processes unseen images—potentially real-world broadcast frames—to produce segmentation outputs. These predictions are then post-processed to extract structural information such as court lines and their intersections.

Finally, this geometric information is used to estimate the camera parameters through techniques such as Homography, enabling the alignment between the image plane and the real-world court. This step is essential for applications such as augmented reality overlays and camera tracking.

The diagram below illustrates the full pipeline. In this README, the focus is placed on the stages highlighted in red, corresponding to training, inference, and geometric reasoning.

<p align="center">
  <img src="../rsc/DiagramaFlujo2.png" width="80%" />
</p>

---

## Dataset and Problem Definition

The dataset used in this project consists of synthetic RGB images and their corresponding segmentation masks, generated using Unreal Engine as described in the previous module.

The core objective is not simply to segment the tennis court, but to extract reliable geometric information from the scene—specifically, the intersections of court lines, which define key reference points.

A direct approach to this problem could involve predicting keypoints (e.g., using heatmaps). However, this strategy was intentionally avoided. Predicting only discrete points would discard a significant amount of structural information present in the scene and make the system more sensitive to noise and occlusions.

Instead, the problem is formulated as a line segmentation task, where the model predicts the full geometry of the court lines at the pixel level. This provides several advantages:

- Preserves the full spatial structure of the court
- Enables more robust extraction of line intersections
- Improves stability under partial occlusions or prediction errors

By segmenting lines rather than directly predicting points, the system retains richer geometric information, which can later be processed to recover precise intersection points.

These intersection points will be used in subsequent stages of the pipeline for geometric reasoning and camera pose estimation.


| HeatMaps (Less Info) | Lines Segmentation (More Info) |
|----------|----------|
| <img src="../rsc/Homography_tennis_visualization.png" width="600" height="270"> | <img src="../rsc/Inference_UniqueColor.png" width="500"> |

---

## Model Arquitecture

The segmentation model used in this project is based on **U-Net**, a widely adopted architecture for semantic segmentation tasks.

<p align="center">
  <img src="../rsc/u-net-architecture.png" width="60%" />
</p>


U-Net was selected due to its ability to produce precise, pixel-level predictions while preserving spatial information. Its **encoder–decoder structure** allows the model to capture both high-level contextual features and fine-grained details, which is particularly important for accurately segmenting thin structures such as tennis court lines.

This makes it well-suited for the problem at hand, where the objective is not only to classify regions but to recover the exact geometry of line-based structures that will later be used for geometric reasoning.

Another advantage of U-Net is its robustness when working with relatively small datasets, which aligns with the iterative training strategy followed in this project.

*Note: This README focuses on the application of the model within the pipeline. A more detailed explanation of the U-Net architecture can be found in the following repository 👉 [Repo](https://github.com/AlejandroFontesAlbeza/U-Net-Image-Segmentation)
