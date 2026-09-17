# Semantic Image Segmentation using DeepLabV3

Command-line image segmentation tool leveraging PyTorch and DeepLabV3 with a ResNet-50 backbone to perform pixel-level classification on images.

## Project Overview
This project provides an automated pipeline for multi-class semantic image segmentation. It processes raw input images, passes them through a pre-trained DeepLabV3 model, maps predicted pixel classes using the Pascal VOC color map, and exports formatted visual masks to an output directory.

## Features
* Pre-trained DeepLabV3 (ResNet-50) deep learning model integration.
* Command Line Interface (CLI) for customizable input/output path handling.
* Automatic Pascal VOC 21-class RGB color palette mapping.
* Visual mask output generation saved directly to disk.

## Technologies Used
* Python 3.10+
* PyTorch & Torchvision
* OpenCV (`opencv-python-headless`)
* NumPy & Pillow

## Steps to Install & Run

1. **Clone repository:**
   ```bash
   git clone https://github.com/Sampadaseth/Computer-vision-vityarthi