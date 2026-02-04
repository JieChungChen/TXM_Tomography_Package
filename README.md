# NSRRC TXM Tomography Core Algorithms and Models

同步輻射中心TLS BL01B, TXM Tomography相關演算法及模型整合包

## Overview

This repository provides tools and algorithms for TXM (Transmission X-ray Microscopy) tomography image processing. It includes a PyQt5-based GUI for managing and processing TXM data, as well as core algorithms for reconstruction and alignment.

## Features

### `app.py` - TXM Toolbox GUI
The `app.py` script provides a comprehensive GUI for TXM data processing. Key features include:
- **File Loading**:
  - Load `.txrm` or `.tif` files for tomography data.
  - Load `.xrm` or `.tif` files for mosaic data.
  - Load and merge multiple `.txrm` files by manually resolving duplicate angles.
- **Image Manipulation**:
- - Manually choose reference images.
  - Vertical flipping.
  - Y-axis shifting.
  - Contrast adjustment.
- **Reconstruction**:
  - Perform FBP (Filtered Back Projection) reconstruction with customizable resolution and angle intervals (support gpu acceleration with astra-toolbox).
- **Mosaic Stitching**:
  - Stitch mosaic images.
- **AI Tools**:
  - Background correction using AI-based reference removal.
- **Image Saving**:
  - Save processed 8-bits `.tif` images in raw or normalized contrast.

### `ref_remover_ddpm/txm_image_correction_gui.py` - Background Correction GUI
This GUI focuses on background correction using DDPM (Denoising Diffusion Probabilistic Models). Features include:
- GPU status display.
- Progress tracking with ETA.
- Image stack browsing and preview.

### Core Modules
#### `recon_algorithms`
- ML-EM and FBP reconstruction algorithms.
- Radon transform implementations using Astra Toolbox and PyTorch.

#### `sinogram_alignment`
- AI Tools for sinogram alignment and preprocessing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JieChungChen/TXM_Tomography_Package.git
   cd TXM_Tomography_Package
   ```
   
2. Install additional modules for specific features:
   - Astra Toolbox for GPU-accelerated FBP:
     ```bash
     conda install -c astra-toolbox -c nvidia astra-toolbox==2.4.0
     ```
    - PyTorch for AI-based tools:
      ```bash
      pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
      ```
      
3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

4. Download pre-trained models for AI background correction:  
  [OneDrive Link](https://nsrrcmail-my.sharepoint.com/:f:/g/personal/chen_jc_nsrrc_org_tw/IgB7usYZ2jylSpkhUTP-UG6aAZiWJLVYOUwR-umvreAxVEU?e=KmsNVi)  
  put the downloaded `checkpoints` folder into `ref_remover` directory.


## Usage

### Running the TXM Toolbox GUI
1. Run the `app.py` script:
   ```bash
   python app.py
   ```

2. Use the GUI to load, process, and save TXM data.

## License
This project is licensed under the MIT License.
