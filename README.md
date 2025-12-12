# NSRRC TXM Tomography Core Algorithms and Models

同步輻射中心TLS BL01B, TXM Tomography相關演算法及模型整合包

## recon_algorithms Module

The `recon_algorithms` module contains implementations of various reconstruction algorithms. 

### Available Functions

#### 1. `mlem_recon(sinogram, ang_inter, iter_count, mask)`
- **Purpose**: Performs ML-EM reconstruction on a sinogram to reconstruct a 2D image.

#### 2. `recon_fbp_astra(sinogram, angle_interval=1, norm=True)`
- **Purpose**: Computes FBP to construct a 2D slice from a sinogram using Astra Toolbox.

#### 3. `radon_transform_astra(image, angles)`
- **Purpose**: Computes the Radon transform of an image using Astra Toolbox.

#### 4. `radon_transform_torch(image, theta, circle=False, fill_outside=True, sub_sample=1)`
- **Purpose**: Computes the Radon transform using PyTorch, based on scikit-image's implementation.

### Installation and Usage
- Install dependencies: `pip install torch astra numpy`
- Example:
  ```python
  import numpy as np
  from recon_algorithms import mlem_recon
  
  # Example sinogram and mask
  sinogram = np.random.rand(181, 512)
  mask = np.ones((512, 512))
  result = mlem_recon(sinogram, ang_inter=1.0, iter_count=20, mask=mask)