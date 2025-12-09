import numpy as np
import astra
from utils import min_max_normalize


def radon_transform_astra(image: np.ndarray, num_projections=181): 
    """
    derive the sinogram of an image using radon transform with ASTRA toolbox.
    
    Parameters
    ----------
    image : ndarray
        Input image (img_size, img_size)
    num_projections : int
        Number of projection angles. Default is 181.(±90 degrees with 1 degree interval)

    Returns
    -------
    sinogram : ndarray
        Sinogram of the input image (num_projections, img_size).
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    if image.shape[0] != image.shape[1]:
        raise ValueError("Input image must be square.")

    image = np.asarray(image, dtype=np.float32)
    angles = np.linspace(0, np.pi, num=num_projections, endpoint=True, dtype=np.float32)

    vol_geom = astra.create_vol_geom(image.shape[0], image.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[1], angles)
    projector_id = astra.create_projector('strip', proj_geom, vol_geom)

    sino_id, sinogram = astra.create_sino(image, projector_id)

    astra.projector.delete(projector_id)
    astra.data2d.delete(sino_id)
    return sinogram


def recon_fbp_astra(sino: np.ndarray, angle_interval=1, norm=True): 
    """
    reconstruct a slice from its sinogram using FBP with ASTRA toolbox.
    
    Parameters
    ----------
    sino : ndarray
        Input sinogram (num_projections, img_size)
    angle_interval : int
        Angle interval in degrees between consecutive projections.
    norm : bool
        If True, normalize the reconstructed image to [0, 255] and convert to uint8.

    Returns
    -------
    recon : ndarray
        Reconstructed image (img_size, img_size)
    """
    if sino.ndim != 2:
        raise ValueError("Input sinogram must be a 2D array.")
    
    num_projections, img_size = sino.shape
    max_angle = np.pi * ((num_projections-1) * angle_interval / 180)
    projection_angles = np.linspace(0, max_angle, num_projections, endpoint=True)
    vol_geom  = astra.create_vol_geom(img_size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, img_size, projection_angles)

    # astra fbp reconstruction
    # ------------------------
    sino_id = astra.data2d.create('-sino', proj_geom, sino)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['option'] = { 'FilterType': 'hann' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    sino_0 = np.ones_like(sino)
    s0_id = astra.data2d.create('-sino', proj_geom, sino_0)
    rec0_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = s0_id
    cfg['ReconstructionDataId'] = rec0_id
    cfg['option'] = { 'FilterType': 'hann' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon0 = astra.data2d.get(rec0_id)
    recon = recon/recon0

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(rec0_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(s0_id)
    # ------------------------

    if norm:
        recon = min_max_normalize(recon) * 255
        recon = np.round(recon).astype(np.uint8)
    return recon