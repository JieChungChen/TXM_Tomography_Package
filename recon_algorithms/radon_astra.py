import numpy as np
import astra
from utils import min_max_normalize


def recon_fbp_astra(vol_geom, proj_geom, sino_sp):
    sino_id = astra.data2d.create('-sino', proj_geom, sino_sp)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['option'] = { 'FilterType': 'hann' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    sino_0 = np.ones((sino_sp.shape[0],sino_sp.shape[-1]))
    s0_id = astra.data2d.create('-sino', proj_geom, sino_0)
    rec0_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = s0_id
    cfg['ReconstructionDataId'] = rec0_id
    cfg['option'] = { 'FilterType': 'hann' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon0 = astra.data2d.get(rec0_id)
    recon_n = recon/recon0

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(rec0_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(s0_id)
    return recon_n


def radon_transform_astra(image, num_projections=181):
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    image = np.asarray(image, dtype=np.float32)
    angles = np.linspace(0, np.pi, num=num_projections, endpoint=True, dtype=np.float32)

    vol_geom = astra.create_vol_geom(image.shape[0], image.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[1], angles)
    projector_id = astra.create_projector('strip', proj_geom, vol_geom)

    sino_id, sinogram = astra.create_sino(image, projector_id)

    astra.projector.delete(projector_id)
    astra.data2d.delete(sino_id)
    return sinogram


def recon_slice_from_sino(sino, angle_interval=1, norm=True):
    num_projections, img_size = sino.shape
    max_angle = np.pi * ((num_projections -1) * angle_interval / 180)
    projection_angles = np.linspace(0, max_angle, num_projections, endpoint=True)
    vol_geom  = astra.create_vol_geom(sino.shape[-1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[-1], projection_angles)
    recon = recon_fbp_astra(vol_geom, proj_geom, sino)

    if norm:
        recon = min_max_normalize(recon) * 255
        recon = np.round(recon).astype(np.uint8)
    return recon