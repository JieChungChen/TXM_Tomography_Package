import numpy as np
import torch
import torch.nn.functional as F


def radon_transform_torch(image, theta, circle=False, fill_outside=True, sub_sample=1):
    """
    radon transform pytorch version based on skimage radon
    
    Parameters
    ----------
    image : ndarray
        Input image (img_size, img_size)
    theta : ndarray
        Projection angles (in radians).
    circle : boolean, optional
        The value will be zero outside the inscribed circle
    fill_outside: boolean, optional
        If True, the area outside the inscribed circle will be filled by the median 
    sub_sample : float, optional
        A ratio between [0, 1] to determine the subsample size, the radon transform will only consider the subsampled image.

    Returns
    -------
    radon_image : ndarray
        Sinogram of the input image (len(theta), img_size).
    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    
    if image.shape[0] != image.shape[1]:
        raise ValueError('The input image must be square')

    if sub_sample<=0 or sub_sample>1:
        raise ValueError('The subsample size out of range')

    fill_outside = False if not circle else fill_outside
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    size = image.shape[0]

    if circle:
        center, radius = size//2, size//2
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center)**2 + (y-center)**2)
        mask = dist_from_center <= radius
        out_med = float(np.median(image[~mask])) # extract the value outside of the inscribed circle
        image = image * mask # only remain the inscribed circle
        if fill_outside:
            image[~mask] = out_med

    out_size = int(size*sub_sample)
    radon_image = torch.zeros((len(theta), out_size)).type(dtype)
    image = torch.from_numpy(image)[None, None, ...].type(dtype)
    theta = torch.Tensor(theta)
    for i, angle in enumerate(theta):
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot_mat = torch.Tensor([[cos_a, sin_a, 0],
                                [-sin_a, cos_a, 0]]).type(dtype)
        grid = F.affine_grid(rot_mat[None, ...], image.size())
        rotated = F.grid_sample(image, grid).squeeze()
        if sub_sample<1:
            start_p, end_p = int(size//2-out_size//2), int(size//2+out_size//2)
            rotated = rotated[start_p:end_p, start_p:end_p]
        radon_image[i, :] = rotated.sum(0)
    return radon_image.detach().cpu().numpy()