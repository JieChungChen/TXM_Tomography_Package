import os, glob, olefile, struct
import numpy as np
from PIL import Image
from src.logic.utils import split_mosaic, norm_to_8bit


def read_txm_raw(filename: str, mode: str):
    """
    read Xradia TXM/TXRM/XRM raw data from OLE file (.txm, .txrm, .xrm)

    Parameters
    ----------
    filename : str
    mode : str
        'tomo' for tomography, 'mosaic' for mosaic, 'single' for single image

    Returns
    -------
    tuple
        For 'tomo': (images: np.ndarray, metadata: dict, thetas: np.ndarray, reference: np.ndarray or None)
        For 'mosaic': (images: np.ndarray, metadata: dict, reference: np.ndarray or None)
        For 'single': (image: np.ndarray, metadata: dict, reference: np.ndarray or None)
    """
    assert mode in ['tomo', 'mosaic', 'single'], 'invalid mode!'

    ole = olefile.OleFileIO(filename)
    n_img = len([entry for entry in ole.listdir() if entry[0] in ['ImageData1', 'ImageData2']])
    if mode == 'mosaic':
        n_img = None
    metadata = read_ole_metadata(ole, mode, n_img)

    if mode == 'tomo':
        images = np.empty((metadata["number_of_images"],
                            metadata["image_height"],
                            metadata["image_width"]),
                            dtype=_get_ole_data_type(metadata))

        for i, idx in enumerate(range(metadata["number_of_images"])):
            img_string = "ImageData{}/Image{}".format(int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
            images[i] = _read_ole_image(ole, img_string, metadata)
        ole.close()
        images = np.flip(images, axis=1)
        thetas = metadata['thetas'][:n_img]
        thetas = np.around(thetas, decimals=1)
        metadata.pop('thetas', None)

    if mode == 'mosaic':
        stream = ole.openstream("ImageData1/Image1")
        data = stream.read()
        data_type = _get_ole_data_type(metadata)
        data_type = data_type.newbyteorder('<')
        image = np.reshape(np.frombuffer(data, data_type), (metadata["image_height"],  metadata["image_width"]))
        ole.close()
        images = split_mosaic(image, metadata['mosaic_row'], metadata['mosaic_column'])
        images = np.flip(images, axis=1)

    if mode == 'single':
        stream = ole.openstream("ImageData1/Image1")
        data = stream.read()
        data_type = _get_ole_data_type(metadata)
        data_type = data_type.newbyteorder('<')
        image = np.reshape(np.frombuffer(data, data_type), (metadata["image_height"],  metadata["image_width"]))
        image = np.flip(image, axis=0)
        image = image[None, ...] 
        ole.close()

    reference = metadata['reference']
    if reference is not None :
        reference = np.flip(reference, axis=0)

    metadata.pop('reference', None)
    metadata.pop('reference_data_type', None)
    metadata.pop('data_type', None)

    if mode == 'tomo':
        return images, metadata, thetas, reference
    if mode == 'mosaic':
        return images, metadata, reference
    if mode == 'single':
        return image, metadata, reference


def read_multiple_txrm(filelist):
    """
    read multiple TXRM files and concatenate the images and angles

    Parameters
    ----------
    filelist : list of str
        list of TXRM file paths

    Returns
    -------
    tuple (images, thetas, reference, file_names)
        images : np.ndarray
            concatenated images of shape (N, H, W)
        thetas : np.ndarray
            concatenated angles of shape (N,)
        reference : np.ndarray
            reference image of shape (H, W) or None
        file_names : list of str
            list of file names corresponding to each image
    """
    all_images = []
    all_thetas = []
    file_names = []
    for f in filelist:
        images, _, thetas, ref = read_txm_raw(f, mode='tomo')
        if ref is not None:
            ref = ref.astype(np.float32)
            images = images.astype(np.float32)
        all_images.append(images)
        all_thetas.append(thetas)
        f = os.path.basename(f)
        file_names.extend([f] * len(images))

    images = np.concatenate(all_images, axis=0)
    thetas = np.concatenate(all_thetas)
    return images, thetas, ref, file_names


def load_tif_folder(folder):
    """
    load all tif images from a folder into a 3D numpy array

    Parameters
    ----------
    folder : str
    
    Returns
    -------
    np.ndarray
        3D numpy array of shape (N, H, W)
    """
    files = sorted(glob.glob(f"{folder}/*tif"))
    if len(files) == 0:
        return
    
    all_imgs = []
    for f in files:
        img_temp = Image.open(f)
        if img_temp.mode in ("RGB", "RGBA"):
            img_temp = img_temp.convert("L")
        all_imgs.append(img_temp) 
    all_imgs = np.array(all_imgs)
    return all_imgs


def load_ref(filename):
    """
    load reference image from file

    Parameters
    ----------
    filename : str
        .xml or .tif file

    Returns
    -------
    np.ndarray
        reference image of shape (H, W)
    """
    image_type = filename.split('.')[-1]
    if image_type=='xrm':
        ref_img, _, _ = read_txm_raw(filename, 'single')
        ref_img = ref_img.squeeze()
    else:
        ref_img = np.array(Image.open(filename))
    return ref_img


def save_tif(folder, sample_name, imgs, mode):
    imgs = imgs.copy()
    sample_name = os.path.splitext(sample_name)[0]
    if mode == 'global':
        imgs = imgs/imgs.max()
        imgs = (imgs*255).astype(np.uint8)
    elif mode == 'each':
        for i in range(len(imgs)):
            imgs[i] = norm_to_8bit(imgs[i])
        imgs = imgs.astype(np.uint8)

    for i in range(len(imgs)):  
        img_temp = imgs[i] 
        img_temp = Image.fromarray(img_temp)
        img_temp.save(f"{folder}/{sample_name}_{str(i+1).zfill(4)}.tif")


# ------- core logic of txm raw data decoding; don't modify unless you know what you are doing ------- #
def read_ole_metadata(ole, mode, n_img=None):
    """
    get metadata from OLE database

    Parameters
    ----------
    ole : OleFileIO instance
    mode : str
        'tomo' or 'mosaic'
    n_img : int, optional
        number of images, by default None

    Returns
    -------
    dict
        metadata dictionary
    """
    if n_img is not None:
        number_of_images = n_img
    else:
        number_of_images = _read_ole_value(ole, "ImageInfo/NoOfImages", "<I")

    metadata = {
        'number_of_images': number_of_images,
        'image_width': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'image_height': _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'exp_time': str(_read_ole_arr(ole, 'ImageInfo/ExpTimes', f'<{number_of_images}f')[0]) + ' sec',
        'data_type': _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'reference_file': _read_ole_value(ole, 'ImageInfo/referencefile', '<260s'),
        'reference_data_type': _read_ole_value(ole, 'referencedata/DataType', '<1I'),
                }
    
    if mode == 'tomo':
        metadata['thetas'] = _read_ole_arr(ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images))
    elif mode == 'mosaic':
        metadata['mosaic_column'] = _read_ole_value(ole, 'ImageInfo/MosiacColumns', '<I')
        metadata['mosaic_row'] = _read_ole_value(ole, 'ImageInfo/MosiacRows', '<I')
    
    ref_path = _read_ole_value(ole, 'ImageInfo/referencefile', '<260s')
    if ref_path is not None:
        ref_path = ref_path.strip(b'\x00').decode()
        metadata['reference_file'] = ref_path.split('\\')[-1]
    
    if ole.exists('ReferenceData/Image'):
        reference = _read_ole_image(ole, 'ReferenceData/Image', metadata, metadata['reference_data_type'], is_ref=True)
    else:
        reference = None
    metadata['reference'] = reference
    return metadata


def _get_ole_data_type(metadata, datatype=None):
    """
    get numpy data type from OLE metadata
    """
    if datatype is None:
        datatype = metadata["data_type"]
    if datatype == 10:
        return np.dtype(np.float32)
    elif datatype == 5:
        return np.dtype(np.uint16)
    else:
        raise Exception("Unsupported data type: %s" % str(datatype))


def _read_ole_struct(ole, label, struct_fmt):
    """
    read struct from OLE file by label
    """
    value = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        if label == 'ImageInfo/Angles':
            value = struct.unpack("<{0}f".format(len(data)//4), data)
        else:
            value = struct.unpack(struct_fmt, data)
    return value


def _read_ole_value(ole, label, struct_fmt):
    """
    read single value from OLE file by label
    """
    value = _read_ole_struct(ole, label, struct_fmt)
    if value is not None:
        value = value[0]
    return value


def _read_ole_arr(ole, label, struct_fmt):
    """
    read array from OLE file by label
    """
    arr = _read_ole_struct(ole, label, struct_fmt)
    if arr is not None:
        arr = np.array(arr)
    return arr


def _read_ole_image(ole, label, metadata, datatype=None, is_ref=False):
    """
    read image from OLE file by label
    """
    stream = ole.openstream(label)
    data = stream.read()
    data_type = _get_ole_data_type(metadata, datatype)
    data_type = data_type.newbyteorder('<')
    image = np.frombuffer(data, data_type)
    if is_ref:
        s = int(np.sqrt(len(image)))
        img_size = (s, s)
    else:
        img_size = (metadata["image_height"], metadata["image_width"])
    image = np.reshape(image, img_size)
    return image