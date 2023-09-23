import math
import random
import os
from PIL import Image
import blobfile as bf
import torch.distributed as dist
import refile
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
# import nori2 as nori
import cv2


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    lq_paths=None,  # for lowlevel vision task
    small_size=None,
    ignore_seed=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    classes = None
    if not ignore_seed:
        seed = dist.get_rank()
        torch.random.manual_seed(seed)
    
    if 'nori' in data_dir:
        dataset = ImageNetDataset(
            data_dir,
            image_size,
            classes=class_cond,
            random_crop=random_crop,
            random_flip=random_flip,
        )
    else:
        classes = None
        all_files = _list_image_files_recursively(data_dir)
        if class_cond:
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        # all_files = _list_image_files_recursively(data_dir) 
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            random_crop=random_crop,
            random_flip=random_flip,
            lq_paths=lq_paths,
            small_size=small_size,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        lq_paths=None,
        classes=None,
        random_crop=False,
        random_flip=True,
        small_size=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.small_size = small_size
        self.local_images = image_paths
        self.lq_paths = lq_paths  # low quality image path
        self.local_classes = None if classes is None else classes
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)
    
    def load_img(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        return pil_image.convert("RGB")
    
    def __getitem__(self, idx):
        path = self.local_images[idx]
        pil_image = self.load_img(path)
        
        if self.lq_paths is not None:
            lq_path = os.path.join(self.lq_paths, path.split('/')[-1])
            pil_lq_image = self.load_img(lq_path)
        else:
            lq_path = ''

        if self.random_crop:
            if lq_path:
                arr, lq_arr = random_crop_arr(pil_image, self.resolution, lq_image=pil_lq_image, small_size=self.small_size)
            else:
                arr = random_crop_arr(pil_image, self.resolution)
        else:
            if lq_path:
                arr, lq_arr = center_crop_arr(pil_image, self.resolution, lq_image=pil_lq_image, small_size=self.small_size)
            else:
                arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            if lq_path:
                lq_arr = lq_arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        if lq_path:
            lq_arr = lq_arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if lq_path:
            out_dict["low_res"] = np.transpose(lq_arr, [2, 0, 1])
            
        return np.transpose(arr, [2, 0, 1]), out_dict


class ImageNetDataset(Dataset):
    def __init__(
        self,
        data_dir,
        resolution,
        classes=None,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip

        # self.nori_fetcher = nori.Fetcher()
        # self.nori_list = list()
        # with open(data_dir, 'r') as fid:
        #     for line in fid.readlines():
        #         split_str_arr = line.split()
        #         nori_id, target = split_str_arr[0], int(split_str_arr[1])
        #         self.nori_list.append((nori_id, target))
        
        self.nori_fetcher = None
        self.nori_list = data_dir
        self.decode_nori_list()

    def _check_nori_fetcher(self):
        if self.nori_fetcher is None:
            self.nori_fetcher = nori.Fetcher()
            
    def __len__(self):
        return len(self.samples)
    
    def decode_nori_list(self):
        self.samples = []
        with refile.smart_open(self.nori_list, 'r') as f:
            for line in f:
                nori_id, *remains = line.strip().split()
                if len(remains) > 0:
                    target = int(remains[0])
                else:
                    target = None
                self.samples.append((nori_id, target))

    def __getitem__(self, index):
        self._check_nori_fetcher()
        nori_id, target = self.samples[index]
        img_bytes = self.nori_fetcher.get(nori_id)
        try:
            pil_image = Image.open(BytesIO(img_bytes)).convert('RGB')
        except:
            pil_image = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(pil_image)

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.classes:
            out_dict["y"] = np.array(target, dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size, lq_image=None, small_size=None):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if lq_image is None:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    if lq_image is not None:
        lq_arr = np.array(lq_image)
        scale = image_size // small_size
        arr = arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        if lq_arr.shape == arr.shape:
            return arr, lq_arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        
        # else, special case: super-resolution
        crop_y //= scale
        crop_x //= scale
        lq_arr = lq_arr[crop_y : crop_y + small_size, crop_x : crop_x + small_size]
        return arr, lq_arr
    
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, lq_image=None, small_size=None):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if lq_image is None:
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    
    if lq_image is not None:
        lq_arr = np.array(lq_image)
        assert small_size is not None
        scale = image_size // small_size
        
        if lq_arr.shape == arr.shape:
            arr = arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
            lq_arr = lq_arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        else:
            # special case, e.g., super-resolution
            crop_y //= scale
            crop_x //= scale
            lq_arr = lq_arr[crop_y : crop_y + small_size, crop_x : crop_x + small_size]

            # method "//= with *=" alleviate pixel mismatch!!!
            crop_y *= scale
            crop_x *= scale
            arr = arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        return arr, lq_arr
    
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
