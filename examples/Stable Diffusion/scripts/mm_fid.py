from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from mmgeneration.mmgen.core.evaluation.metrics import FID
import numpy as np
import torchvision.transforms as TF
from PIL import Image
import torch
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
import pathlib


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, size=None):
        self.files = files
        self.transforms = transforms
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img, self.size)
        return img

def transform_fun(image, size):
    import albumentations
    val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)])
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))                   
    w, h = image.size
    if w < h:
        w_ = size 
        h_ = int(h * w_/w)
    else:
        h_ = size
        w_ = int(w * h_/h)
    # if w_ == size and h_ == size:
    #     image = np.array(image).astype(np.uint8)
    #     fun = TF.ToTensor()
    #     image = fun(image)
    #     return image
    image = image.resize((w_, h_))
    image = np.array(image).astype(np.uint8)
    image = val_preprocessor(image=image)['image']
    image = np.array(image).astype(np.uint8)
    # image = image[None,:,:,:]
    fun = TF.ToTensor()
    image = fun(image)
    return image


def main():
    fid_calculator = FID(num_images=1000, image_shape=512)
    fid_calculator.prepare()
    args = parser.parse_args()

    if args.path[0].endswith('.npz'):
        with np.load(args.path[0]) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path1 = pathlib.Path(args.path[0])
        files1 = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path1.glob('*.{}'.format(ext))])

        dataset1 = ImagePathDataset(files1, transforms=transform_fun, size=512)

        dataloader1 = torch.utils.data.DataLoader(dataset1,
                                                batch_size=64,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=4) 

        for batch in tqdm(dataloader1):
            fid_calculator.feed(batch, 'real')
    
    if args.path[1].endswith('.npz'):
        with np.load(args.path[1]) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path2 = pathlib.Path(args.path[1])
        files2 = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path2.glob('*.{}'.format(ext))])

        dataset2 = ImagePathDataset(files2, transforms=transform_fun, size=512)

        dataloader2 = torch.utils.data.DataLoader(dataset2,
                                                batch_size=64,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=4) 

        for batch in tqdm(dataloader2):
            fid_calculator.feed(batch, 'fakes')
    import pdb
    pdb.set_trace()
    fid, _, _ = fid_calculator.summary()

