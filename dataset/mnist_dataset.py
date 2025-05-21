# coding:UTF-8
# RuHe  2025/5/20 17:23
import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    r"""
    Nothing any special here. Just a simple dataset class for mnist images.
    Create a dataset class rather using torchvision to allow replacement with any other image dataset.
    """
    def __init__(self, split, im_path, im_ext='png'):
        r"""
        Init method for initializing the dataset properties.
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext:
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified and stacks them all up.
        :param im_path: path
        :return:
        """
        assert os.path.exists(im_path), f'images path {im_path} does not exist.'
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for f_name in glob.glob(os.path.join(im_path, d_name, f'*.{self.im_ext}')):
                ims.append(f_name)
                labels.append(int(d_name))
        print(f'Found {len(ims)} images for split {self.split}')
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        # Convert input to -1 to 1 range.
        im_tensor = 2 * im_tensor - 1
        return im_tensor
