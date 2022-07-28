from tabnanny import check
from torch.utils.data import Dataset
import json
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import numbers
import cv2
  
class MonoConEvalDataset(Dataset):
    def __init__(self, imgs_dir, labels_file, transform=None, divisor=32):
        self.transform = transform
        self.divisor = divisor
        _f = open(labels_file)
        self.data = json.load(_f)
        _f.close()
        self.imgs_dir = imgs_dir
        self.imgs_list, self.metas_list = self.__make_files_list()

    def impad(self, img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant'):
        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            h_pad = int((shape[1] - img.shape[1])/2)
            w_pad = int((shape[0] - img.shape[0])/2)
            padding = (h_pad, w_pad)
        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')
        padding = (padding[0], padding[1], padding[0], padding[1])

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        img = np.array(img) 
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)
        return img

    def __make_files_list(self):
        imgs_list = []
        metas_list = []
        for img_meta in self.data["images"]:
            img_name = img_meta["file_name"].split("/")[-1]
            img_path = Path(f"{self.imgs_dir}/{img_name}")
            if img_path.is_file():
                imgs_list.append(img_path)
                metas_list.append(img_meta)
        return imgs_list, metas_list

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        # Image
        img_path = self.imgs_list[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        pad_h = int(np.ceil(height / self.divisor)) * self.divisor
        pad_w = int(np.ceil(width / self.divisor)) * self.divisor
        img = self.impad(img, shape=(pad_h, pad_w))
        pad_img_size = torch.IntTensor(img.shape[:2])

        if self.transform:
            img = self.transform(img)
        
        # Metadata

        img_meta = {
            'filename': str(img_path),
            "cam_intrinsic": torch.FloatTensor(self.metas_list[idx]['cam_intrinsic']),
            'scale_factor': 1.0, 
            'pad_shape': pad_img_size,
            'img_shape': torch.IntTensor((height, width, 3)), 
        }
        return img, img_meta