import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import h5py
from torch.utils.data import Dataset
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import fastmri
from fastmri.data import subsample, transforms, mri_data




class DataTransform_M4RAW:
    def __init__(
            self,
            img_size=256,
            combine_coil=True,
            flag_singlecoil=False,
    ):
        self.img_size = img_size
        self.combine_coil = combine_coil  # whether to combine multi-coil imgs into a single channel
        self.flag_singlecoil = flag_singlecoil
        if flag_singlecoil:
            self.combine_coil = True


    def normalize(self, tensor):
        """Normalize tensor to range [0,1]."""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # Avoid division by zero


    def __call__(self, kspace, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]
        Nc = kspace.shape[0]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # # center cropping
        # image_full = transforms.complex_center_crop(image_full, [self.img_size, self.img_size])  # [Nc,H,W,2]
        # # resize img
        # if self.img_size != 320:
        #     image_full = torch.einsum('nhwc->nchw', image_full)
        #     image_full = T.Resize(size=self.img_size)(image_full)
        #     image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]
        image_full = self.normalize(image_full)
        # ====== RSS coil combination ======
        if self.combine_coil:
            image_full = fastmri.rss(image_full, dim=0)  # [H,W]

            # img [B,1,H,W], 
            return  image_full.unsqueeze(0)

        else:  # if not combine coil
            # img [B,Nc,H,W], 
            return  image_full


class LF_M4RawDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (str): Directory with all the .h5 files
            transform (callable): DataTransform_M4RAW instance
        """
        self.transform = transform
        self.examples = []  # (file_path, slice_idx)

        # Precompute all (file_path, slice_idx) pairs
        file_paths = sorted(glob.glob(os.path.join(root_dir, "*.h5")))
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                num_slices = hf['kspace'].shape[0]
                self.examples += [(file_path, slice_idx) for slice_idx in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        file_path, slice_idx = self.examples[idx]
        with h5py.File(file_path, 'r') as hf:
            kspace_slice = hf['kspace'][slice_idx]

        img_tensor = self.transform(
            kspace_slice,
            target=None,
            data_attributes=None,
            filename=os.path.basename(file_path),
            slice_num=slice_idx
        )

        return img_tensor  # [1, H, W]

    

lf_transform = DataTransform_M4RAW(img_size=256, combine_coil=True)
lf_dataset = LF_M4RawDataset(root_dir='dataset/low_field/multicoil_train', transform=lf_transform)

joint_loader = torch.utils.data.DataLoader(lf_dataset, batch_size=2, shuffle=False, num_workers=4)


