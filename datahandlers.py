import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import h5py
from torch.utils.data import Dataset, ConcatDataset
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from fastmri.data import subsample, transforms, mri_data
import pydicom
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

def normalize(tensor):
    """Normalize tensor to range [0,1]."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # Avoid division by zero



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
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]        
        # ====== RSS coil combination ======
        if self.combine_coil:
            image_full = fastmri.rss(image_full, dim=0)  # [H,W]
            # img [B,1,H,W], 
            return  self.normalize(image_full).unsqueeze(0)

        else:  # if not combine coil
            # img [B,Nc,H,W], 
            return  self.normalize(image_full)


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


class HF_MRI_Dataset(Dataset):
    def __init__(self, root_dir, transform, split="train", val_size=0.1, test_size=0.1):
        """
        Args:
            root_dir (str): Directory with subject folders containing DICOM files
            transform (callable): Transform instance for processing images
            split (str): Data split ('train', 'val', 'test')
            val_size (float): Proportion of the data to use for validation
            test_size (float): Proportion of the data to use for test
        """
        self.subject_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))  # Each folder represents a subject
        self.transform = transform

        # Prepare list of all DICOM files (each DICOM file corresponds to one slice)
        self.dicom_files = []
        for subject_dir in self.subject_dirs:
            dicom_files = sorted(glob.glob(os.path.join(subject_dir, "*.dcm")))  # All DICOM files for the subject
            for dicom_file in dicom_files:
                self.dicom_files.append(dicom_file)

        # Split the data into train, validation, and test sets
        train_files, test_files = train_test_split(self.dicom_files, test_size=test_size, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=42)

        # Assign data split based on the argument
        if split == "train":
            self.dicom_files = train_files
        elif split == "val":
            self.dicom_files = val_files
        elif split == "test":
            self.dicom_files = test_files
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        # Retrieve one DICOM file (one slice)
        dicom_file = self.dicom_files[idx]
        dicom_data = pydicom.dcmread(dicom_file)

        img_array = dicom_data.pixel_array.astype(np.float32)
        img_transformed = self.transform(image=img_array)["image"]
        return img_transformed

class UnpairedMergedDataset(torch.utils.data.Dataset):
    def __init__(self, lf_dataset, hf_dataset):
        self.lf_dataset = lf_dataset
        self.hf_dataset = hf_dataset
        self.length = min(len(lf_dataset), len(hf_dataset))  # ensure same length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        lf_sample = self.lf_dataset[idx]
        hf_sample = self.hf_dataset[idx]

        # Return both samples as a tuple
        return lf_sample, normalize(hf_sample)

def lf_hf_collate_fn(batch):
    """
    Batch: a list of 1 element -> [(lf_sample, hf_sample)]
    """
    lf_sample, hf_sample = batch[0]  # batch_size = 1

    images = torch.stack([lf_sample, hf_sample], dim=0)  # [2, C, H, W]
    return images


# ###### Transforms #######
# transform_hf = A.Compose([
#     A.Resize(256, 256),
#     ToTensorV2()
# ])
# lf_transform = DataTransform_M4RAW(img_size=256, combine_coil=True)

# #### Low Field Datasets: Train, Test, Val ####
# lf_dataset_train = LF_M4RawDataset(root_dir='dataset/low_field/multicoil_train', transform=lf_transform)
# lf_dataset_val = LF_M4RawDataset(root_dir="dataset/low_field/multicoil_val", transform=lf_transform)
# lf_dataset_test = LF_M4RawDataset(root_dir="dataset/low_field/multicoil_test", transform=lf_transform)

# #### High Field Datasets: Train, Test, Val
# hf_dataset_train = HF_MRI_Dataset(root_dir="dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
#                                   transform=transform_hf,
#                                   split="train")
# hf_dataset_val = HF_MRI_Dataset(root_dir="dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
#                                   transform=transform_hf,
#                                   split="val")
# hf_dataset_test = HF_MRI_Dataset(root_dir="dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM", 
#                                   transform=transform_hf,
#                                   split="test")


# #### Concat Datasets ####
# train_set = UnpairedMergedDataset(lf_dataset_train, hf_dataset_train)
# val_set = UnpairedMergedDataset(lf_dataset_val, hf_dataset_val)
# test_set = UnpairedMergedDataset(lf_dataset_test, hf_dataset_test)

# #### DataLoaders ####
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, collate_fn=lf_hf_collate_fn)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=lf_hf_collate_fn)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=lf_hf_collate_fn)
