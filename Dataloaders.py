import sys
import os
import glob
import pathlib

import numpy as np
import random
from PIL import Image


import torch as tc
# import torch.nn as nn
# import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
# import matplotlib.pyplot as plt

###############################################################################
# General Dataloaders

class Image_Dataset_(Dataset):
    def __init__(self, root_dir, image_size=(128, 128), transform=None, extra=None):
        """
        Custom Image Dataset with optional extra data.
        
        Args:
            root_dir (str): Path to the parent directory containing images.
            image_size (tuple): (height, width) to resize images to.
            transform (callable, optional): Transformations to apply to images.
            extra (str | np.ndarray | torch.Tensor | None): Optional extra data.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.image_paths = self.gather_image_paths()
        
        # Load optional extra data
        self.extra = self._load_data(extra) if extra is not None else None

        # Ensure extra data matches the number of images
        if self.extra is not None and len(self.extra) != len(self.image_paths):
            raise ValueError(f"Image paths ({len(self.image_paths)}) and Extra ({len(self.extra)}) must match.")

    def gather_image_paths(self):
        """Collect all image file paths under the root directory."""
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[-1].lower() in valid_exts:
                    paths.append(os.path.join(root, file))
        paths.sort()  # Optional: sort paths for consistent order
        return paths

    def _load_data(self, data):
        """Load optional extra data."""
        if isinstance(data, np.ndarray):
            return tc.tensor(data)
        elif isinstance(data, tc.Tensor):
            return data
        elif isinstance(data, str) and data.endswith('.npy'):
            data = np.load(data)
            return tc.tensor(data)
        else:
            raise ValueError(f"Unsupported extra data type: {type(data)}")

    def __len__(self):
        """Total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve (image, extra) pair.
        
        Returns:
            x: torch.Tensor (C, H, W)
            c: torch.Tensor or None
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Always convert to RGB

        if self.transform:
            x = self.transform(image)
        else:
            x = transforms.ToTensor()(image)

        c = self.extra[idx] if self.extra is not None else tc.tensor(0)  # Safe tensor placeholder

        return x, c

    
class Video_Dataset_(Dataset):
    def __init__(self, root_dir, transform=None, frame_limit=None):
        """
        Args:
            root_dir (str): Path to the root directory containing video folders.
            transform (callable, optional): A function/transform to apply to the images.
            frame_limit (int, optional): Maximum number of frames to load from each video.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_limit = frame_limit
        
        # List all video folders in the root directory
        self.video_folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, folder))]
        
        # Sort the video folders to ensure consistency
        self.video_folders.sort()
    
    
    def __len__(self):
        return len(self.video_folders)
    
    
    def __getitem__(self, idx):
        """
        Load all frames from a single video folder.
        """
        video_folder = self.video_folders[idx]
        
        # List all frame image paths and sort them numerically
        frame_files = [os.path.join(video_folder, file) for file in os.listdir(video_folder)
                        if file.endswith(".png")]
        frame_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))  # Sort by frame number
        
        # Limit the number of frames if frame_limit is set
        if self.frame_limit:
            frame_files = frame_files[:self.frame_limit]
        
        # Load all frames
        frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # Stack frames into a tensor of shape (num_frames, C, H, W)
        frames_tensor = tc.stack(frames, dim=0)
        
        return frames_tensor
    
    # def get_batch_(self):
        
        # x = self.__getite__(self)
        
    pass
        

class Numpy_Dataset_(Dataset):
    def __init__(self, data, label=None, extra=None, transform=None,
                 data_batch_axis=0, label_batch_axis=0, extra_batch_axis=0,
                 repeat=1, dtype=None, device=None):
        """
        A flexible PyTorch Dataset that loads main data and optional extra data.
        
        Args:
            data (str | np.ndarray | torch.Tensor): Path to .npy file or raw data array.
            extra (str | np.ndarray | torch.Tensor | None): Optional extra data.
            transform (callable, optional): Transform function for main data.
            batch_axis: the axis with which to sample batches from.
            repeat (int): Repeat dataset multiple times (useful for data augmentation).
        """
        self.dtype = dtype
        self.device = device
        self.transform = transform
        self.data_batch_axis = data_batch_axis
        self.label_batch_axis = label_batch_axis
        self.extra_batch_axis = extra_batch_axis
        
        # Load main data
        self.data = self._load_data(data)

        # Load supervised label data (can be None)
        self.label = self._load_data(label) if label is not None else None
        
        # Load extra conditional data (can be None)
        self.extra = self._load_data(extra) if extra is not None else None

        # Ensure both have the same number of samples
        if self.extra is not None and len(self.extra) != len(self.data) != len(self.label):
            raise ValueError(f"Data length {len(self.data)}, Extra length {len(self.extra)} and Label length {len(self.label)} do not match.")

        # Apply optional repetition (for augmentation)
        if repeat > 1:
            self.data = self.data.repeat((repeat,) + (1,) * (self.data.dim() - 1))

            if self.label is not None:
                self.label = self.label.repeat((repeat,) + (1,) * (self.label.dim() - 1))

            if self.extra is not None:
                self.extra = self.extra.repeat((repeat,) + (1,) * (self.extra.dim() - 1))
                
        # print(self.data.shape, self.extra.shape, self.label.shape)
        return
    
    
    def _load_data(self, data):
        """Helper function to load data from file or tensor."""
        if isinstance(data, np.ndarray):
            data = tc.tensor(data, dtype=self.dtype)
        elif isinstance(data, tc.Tensor):
            pass
        elif isinstance(data, str) and data.endswith('.npy'):
            data = np.load(data)    # Load .npy file
            data = tc.tensor(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        if self.dtype is not None:
            data = data.to(dtype=self.dtype)
        if self.device is not None:
            data = data.to(device=self.device)
        
        return data
    

    def __len__(self):
        """Return number of samples."""
        return self.data.shape[0]


    def __getitem__(self, idx):
        """Return a batch (x, c), where c can be None."""
        x = self.data[idx]                                          # (B, D1...)
        y = self.label[idx] if self.label is not None else tc.nan   # (B, D2...)
        c = self.extra[idx] if self.extra is not None else tc.nan   # (B, C...)

        if self.transform:
            x = self.transform(x)

        return x, y, c  # Return tuple (data, label, extra)
    
    pass


###############################################################################
# Custom Dataloaders
    
class Sequence_Dataset_(Numpy_Dataset_):
    def __init__(self, data, label=None, extra=None, transform=None, repeat=-1, dtype=None, device=None, config=0, 
                 seq_len=16):
        """
        Sequence-aware dataset that extracts fixed-length subsequences from full sequences.

        Args:
            seq_len (int): Length of each subsequence to sample.
            num_samples (int): Total number of subsequences to sample randomly.
        """
        super().__init__(data, label, extra, transform, repeat, dtype, device)
        self.seq_len = seq_len

        self.config = config
        # 0: x=x, y=y, c=c                  Uses as is.   Used for Full Sequence Diffusion
        # 1: x=x[t], y=y, c=[t-l:t]         Overwrites c. Used for Sequential Diffusion
        # 2: x=x[t-l:t], y=x[t], c=c        Overwrites y. Used for Sequential Regression
        
        return

    def __getitem__(self, idx):
        # Select a full sequence (B, L, D1...)
        x = self.data[idx]                                          # (L, D1...)
        y = self.label[idx] if self.label is not None else tc.nan   # (D2...)
        c = self.extra[idx] if self.extra is not None else tc.nan   # (C...)
        
        # Random start index
        L = x.shape[0]
        max_start = L - self.seq_len
        start_idx = random.randint(0, max_start)
        
        # Extract subsequence
        x = x[start_idx:start_idx + self.seq_len]   # (l, D1...)
        
        # Optional transform
        if self.transform:
            x = self.transform(x)
        
        if self.config == 0:
            pass
            
        elif self.config == 1:
            x, c = x[-1], x[:-1]                    # (D1...), (l-1, D1...)
            
        elif self.config == 2:
            x, y = x[:-1], x[-1]                    # (l-1, D1...), (D1...)
        
        return x, y, c  # Return tuple (data, label, extra)
    
###############################################################################

class Repeating_Sampler_(Sampler):
    def __init__(self, data_source, batch_size, num_batches):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        dataset_size = len(self.data_source)
        indices = tc.randint(0, dataset_size, size=(self.num_batches * self.batch_size,))
        return iter(indices.tolist())

    def __len__(self):
        return self.num_batches * self.batch_size

###############################################################################





