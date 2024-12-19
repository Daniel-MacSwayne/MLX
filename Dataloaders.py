import sys
import os

import numpy as np
# import xgboost as xgb
from PIL import Image


import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# from Architectures import *
# from Diffusion import *
# from Autoencoders import *

###############################################################################

class VideoFramesDataset(Dataset):
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


