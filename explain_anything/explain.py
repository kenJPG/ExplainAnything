from typing import Optional, List, Callable, Union
import sys
sys.path.insert(0, '/app/third_party/Painter/SegGPT/SegGPT_inference')

import os
import numpy as np
import cv2
import torch
from PIL import Image
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from captum.attr import FeatureAblation
from captum.attr import visualization as viz

from tqdm import tqdm

import matplotlib.pyplot as plt

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

class ExA:
	"""
	ExA class for processing images, creating masks, and performing feature ablation. This is the
	main function that is responsible from converting segmentations to more interpretable results.
	"""

	def __init__(
		self,
		device: Optional[str] = 'cuda'
	):
		self.device = device

	def ablate(
		self,
		subject_model: Callable,
		imgs_dir: Union[List[str], str],
		masks_dir: Union[List[str], str],
		features: List[str], # list of feature names
		batch_size: Optional[int] = 8,
		num_workers: Optional[int] = 4,
		device: Optional[str] = 'cuda',
		custom_baseline: Optional[torch.tensor] = None,
		grayscale: Optional[bool] = False
	):
		"""
		This is the main function that will perform feature ablation, according to the different features present
		in the mask.

		Arguments:
		- subject_model: the model that we aim to understand / interpret
		- imgs_dir: path / list of files regarding the images to ablate the model upon
		- masks_dir: path / list of masks (PNG file) to indicate the different segments of an image
		- features: simply a list of the feature names
		- num_workers: doesn't work in docker without IPC host
		- custom_baseline: this acts as a post-processing step that can be customized accordingly
		- grayscale: whether we are dealing with grayscale images

		"""
		exa_dataset = ExADataset(imgs_dir, masks_dir, transform=transforms.ToTensor())
		exa_loader = DataLoader(exa_dataset)
		# , batch_size = batch_size, num_workers = num_workers)
		ablator = FeatureAblation(subject_model.to(device))
		features = ['background'] + features

		feature_sum = np.zeros(len(features))
		feature_count = np.zeros(len(features))

		# Process each batch of images and masks
		for img_batch, mask_batch in tqdm(exa_loader):
			img_batch = img_batch.to(device)

			mask_batch = mask_batch.to(device).int()

			# Convert images and masks to grayscale if needed
			if grayscale:
				img_batch = img_batch.mean(dim=1, keepdim=True)
				mask_batch = mask_batch.float().mean(dim=1, keepdim=True).int()

			if custom_baseline is None:
				baseline = torch.tensor((np.random.rand(*mask_batch.shape) * 255).astype(np.uint8)).to(device)
			else:
				baseline = custom_baseline(mask_batch.shape)

			# Compute the attribution for the batch
			attr = ablator.attribute(img_batch, feature_mask = mask_batch, baselines = baseline)

			# Process the attributions and update the feature importance scores
			for i in range(batch_size):
				attr_flat = attr.mean(axis=1).cpu().numpy().flatten()
				mask_flat = mask_batch.cpu().numpy().max(axis=1).flatten().astype(np.uint8)

				np.add.at(feature_sum, mask_flat, np.abs(attr_flat))

				bin_count = np.bincount(mask_flat)
				padded_bin_count = np.pad(bin_count, (0, len(features) - len(bin_count)), 'constant') if len(bin_count) < len(features) else bin_count
				feature_count += padded_bin_count

		# Calculate and return normalized scores
		scores = feature_sum / feature_count
		scores[np.isnan(scores)] = 0
		return scores / scores.sum()

class ExADataset(Dataset):
	"""
	This is a custom torch dataset used by Explain Anything module to iterate over
	the different features of an image. Serving mainly as a wrapper and a translator
	between data types.
	"""
	def __init__(self, imgs_dir, masks_dir, transform=None):
		self.imgs_dir = imgs_dir
		self.masks_dir = masks_dir

		if type(self.imgs_dir) == str:
			self.imgs = [filename for filename in os.listdir(imgs_dir) if isfile(join(imgs_dir, filename))]

		if type(self.masks_dir) == str:
			self.masks = [filename for filename in os.listdir(masks_dir) if isfile(join(masks_dir, filename))]

		self.transform = transform

	def __len__(self):
		if type(self.imgs_dir) == list:
			return len(self.imgs_dir)
		return len(os.listdir(self.imgs_dir))

	def __getitem__(self, idx):
		if type(self.imgs_dir) == list:
			img_name = self.imgs_dir[idx]
		else:
			img_name = os.path.join(self.imgs_dir,
									self.imgs[idx])

		if type(self.masks_dir)	 == list:
			mask_name = self.masks_dir[idx]
		else:
			mask_name = os.path.join(self.masks_dir,
							self.masks[idx])
		
		image = transforms.ToTensor()(Image.open(img_name))
		mask_tensor = torch.tensor(np.array(Image.open(mask_name).convert('RGB'))).permute((2, 0, 1))

		image = transforms.Resize((480, 640))(image).float()
		mask = transforms.Resize((480, 640))(mask_tensor).float()

		return image, mask