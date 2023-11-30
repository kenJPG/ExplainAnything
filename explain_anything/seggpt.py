from typing import Optional, Union, List

import sys
sys.path.insert(0, '/app/third_party/Painter/SegGPT/SegGPT_inference')

from seggpt_engine import run_one_image
import models_seggpt

import torch
import torch.nn.functional as F

from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt


class SegGPT:
	IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
	IMAGENET_STD = np.array([0.229, 0.224, 0.225])

	def __init__(
		self,
		checkpoint: str,
		seg_type: str = 'instance',
		device: Optional[str] = 'cuda:0'
	):
		"""
		Set up the SegGPT Few Shot segmentation class.
		This is intended to act as a class wrapper around the actual code
		from the repo itself, to combine all preprocessing and inference
		steps into one class.

		Arguments:
		- checkpoint: location of the 
		- seg_type: the type of segmentation to perform
		"""
		arch = 'seggpt_vit_large_patch16_input896x448'

		self.model = getattr(models_seggpt, arch)()
		self.model.seg_type = seg_type
		
		checkpoint = torch.load(checkpoint, map_location = 'cpu')
		self.model.load_state_dict(checkpoint['model'], strict = False)

		self.model.eval()
		self.model = self.model.to(torch.device(device))
		print("[SegGPT] Model Loaded")

		self.device = torch.device(device)

	def __process_image(
		self,
		image: Union[Image.Image, np.array],
		desired_shape: List[int]
	):
		"""
		This is used to ensure that any images that are passed as input
		will be parsed and formatted to the correct output before being 
		given to the inference step.

		Arguments:
		- image: the image to be preprocessed
		- desired_shape: the shape to resize this image to
		"""
		# Convert to numpy if it's an image object
		if isinstance(image, Image.Image):
			image = np.array(image)
		
		# Handle grayscale to RGB conversion for both arrays and images
		if len(image.shape) == 2 or image.shape[2] == 1:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		
		# Resize and normalize
		image_resized = cv2.resize(image, desired_shape) / 255.
		image_normalized = (image_resized - self.IMAGENET_MEAN) / self.IMAGENET_STD

		return image, image_normalized

	def predict(
		self,
		images: List[Union[str, np.ndarray, Image.Image]],
		support_images: List[Union[str, np.ndarray, Image.Image]],
		support_masks: List[Union[str, np.ndarray, Image.Image]],
		out_path=None
	):
		"""
		Whether to visualize
		"""

		wres, hres = 448, 448
		
		image_batch, mask_batch = [], []

		# Convert single path or image array to list format for easier processing
		if type(images) in [str, np.ndarray, Image.Image]:
			images = [images]

		for img, support_img, support_mask in zip(images, support_images, support_masks):
			# Handle both paths and direct arrays
			if type(img) == str:
				img = Image.open(img).convert("RGB")
			
			# Process the main image
			input_img, p_img = self.__process_image(img, (wres, hres))
			size = input_img.shape[:2][::-1]
			
			# Process the support image
			if type(support_img) == str:
				support_img = Image.open(support_img).convert("RGB")
			_, p_support_img = self.__process_image(support_img, (wres, hres))
			
			# Process the support target
			if type(support_mask) == str:
				support_mask = Image.open(support_mask).convert("RGB")
			_, p_support_mask = self.__process_image(support_mask, (wres, hres))
			
			# Duplicate tgt2 as tgt
			p_support_mask_2 = np.copy(p_support_mask)
			
			# Concatenate
			combined_img = np.concatenate((p_support_img, p_img), axis=0)
			combined_mask = np.concatenate((p_support_mask_2, p_support_mask), axis=0)

			image_batch.append(combined_img)
			mask_batch.append(combined_mask)

		imgs = np.stack(image_batch, axis=0)
		masks = np.stack(mask_batch, axis=0)

		# Run the segmentation process
		raw_output = run_one_image(imgs, masks, self.model, self.device)

		raw_output = F.interpolate(
			raw_output[None, ...].permute(0, 3, 1, 2),
			size=[size[1], size[0]],
			mode='nearest',
		).permute(0, 2, 3, 1)[0].numpy()

		# Visualize image
		if out_path is not None:
			output = Image.fromarray((input_img, * (0.6 * raw_output / 255 + 0.4)).astype(np.uint8))
			output.save(out_path)

		return raw_output