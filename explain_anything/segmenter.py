from typing import Tuple, Union, List, Optional
import sys

# Update the system path to include a specific directory
sys.path.insert(0, '/app/third_party/Painter/SegGPT/SegGPT_inference')

from PIL import Image

import numpy as np
import cv2

from .seggpt import SegGPT

from pathlib import Path

class FeatureSegmenter:
	"""
	FeatureSegmenter will allow us to essentially execute multi-feature segmentation. We discover that
	few shot segmentation, SegGPT works best specifically for one feature at a time. Thus, this class
	will help us isolate and iterate through all the different features, and apply SegGPT. Ultimately,
	we use this to generate the multi-feature masks and works best when the features are visually
	noticeable, unique or stands out.

	For small objects that are hard to make out, (as in humans should struggle), see `BboxSegmenter` instead.
    """
	def __init__(self, seggpt_checkpoint: str, feature_count: int):
		"""
		Arguments:
        - seggpt_checkpoint: the checkpoint / weight location of SegGPT
		- feature_count: the number of features to be evaluated
        """
		self.model: SegGPT = SegGPT(seggpt_checkpoint)
		self.feature_count: int = feature_count

		# Initialize the support data storage as lists of lists.
		self.reset_support_data()
	
	def __get_feature_masks(self, mask_path: str):
		"""
		Given a mask containing different integers, with each representing,
		this function isolates each feature onto a different layer, and assigns
		the pixel values to 255. This is used internally to process marks for
		few shot inference.

		Arguments:
		- mask_path: path of mask, should have different integers each representing a different feature
		"""
		ori_feature_map = np.array(Image.open(mask_path).convert("RGB"))
		feature_masks = []
		for feature_id in range(1, self.feature_count + 1):
			mask = (ori_feature_map == feature_id).astype(np.uint8) * 255
			feature_masks.append(mask)
		return feature_masks

	def reset_support_data(self):
		self.support_images = []
		self.support_masks = []

	def add_support_data(self, image_path: str, mask_path: str):
		"""
		This is to be used to add data for SegGPT (few shot segmentation)
		model to learn from on the fly.

		Arguments:
		- image_path: the path of 
		- mask_path: the mask of the above image, consisting of consecutive integers where each integer
					 represents a particular feature. 0 should represent the background, while 1, 2... should represent
					 features.
        """
		image = cv2.imread(image_path)
		feature_masks = self.__get_feature_masks(mask_path)

		self.support_images.append(image)
		self.support_masks.append(feature_masks)

	def generate_segmentation(self, input_image_path: str, output_dir: Optional[str] = None):
		"""
		This performs inference, generating a multi-feature segmentation output. Given the path
		of the image to perform inference on, we iterate through each feature layer and apply
		few shot segmentation to each isolated feature, to which they are merged together at the end,
		ultimately returning one mask containing all features as different integer values.
	
		Arguments:
		- input_image_path: path of the image to segment out
		- output_dir: the local directory to save to, `None` will prevent this from happening.
		"""
		input_image = cv2.imread(input_image_path)
		full_mask = np.zeros(input_image.shape[:2], dtype=np.uint8)

		np_mask = np.array(self.support_masks)

		for feature_id in range(self.feature_count):
			seg_mask = self.model.predict(input_image, self.support_images, np_mask[:, feature_id])

			seg_mask = cv2.cvtColor(seg_mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
			seg_mask = (seg_mask > 0).astype(np.uint8) * (feature_id + 1)

			if len(full_mask.shape) == 2 and len(seg_mask.shape) == 3:  # if full_mask is grayscale and seg_mask is RGB
				seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_RGB2GRAY)

			full_mask = np.maximum(full_mask, seg_mask)
		
		# Save the final segmentation mask
		if output_dir is not None:
			output_path = Path(output_dir) / (Path(input_image_path).stem + '.png')
			cv2.imwrite(str(output_path), full_mask.astype(np.uint8))
		
		return full_mask


class BboxSegmenter:
	"""
	This is also a FeatureSegmenter. The idea is that for smaller objects that take up very little
	space of the whole image, SegGPT is unable to perform well. Thus, this class will take in an image,
    as well as bounding boxes, to which it will perform feature segmentation on this 'cropped out' bounding
    box of the image.
    """
	def __init__(self, seggpt_checkpoint: str, feature_count: int):
		"""
		Arguments:
        - seggpt_checkpoint: the checkpoint / weight location of SegGPT
		- feature_count: the number of features to be evaluated
        """
		self.model: SegGPT = SegGPT(seggpt_checkpoint)
		self.feature_count: int = feature_count

		# Initialize the support data storage as lists of lists.
		self.reset_support_data()

	def _load_yolo_bboxes(self, label_path: str):
		"""
		Given the path of labels, annotated in YOLO format, we extract the information
		and store it in a list of tuples.
		"""
		with open(label_path, 'r') as f:
			bboxes = f.readlines()
		
		parsed_bboxes = []
		for bbox in bboxes:
			_, x_center, y_center, width, height = map(float, bbox.strip().split())
			parsed_bboxes.append((x_center, y_center, width, height))
		
		return parsed_bboxes
	
	def _extract_object(self, image: np.ndarray, bbox: Tuple[int]):
		"""
		Given a bounding box and an image, this returns a cropped
        view of the image, from that bounding box.

		Arguments:
        - image: np.
        """
		h, w = image.shape[:2]
		x_center, y_center, width, height = bbox
		x_center *= w
		y_center *= h
		width *= w
		height *= h
		
		x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
		x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

		x1, y1, x2, y2 = list(map(lambda x: max(x, 0), [x1, y1, x2, y2]))

		return image[y1:y2, x1:x2], (x1, y1, x2, y2)
		
	def __get_feature_masks(self, mask_path: str):
		"""
		Given a mask containing different integers, with each representing,
		this function isolates each feature onto a different layer, and assigns
		the pixel values to 255. This is used internally to process marks for
		few shot inference.

		Arguments:
		- mask_path: path of mask, should have different integers each representing a different feature
		"""

		ori_feature_map = np.array(Image.open(mask_path).convert("RGB"))
		feature_masks = []
		for feature_id in range(1, self.feature_count + 1):
			mask = (ori_feature_map == feature_id).astype(np.uint8) * 255
			feature_masks.append(mask)
		return feature_masks

	def reset_support_data(self):
		self.support_images = [[] for _ in range(self.feature_count)]
		self.support_masks = [[] for _ in range(self.feature_count)]

	def add_support_data(self,
		image_path: str,
		mask_path: str,
		labels: Union[List[List[int]], str]
	):
		"""
		This is to be used to add data for SegGPT (few shot segmentation)
		model to learn from on the fly.

		Arguments:
		- image_path: the path of 
		- mask_path: the mask of the above image, consisting of consecutive integers where each integer
					 represents a particular feature. 0 should represent the background, while 1, 2... should represent
					 features.
		- labels: in the form of a 2D 
        """
		image = cv2.imread(image_path)

		if type(labels) == str:
			bboxes = self._load_yolo_bboxes(labels)
		else:
			bboxes = labels

		feature_masks = self.__get_feature_masks(mask_path)

		for bbox in bboxes:
			obj, _ = self._extract_object(image, bbox)

			for feature_id, mask in enumerate(feature_masks):
				obj_mask, _ = self._extract_object(mask, bbox)
				self.support_images[feature_id].append(obj)

				resized_mask = cv2.resize(obj_mask, (obj.shape[1], obj.shape[0]))
				obj_mask = cv2.cvtColor(resized_mask, cv2.COLOR_RGB2GRAY)
				self.support_masks[feature_id].append(obj_mask)

	def generate_segmentation(
			self,
			input_image_path: str,
			input_label_path: str,
			output_dir: Optional[str] = None,
			pixel_threshold: Optional[int] = 10 
		):
		"""
		This performs inference, generating a multi-feature segmentation output. Given the path
		of the image to perform inference on, we iterate through each feature layer and apply
		few shot segmentation to each isolated feature, to which they are merged together at the end,
		ultimately returning one mask containing all features as different integer values.
	
		Arguments:
		- input_image_path: path of the image to segment out
		- input_label_path: path of the label to identify the object of interest
		- output_dir: the local directory to save to, `None` will prevent this from happening.
		- pixel_thresholds: threshold to filter out poor predictions by few shot model
		"""

		input_image = cv2.imread(input_image_path)
		bboxes = self._load_yolo_bboxes(input_label_path)

		full_mask = np.zeros(input_image.shape[:2], dtype=np.uint8)

		for bbox in bboxes:
			obj, coords = self._extract_object(input_image, bbox)

			for feature_id in range(self.feature_count):
				seg_mask = self.model.predict(obj, self.support_images[feature_id], self.support_masks[feature_id])

				# CV filters and some thresholds are applied here, as I personally found it to help with consistency.
				# Otherwise, SegGPT seems to put random dotted colors all over the background as well.
				seg_mask = cv2.cvtColor((seg_mask * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
				seg_mask = cv2.bilateralFilter(seg_mask, 15, 75, 75)
				seg_mask = (seg_mask > pixel_threshold).astype(np.uint8) * (feature_id + 1)

				if len(full_mask.shape) == 2 and len(seg_mask.shape) == 3:  # if full_mask is grayscale and seg_mask is RGB
					seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_RGB2GRAY)

				x1, y1, x2, y2 = coords

				full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], seg_mask)
		
		# Save the final segmentation mask
		if output_dir is not None:
			output_path = Path(output_dir) / (Path(input_label_path).stem + '.png')
			cv2.imwrite(str(output_path), full_mask.astype(np.uint8))
		
		return full_mask