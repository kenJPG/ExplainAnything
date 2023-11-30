# Explain Anything
Explaining any model through user-defined concepts

--- 
### Setup
Dockerfile and script to are provided. Simply `docker build . -t explain_anything` and perform `sudo bash run_docker.sh`.

Please note that weights for the examples (only for the example ResNet and YOLO model, **SegGPT** is provided via Dockerfile) are **not available** in this repo and will have to be re-trained.

Additionally, the datasets are removed however, the classification dataset CelebAMask-HQ is kept here for demonstration purposes (not full size, roughly *400 images*).

---
### Usage
For best demonstration and understanding of how to use, refer to the notebooks found in `examples`. 

Explaining classification scores can be found in `examples/classification`. The dataset used is CelebA.
Explaining object detection through detection scores can be found in `examples/detection/main.ipynb`. The dataset used here is NEA Rodent dataset.

However, as a high level API, you can interact with it like so:
```
# We have some hand labelled segmentations. Let's use these to generate some more!
from explain_anything.segmenters import FeatureSegmenter

fs = FeatureSegmenter(
	seggpt_checkpoint = '/seggpt_vit_large.pth'
	feature_count = 3
)

support_images = ['img1.jpg', 'img2.jpg']
support_masks = ['img1_mask.png', 'img2_mask.png']
for s_img, s_mask in zip(support_images, support_masks):
	fs.add_support_data(
		s_img, s_mask
	)

for image in ['img3.jpg', 'img4.jpg', 'img5.jpg']:
	fs.generate_segmentation(
		input_image_path = image,
		output_dir = '/output_masks'
	)
```

Once we have gotten more samples and features, we can evaluate our model
```
from explain_anything.explain import ExA
model = resnet50() # Note this model should only return a confidence score

exa = ExA()
features = ['leaves', 'trunk', 'roots']

importances = exa.ablate(
	model,
	['img3.jpg', 'img4.jpg', 'img5.jpg'],
	'/output_masks',
	features
)

for importance, feature_name in list(sorted(zip(importances, ['background] + features)))[::-1]:
	print(f'{feature_name}:', importance)
```