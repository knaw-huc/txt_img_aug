# txt_img_aug

## Prerequisites
1. Have python3 installed
2. Have pip3 installed

## How to use
### Install dependencies
```
pip3 install -r requirements.txt
```

### Image augmentation
```
python3 image_augmentation.py
```
With custom image:
```
python3 image_augmentation.py image=/full/path/to/image
```
With custom random seed of `1`:
```
python3 image_augmentation.py seed=1
```

### Page deformation
```
python3 page_deformation.py
```
With custom image:
```
python3 page_deformation.py image=/full/path/to/image
```
With custom random seed of `1`:
```
python3 page_deformation.py seed=1
```
With black padding, white padding is default:
```
python3 page_deformation.py padding=BLACK
```
With a constant outputsize defined in pixels:
```
python3 page_deformation.py height=1024 width=768
```