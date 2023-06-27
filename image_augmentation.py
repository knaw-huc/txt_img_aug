from PIL import Image
from txt_img_aug import TxtImgAug
from pathlib import Path
from itertools import permutations
import sys
import numpy as np

outputPath = "output/augmentation"
Path(outputPath).mkdir(parents=True, exist_ok=True)

imagePath = "testimage.png"
seed = None
isFirst = True
for arg in sys.argv:
    if isFirst:  # ignore the first argument, because it is the program
        isFirst = False
        continue
    name_val = arg.split("=")
    if str.startswith(name_val[0], "image"):
        imagePath = name_val[1]
    elif str.startswith(name_val[0], "seed"):
        seed = int(name_val[1])

image = np.asarray(Image.open(imagePath))

print("using image: {}".format(imagePath))

tia = TxtImgAug(seed)

augs = (tia.add_gausian_noise, tia.change_gamma_contrast, tia.shear, tia.blur)

perm = list(permutations(augs)) + list(permutations(augs, 3)) + list(permutations(augs, 2)) + list(
    permutations(augs, 1))

no = 0
for permutation in perm:
    augImage = image
    for augmentation in permutation:
        if seed is not None:
            augImage = augmentation(augImage)
        else:
            augImage = augmentation(augImage)

    Image.fromarray(augImage).save("{}/testimage{}.png".format(outputPath, no))
    no += 1
