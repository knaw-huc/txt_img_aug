import imageio
import txt_img_aug as tia
from pathlib import Path
import numpy.random as random
import numpy as np
from itertools import permutations
import sys

Path("output").mkdir(parents=True, exist_ok=True)

image = None
seed = None
for arg in sys.argv:
    name_val = arg.split("=")
    if str.startswith(name_val[0], "image"):
        image = imageio.imread(name_val[1])
    elif str.startswith(name_val[0], "seed"):
        seed = int(name_val[1])


if image is None:
    print("using default image: testimage.png")
    image = imageio.imread("testimage.png")

augs = (tia.add_gausian_noise, tia.change_gamma_contrast, tia.shear, tia.blur)

perm = list(permutations(augs)) + list(permutations(augs,3)) + list(permutations(augs,2)) + list(permutations(augs,1))

nr=0
for permutation in perm:
    augImage = image
    for augmentation in permutation:
        if seed is not None:
            augImage = augmentation(augImage, seed)
        else:
            augImage = augmentation(augImage)

    imageio.imwrite("output/testimage{}.png".format(nr), augImage)
    nr+=1
