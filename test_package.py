import imageio
import txt_img_aug as tia
from pathlib import Path
import numpy.random as random
import numpy as np
from itertools import permutations

Path("output").mkdir(parents=True, exist_ok=True)

image = imageio.imread("testimage.png")

augs = (tia.add_gausian_noise, tia.change_gamma_contrast, tia.shear, tia.blur)

perm = list(permutations(augs)) + list(permutations(augs,3)) + list(permutations(augs,2)) + list(permutations(augs,1))

nr=0
for permutation in perm:
    augImage = image
    for augmentation in permutation:
        augImage = augmentation(augImage)

    imageio.imwrite("output/testimage{}.png".format(nr), augImage)
    nr+=1
