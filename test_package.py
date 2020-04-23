import imageio
import txt_img_aug as tia
from pathlib import Path
import numpy.random as random
import numpy as np

Path("output").mkdir(parents=True, exist_ok=True)

image = imageio.imread("testimage.png")

augs = (tia.add_gausian_noise, tia.change_gamma_contrast, tia.shear, tia.blur)

for nr in range(0, 100):
    augImage = image
    numAugs = random.random_integers(1, 4)
    
    for augNum in range(0, numAugs):
        aug = augs[random.random_integers(0, 3)]
        augImage = aug(augImage)

    imageio.imwrite("output/testimage{}.png".format(nr), augImage)
    


# noiseImage = tia.add_gausian_noise(image)
# contrastImage = tia.change_gamma_contrast(image)
# shearedImage = tia.shear(image)
# blurredImage = tia.blur(image)

# imageio.imwrite("output/testimage1.png", noiseImage)
# imageio.imwrite("output/testimage2.png", contrastImage)
# imageio.imwrite("output/testimage3.png", shearedImage)
# imageio.imwrite("output/testimage4.png", blurredImage)
