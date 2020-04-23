import imageio
import txt_img_aug as tia
from pathlib import Path
import imgaug as ia

Path("output").mkdir(parents=True, exist_ok=True)

image = imageio.imread("testimage.png")

noiseImage = tia.add_gausian_noise(image)
contrastImage = tia.change_gamma_contrast(image)
shearedImage = tia.shear(image)
blurredImage = tia.blur(image)

imageio.imwrite("output/testimage1.png", noiseImage)
imageio.imwrite("output/testimage2.png", contrastImage)
imageio.imwrite("output/testimage3.png", shearedImage)
imageio.imwrite("output/testimage4.png", blurredImage)
