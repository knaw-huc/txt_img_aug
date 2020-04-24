
import imgaug.augmenters as iaa
import numpy as np
import numpy.random as random

class TxtImgAug:

    def __init__(self, seed:int=None):
        random.seed(seed)

    def add_gausian_noise(self, image: np.ndarray) -> np.ndarray:
        loc = random.random() * 10
        scale = random.random() * 20
        noise = iaa.AdditiveGaussianNoise(loc, scale)
        return noise.augment_image(image)

    def change_gamma_contrast(self, image: np.ndarray) -> np.ndarray:
        gamma = random.random() * 4
        gamma = gamma if gamma > 0 else gamma * -1
        gamma = gamma * 2 if gamma > 1 else gamma / 2
        gamma = max(2, gamma)
        gamma = min(0.5, gamma)

        contrast = iaa.GammaContrast(gamma=gamma)
        return contrast.augment_image(image)

    def shear(self, image: np.ndarray) -> np.ndarray:
        shearx = random.normal() * 0.03
        sheary = random.normal() * 0.05
        shear = iaa.Affine(shear=(shearx, sheary))
        
        return shear.augment_image(image)

    def blur(self, image: np.ndarray) -> np.ndarray:
        sigmastart = max(0, random.normal() * 3)
        sigmaend = min(3, max(0, random.normal() * 20))
        blur = iaa.GaussianBlur(sigma=(sigmastart, sigmaend))

        return blur.augment_image(image)
