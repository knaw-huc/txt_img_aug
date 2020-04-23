
import imgaug.augmenters as iaa
import numpy as np
import numpy.random as random


def add_gausian_noise(image: np.ndarray) -> np.ndarray:
    loc = random.random() * 10
    scale = random.random() * 20
    noise = iaa.AdditiveGaussianNoise(loc, scale)
    return noise.augment_image(image)

def change_gamma_contrast(image: np.ndarray) -> np.ndarray:
    var = random.random() * 4
    gamma = random.normal(1, var)
    gamma = gamma if gamma > 0 else gamma * -1
    gamma = gamma * 2 if gamma > 1 else gamma / 2

    contrast = iaa.GammaContrast(gamma=gamma)
    return contrast.augment_image(image)

def shear(image: np.ndarray) -> np.ndarray:
    shearx = random.normal()
    sheary = random.normal()
    shear = iaa.Affine(shear=[shearx, sheary])
    
    return shear.augment_image(image)

def blur(image: np.ndarray) -> np.ndarray:
    blur = iaa.GaussianBlur()

    return blur.augment_image(image)
