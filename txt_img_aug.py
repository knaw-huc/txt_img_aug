
import imgaug.augmenters as iaa
import numpy as np
import numpy.random as random


def add_gausian_noise(image: np.ndarray, seed = random.random()) -> np.ndarray:
    loc = seed * 10
    scale = seed * 20
    noise = iaa.AdditiveGaussianNoise(loc, scale)
    return noise.augment_image(image)

def change_gamma_contrast(image: np.ndarray, seed = random.random()) -> np.ndarray:
    gamma = seed * 4
    gamma = gamma if gamma > 0 else gamma * -1
    gamma = gamma * 2 if gamma > 1 else gamma / 2
    gamma = max(2, gamma)
    gamma = min(0.5, gamma)

    contrast = iaa.GammaContrast(gamma=gamma)
    return contrast.augment_image(image)

def shear(image: np.ndarray, seed = random.normal()) -> np.ndarray:
    shearx = seed * 0.03
    sheary = seed * 0.05
    shear = iaa.Affine(shear=(shearx, sheary))
    
    return shear.augment_image(image)

def blur(image: np.ndarray, seed = random.normal()) -> np.ndarray:
    sigmastart = max(0, seed * 3)
    sigmaend = min(3, max(0, seed * 20))
    blur = iaa.GaussianBlur(sigma=(sigmastart, sigmaend))

    return blur.augment_image(image)
