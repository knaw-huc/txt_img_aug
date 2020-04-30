
import imgaug.augmenters as iaa
import numpy as np
import numpy.random as random
from PIL import Image

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
        shearx = random.normal() * 0.2
        sheary = random.normal() * 0.2
        shear = iaa.Affine(shear=(shearx, sheary))
        
        return shear.augment_image(image)

    def blur(self, image: np.ndarray) -> np.ndarray:
        sigmastart = max(0, random.normal() * 3)
        sigmaend = min(3, max(0, random.normal() * 20))
        blur = iaa.GaussianBlur(sigma=(sigmastart, sigmaend))

        return blur.augment_image(image)

    def elastic_defromation(self, image: np.ndarray) -> np.ndarray:
        alpha = random.normal(scale=10) * 3
    
        alpha = alpha if alpha > 0 else alpha * -1

        deform = iaa.ElasticTransformation(alpha=alpha)

        return deform.augment(image=image)

    def pad(self, image: np.ndarray, width:int=None, height:int=None, padding:str="WHITE") -> np.ndarray:
        real_image = Image.fromarray(image)
        new_height = height if height is not None else real_image.height
        new_width = width if width is not None else real_image.width
        
        new_image = Image.new("RGBA", (new_width, new_height), padding)
        new_image.paste(real_image, (0, 0), real_image)
        new_image.convert('RGB')
        
        return np.asarray(new_image)
    
    def scale(self, image:np.ndarray, width:int=None, height:int=None, padding:str="WHITE", max_scale:int=4) -> np.ndarray:
        if height is None and width is None:
            return image

        real_image = Image.fromarray(image)

        max_height = real_image.height * max_scale
        max_width = real_image.width * max_scale
        
        to_height = height if height is not None and height and height < max_height else max_height
        to_width = width if width is not None and width and width < max_width else max_width
        real_image = real_image.resize(size=(to_width, to_height))
        
        return np.asarray(real_image)
