import imageio
from txt_img_aug import TxtImgAug
from pathlib import Path
import sys

imagePath = "testimage.png"
seed = None
padding = "WHITE"
resize_height = None
resize_width = None
for arg in sys.argv:
    name_val = arg.split("=")
    if name_val[0].strip() == "image":
        imagePath = name_val[1].strip()
    elif name_val[0].strip() == "seed":
        seed = int(name_val[1])
    elif name_val[0].strip() == "padding":
        padding = name_val[1].strip() if name_val[1].strip() == "BLACK" else "WHITE"
    elif name_val[0].strip() =="height":
        resize_height = int(name_val[1])
    elif  name_val[0].strip() == "width":
        resize_width = int(name_val[1])

image = imageio.imread(imagePath)

print("using image: {}".format(imagePath))
print("padding colour: {}".format(padding))
print("resize to height: {}".format(resize_height))
print("resize to width: {}".format(resize_width))

tia = TxtImgAug(seed)

outputPath = "output/page_deformation"
Path(outputPath).mkdir(parents=True, exist_ok=True)

for no in range(0,50):
    deformed = tia.elastic_defromation(image)
    deformed = tia.scale(deformed, resize_width, resize_height)
    deformed = tia.pad(deformed, resize_width, resize_height, padding)

    imageio.imwrite("{}/testimage{}.png".format(outputPath, no), deformed)
