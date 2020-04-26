import os

import numpy as np
from PIL import Image


def get_patches(
    porcio_patch, original_folder_path, image_name, destiny_folder_path, format
):
    img = np.asarray(Image.open(original_folder_path + "/" + image_name))
    tamany = img.shape
    x_step = int(tamany[0] / porcio_patch)
    y_step = int(tamany[1] / porcio_patch)
    count = 0
    if not os.path.exists(destiny_folder_path):
        os.makedirs(destiny_folder_path)
    for x in range(0, tamany[0], x_step):
        for y in range(0, tamany[1], y_step):
            p = img[x : (x + x_step), y : (y + y_step), :]
            result = Image.fromarray((p).astype(np.uint8))
            result.save(
                destiny_folder_path
                + "/"
                + image_name.replace("." + format, "")
                + "_patch"
                + str(count)
                + "."
                + format
            )
            count += 1


"""

red = image.extract_patches_2d(img[:,:,0], (int(tamany[0]/porcio_patch), int(tamany[1]/porcio_patch)))
green = image.extract_patches_2d(img[:,:,1], (int(tamany[0]/porcio_patch), int(tamany[1]/porcio_patch)))
blue = image.extract_patches_2d(img[:,:,2], (int(tamany[0]/porcio_patch), int(tamany[1]/porcio_patch)))
print(red.shape)
final_array=np.array((red.shape,3))

print((final_array.shape))
"""
