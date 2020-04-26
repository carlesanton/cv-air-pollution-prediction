import os

from PIL import Image

path = "images/"

for foto in os.listdir(path):
    if foto.endswith(".jpg"):
        print(foto)
        im = Image.open(path + foto)
        im.save(path + foto.replace("jpg", "png"))
