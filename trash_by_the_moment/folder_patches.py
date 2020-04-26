import glob

import image_patches as im_p


def patch_folder(porcio_patches, carpetes, origin_path, destiny_path, form):
    for file in glob.iglob(origin_path + "/*." + form):
        image_name = file.replace(origin_path + "/", "")
        if carpetes == 1:
            desti_final = destiny_path + "/" + image_name.replace("." + form, "")
        else:
            desti_final = destiny_path

        im_p.get_patches(porcio_patches, origin_path, image_name, desti_final, form)
