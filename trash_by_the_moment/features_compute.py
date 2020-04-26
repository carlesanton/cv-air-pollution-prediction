import os
import sqlite3

import matlab.engine

import download_amos as da
import pollution_parser as pp
import utils as u


def get_features_from_folder(eng: matlab.engine, folder_path):
    folder_name = folder_path.split("/")[-1]
    samples = []
    classes = []

    for f in os.listdir(folder_path):
        if f.endswith(".jpg"):
            print("Getting features from: " + f, end="\r")
            data_path = folder_path + "/features"
            u.check_folder_and_create(data_path, 0)
            raw_desc_feat = eng.first_test(
                folder_path, f, data_path, [12, 16, 20], 8, 300
            )
            print("Saved faetures into: " + data_path)
            final_fisher_vector = []
            for raw in raw_desc_feat:
                final_fisher_vector.append(raw)
            samples.append(final_fisher_vector)
            classes.append(int(folder_name))

    return (samples, classes)


def patch_and_features_from_folder(
    camera_id,
    folder_location,
    final_data_path,
    patches_info_path,
    db_cursor,
    eng: matlab.engine,
):
    cameras_params = pp.get_fields_from_table(
        db_cursor,
        "camera_sites",
        "camera_id,O2, SO2, CO, NO2, PM25, PM10",
        "where camera_id=" + str(camera_id),
    )
    camera_params = cameras_params[0]
    par_ids = [camera_params[i] for i in range(len(camera_params)) if i != 0]
    camera_id = camera_params[0]
    print(camera_params)
    print(par_ids)
    print(camera_id)
    # print(par_ids)
    print("Getting AQI data")
    aqi_query_condition = (
        "where site LIKE '"
        + camera_params[1].replace(camera_params[1].split("-")[-1], "")
        + "%'"
    )  # que almenys estigui en el mateix condat
    all_aqi_site_data = pp.get_fields_from_table(
        db_cursor, "poll_aqi", "site,data,sample", aqi_query_condition
    )
    u.delete_last_lines()
    print("Got all site AQI data ")
    # print(aqi_index)

    print("Getting patches from " + folder_location)
    da.put_images_to_aqi_folder(
        camera_id,
        all_aqi_site_data,
        par_ids,
        folder_location,
        patches_info_path,
        final_data_path,
    )
    print("Done")


camera_id = 30877

patches_info_path = "/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/camera_patches_info"

final_data_path = (
    "/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/image_data/"
    + "{:0>8}".format(camera_id)
)
folder_location = "/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/downloaded_images/00030877_201701"

print("Startting matlab engine")
eng = matlab.engine.start_matlab()
print("Mtlab engine started")

db_connection, db_cursor = pp.initialise_db("pollution")

# patch_and_features_from_folder(camera_id, folder_location,final_data_path,patches_info_path,db_cursor,eng)
eng.cd("mcloud")
for f in os.listdir(final_data_path):
    if os.path.isdir(os.path.join(final_data_path, f)) and f != "10":
        print("Getting features from: " + str(os.path.join(final_data_path, f)))
        get_features_from_folder(eng, os.path.join(final_data_path, f))


pp.close_db_cursor(db_connection)

"""
final_data_path='/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/image_data/00030877'
ext = ['33.jpg','34.jpg','35.jpg','36.jpg','37.jpg','38.jpg','39.jpg','40.jpg','41.jpg','42.jpg','43.jpg','44.jpg']
for f in os.listdir(final_data_path):
	if os.path.isdir(os.path.join(final_data_path,f)):
		for ff in os.listdir(os.path.join(final_data_path,f)):
			if ff.endswith(tuple(ext)):
				os.remove(os.path.join(final_data_path,f,ff))
				print("Deleted image: " + str(os.path.join(f,ff)))
"""
