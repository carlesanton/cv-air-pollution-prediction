# sudo pip install oct2py
import time

import matlab.engine

eng = matlab.engine.start_matlab()

final_data_path = "/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/image_data/7"
data_path = "/Volumes/Data_HD/Users/Carles/TFG/codes/final_data/image_data/Matlab"
nom_patch = "20170112_140200_patch1.jpg"


t = time.time()
print("Starting")
eng.cd("mcloud")
(a, b, c) = eng.first_test(final_data_path, nom_patch, data_path, [12, 16, 20], 8, 300)
print("Done in: " + str(time.time() - t) + " s")
eng.quit()
