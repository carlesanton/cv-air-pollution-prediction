import os
import shutil

import numpy as np

source1 = "swimcat_dataset/B-pattern_train/images"
dest11 = "swimcat_dataset/B-pattern_test"
files = os.listdir(source1)

for f in files:
    print(f)
    if np.random.rand(1) < 0.2:
        print(source1 + "/" + f, dest11 + "/" + f)
        shutil.move(source1 + "/" + f, dest11 + "/" + f)
