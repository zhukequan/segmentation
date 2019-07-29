import os,shutil
import SimpleITK as sitk
import re
import glob
import pandas as pd
import numpy as np
paths_imaging = "/home/zkq/Project/Data/kits19/test_data/test_data/case_*/imaging.nii.gz"
imagings = glob.glob(paths_imaging)
com = re.compile("(\d+)")
imagings =sorted(imagings,key = lambda item:int(com.findall(item)[-1]))

image_frame = pd.DataFrame(columns=["spacing(d,w,h)","size(d,w,h)","dirction(d,w,h)","value_range"])
num = 0
min_h = 512
max_h = 0
for file in imagings:
    print(file)
    name = com.findall(file)[-1]
    image = sitk.ReadImage(file)
    spacing = image.GetSpacing()
    size = image.GetSize()
    num+=size[2]
    min_h = min(size[0],min_h)
    max_h = max(size[0], max_h)
    dirction = np.array(image.GetDirection()).reshape((3,3))
    arr = sitk.GetArrayFromImage(image)
    value_min = np.min(arr)
    value_max = np.max(arr)
    image_frame.loc[name] = [spacing,size,dirction,[value_min,value_max]]

image_frame.to_csv("/home/zkq/Project/Data/kits19/test_data/test_information.csv")
