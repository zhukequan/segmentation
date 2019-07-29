import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from resample import Resample
from resegmentNet import Net
import os
from Swell import Swell
from unetpp import CurUnetppModel2
import glob
import pandas as pd
import re
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sess = tf.InteractiveSession()
resegnet = Net(1,sess,1,32,512,512,model_dir="/home/zkq/Project/MICCAI2019/2D_3D_V-net/temp/12000")
swell_op = Swell(sess)
unetpp = CurUnetppModel2(sess=sess,model_dir="/home/zkq/Project/MICCAI2019/unetpp2/unetpp_trained_1_1_1_90/run_000",
               channels=1,
               n_class=3,
                layers=5,
                features_root=32,
                cost_kwargs={"class_weights":[1,1,1]},
               model_type = "nestnet",
               deep_supervision = True)

com = re.compile("(\d+)")
# data_root = "/home/zkq/Project/Data/liver/ISICDM2019/liver_isicdm2"
image_reg = "/home/zkq/Project/Data/kits19/test_data/test_data/case_*/imaging.nii.gz"

image_files = glob.glob(image_reg)


image_files = sorted(image_files, key=lambda item: int(com.findall(item)[-1]))
out_path = "./predictions_unetpp2"
len_test = len(image_files)
if(not os.path.exists(out_path)):
    os.makedirs(out_path)
for image_file_name in image_files:
    keyname=image_file_name.split("/")[-2]
    keyname = com.findall(keyname)[-1]
    input_image = sitk.ReadImage(image_file_name)
    input_arr = sitk.GetArrayFromImage(input_image)
    input_arr = input_arr.astype(np.float32)
    input_arr2 = input_arr.transpose(2,1,0)
    input_arr2 = (input_arr2-np.mean(input_arr2))/np.std(input_arr2)
    input_spacing  = input_image.GetSpacing()
    input_size = input_image.GetSize()
    down_spacing = [5,input_spacing[1],input_spacing[2]]
    down_image = Resample(input_image,down_spacing)
    down_arr = sitk.GetArrayFromImage(down_image)
    down_arr2 = down_arr.transpose(2,1,0).astype(np.float32)
    # down_arr2[down_arr2 < -1024] = -1024
    # down_arr2[down_arr2 > 2000] = 2000
    down_arr2 =(down_arr2-np.mean(down_arr2))/np.std(down_arr2)
    down_reseg = resegnet.test_slice(down_arr2,16)
    down_predict = down_reseg[...,1]>0.5
    down_predict = down_predict.astype(np.float32)

    down_predict = down_predict[np.newaxis,...,np.newaxis]
    down_swell = swell_op(down_predict)
    down_swell = np.squeeze(down_swell)
    swell_arr =down_swell[...,np.newaxis]
    reseg_arr = down_reseg
    down_arr2 = down_arr2[...,np.newaxis]
    predict = unetpp(down_arr2,reseg_arr,swell_arr,-1)
    predict =np.argmax(predict,axis=-1)
    predict = predict.transpose(2,1,0)
    predict_image = sitk.GetImageFromArray(predict)
    predict_image.CopyInformation(down_image)
    predict_image = Resample(predict_image,input_spacing,input_size,interpolator="NearestNeighbor")
    out_file_name = "prediction_" + keyname + ".nii.gz"
    sitk.WriteImage(predict_image, os.path.join(out_path, out_file_name))
    print("finish_" + out_file_name)

