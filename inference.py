import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from resample import Resample
from resegmentNet import Net
import os
from Swell import Swell
from unetpp import CurUnetppModel
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sess = tf.InteractiveSession()
resegnet = Net(1,sess,1,32,512,512,model_dir="/home/zkq/Project/MICCAI2019/2D_3D_V-net/temp/12000")
swell_op = Swell(sess)
unetpp = CurUnetppModel(sess=sess,model_dir="/home/zkq/Project/MICCAI2019/unetpp/unetpp_trained_1_1_1_90/run_000",
               channels=1,
               n_class=3,
                layers=5,
                features_root=32,
                cost_kwargs={"class_weights":[1,1,1]},
               model_type = "nestnet",
               deep_supervision = True)
image_file_name = "/home/zkq/Project/Data/kits19/train_data/data/case_00141/imaging.nii.gz"
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
down_arr2 =(down_arr2-np.mean(down_arr2))/np.std(down_arr2)
down_reseg = resegnet.test_slice(down_arr2,16)[...,1]
down_predict = down_reseg>0.5
down_predict = down_predict.astype(np.float32)

down_predict = down_predict[np.newaxis,...,np.newaxis]
down_swell = swell_op(down_predict)
down_swell = np.squeeze(down_swell)
down_reseg2  = down_reseg.transpose(2,1,0)
down_swell2  = down_swell.transpose(2,1,0)
down_reseg_image = sitk.GetImageFromArray(down_reseg2)
down_reseg_image.CopyInformation(down_image)
down_swell_image = sitk.GetImageFromArray(down_swell2)
down_swell_image.CopyInformation(down_image)
reseg_image = Resample(down_reseg_image,input_spacing,input_size,interpolator="Linear")
swell_image = Resample(down_swell_image,input_spacing,input_size,interpolator="Linear")
reseg_arr = sitk.GetArrayFromImage(reseg_image)
swell_arr = sitk.GetArrayFromImage(swell_image)
swell_arr = (swell_arr>0.5).astype(np.float32)
reseg_arr = reseg_arr.transpose(2,1,0)
reseg_arr = np.stack([1-reseg_arr,reseg_arr],axis=-1)
swell_arr = swell_arr.transpose(2,1,0)
input_arr2 = input_arr2[...,np.newaxis]
swell_arr =swell_arr[...,np.newaxis]
predict = unetpp(input_arr2,reseg_arr,swell_arr,-1)
predict =np.argmax(predict,axis=-1)
predict = predict.transpose(2,1,0)
seg_file_name =  "/home/zkq/Project/Data/kits19/train_data/data/case_00141/segmentation.nii.gz"
seg_image = sitk.ReadImage(seg_file_name)
seg_arr = sitk.GetArrayFromImage(seg_image)
origan = seg_arr==1
tumor = seg_arr==2
predict_origan = predict==1
predict_tumor = predict==2

inter_origan = np.sum(origan*predict_origan)
union_origan = np.sum(origan)+np.sum(predict_origan)
origan_dice = 2*inter_origan/union_origan
inter_tumor = np.sum(tumor*predict_tumor)
union_tumor = np.sum(tumor)+np.sum(predict_tumor)
tumor_dice = 2*inter_tumor/union_tumor

print("origan_dice:%f , tumor_dice:%f"%(origan_dice,tumor_dice))
