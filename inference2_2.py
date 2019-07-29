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
image_reg = "/home/zkq/Project/Data/kits19/train_data/data/case_*/imaging.nii.gz"
label_reg = "/home/zkq/Project/Data/kits19/train_data/data/case_*/segmentation.nii.gz"

image_files = glob.glob(image_reg)
label_files = glob.glob(label_reg)

f = h5py.File("/home/zkq/Project/Data/kits19/train_data/groups.h5", "r")
groups = f["test"].value
test_idex = groups.reshape(-1)
f.close()
removes = ['case_00025', 'case_00117', 'case_00160', 'case_00061', 'case_00015']
frame = pd.DataFrame(columns=["kidney_dice","tumor_dice","avg_kidney_and_tumor"])

Avg_kidney_DiceRatio = 0
Max_kidney_DiceRatio = 0
Min_kidney_DiceRatio = 1
Avg_tumor_DiceRatio = 0
Max_tumor_DiceRatio = 0
Min_tumor_DiceRatio = 1
Avg_kidney_and_tumor_DiceRatio = 0
Max_kidney_and_tumor_DiceRatio = 0
Min_kidney_and_tumor_DiceRatio = 1



def need_remove(file, removes):
    for remove in removes:
        if (remove in file):
            return True


image_files = sorted(image_files, key=lambda item: int(com.findall(item)[-1]))
label_files = sorted(label_files, key=lambda item: int(com.findall(item)[-1]))

test_image_files = [image_files[i] for i in test_idex]
test_label_files = [label_files[i] for i in test_idex]

test_image_files = [item for item in test_image_files if not need_remove(item, removes)]
test_label_files = [item for item in test_label_files if not need_remove(item, removes)]

len_test = len(test_image_files)
for image_file_name,seg_file_name in zip(test_image_files,test_label_files):
    keyname=image_file_name.split("/")[-2]
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
    predict = sitk.GetArrayFromImage(predict_image)
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
    mean_origan_tumor = (origan_dice + tumor_dice) / 2
    frame.loc[keyname] = [origan_dice, tumor_dice, mean_origan_tumor]
    print('test ' + str(keyname) + ' kidney_DiceRatio:' + str(origan_dice) + "; tumor_DiceRatio:" + str(
        tumor_dice) + "; kidney_and_tumor_DiceRatio:" + str(mean_origan_tumor))
    Avg_kidney_DiceRatio += origan_dice / len_test
    Max_kidney_DiceRatio = max(Max_kidney_DiceRatio, origan_dice)
    Min_kidney_DiceRatio = min(Min_kidney_DiceRatio, origan_dice)
    Avg_tumor_DiceRatio += tumor_dice / len_test
    Max_tumor_DiceRatio = max(Max_tumor_DiceRatio, tumor_dice)
    Min_tumor_DiceRatio = min(Min_tumor_DiceRatio, tumor_dice)
    Avg_kidney_and_tumor_DiceRatio += mean_origan_tumor / len_test
    Max_kidney_and_tumor_DiceRatio = max(Max_kidney_and_tumor_DiceRatio, mean_origan_tumor)
    Min_kidney_and_tumor_DiceRatio = min(Min_kidney_and_tumor_DiceRatio, mean_origan_tumor)
    print("origan_dice:%f , tumor_dice:%f"%(origan_dice,tumor_dice))
frame.loc["Max"] = [Max_kidney_DiceRatio,Max_tumor_DiceRatio,Max_kidney_and_tumor_DiceRatio]
frame.loc["Min"] = [Min_kidney_DiceRatio,Min_tumor_DiceRatio,Min_kidney_and_tumor_DiceRatio]
frame.loc["Avg"] = [Avg_kidney_DiceRatio,Avg_tumor_DiceRatio,Avg_kidney_and_tumor_DiceRatio]

print('test ' + 'Max' + ' kidney_DiceRatio:' + str(Max_kidney_DiceRatio) + "; tumor_DiceRatio:" + str(Max_tumor_DiceRatio)+ "; kidney_and_tumor_DiceRatio:" + str(Max_kidney_and_tumor_DiceRatio))
print('test ' + 'Min' + ' kidney_DiceRatio:' + str(Min_kidney_DiceRatio) + "; tumor_DiceRatio:" + str(Min_tumor_DiceRatio)+ "; kidney_and_tumor_DiceRatio:" + str(Min_kidney_and_tumor_DiceRatio))
print('test ' + 'Avg' + ' kidney_DiceRatio:' + str(Avg_kidney_DiceRatio) + "; tumor_DiceRatio:" + str(Avg_tumor_DiceRatio)+ "; kidney_and_tumor_DiceRatio:" + str(Avg_kidney_and_tumor_DiceRatio))

#frame.to_csv("./test_result17500.csv")
frame.to_csv("./unetpp2_test_result.csv")
