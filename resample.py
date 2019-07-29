import SimpleITK as sitk
import numpy as np
def Resample(input_image,output_spacing,output_size=None,interpolator="Linear",type = None):
    static = sitk.StatisticsImageFilter()
    resample = sitk.ResampleImageFilter()
    if(interpolator=="NearestNeighbor"):
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif(interpolator=="Linear"):
        resample.SetInterpolator(sitk.sitkLinear)
    static.Execute(input_image)
    spacing = input_image.GetSpacing()
    input_size = input_image.GetSize()
    if(output_size==None):
        outputsize = []
        for i in range(len(input_size)):
            outputsize.append(int(np.ceil(input_size[i]*spacing[i]/output_spacing[i])))
    else:
        outputsize = output_size
    min_data = static.GetMinimum()
    resample.SetReferenceImage(input_image)
    resample.SetDefaultPixelValue(min_data)
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(outputsize)
    if(type!=None):
        resample.SetOutputPixelType(type)
    output_image = resample.Execute(input_image)
    return output_image
