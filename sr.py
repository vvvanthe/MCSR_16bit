import tensorflow as tf
from model import SR_QP2
import pathlib
import os
import glob
import numpy as np
import utilty as util
import time
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="1"




layer= np.array([48, 48, 24, 24, 24, 12, 12, 8],dtype=np.int)     #L8_48_8

super_reslution=SR_QP2(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=False,BatchNorm=False)
IMG_SHAPE = (None, None, 1)
inputs = tf.keras.Input(shape=IMG_SHAPE)
predictions=super_reslution(inputs)

full_model_22 = tf.keras.models.Model(inputs=inputs, outputs=predictions)

super_reslution2=SR_QP2(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=False,BatchNorm=False)
IMG_SHAPE = (None, None, 1)
inputs2 = tf.keras.Input(shape=IMG_SHAPE)
predictions2=super_reslution2(inputs2)



full_model_27 = tf.keras.models.Model(inputs=inputs2, outputs=predictions2)


super_reslution3=SR_QP2(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=False,BatchNorm=False)
IMG_SHAPE = (None, None, 1)
inputs3 = tf.keras.Input(shape=IMG_SHAPE)
predictions3=super_reslution3(inputs3)


full_model_32 = tf.keras.models.Model(inputs=inputs3, outputs=predictions3)

super_reslution4=SR_QP2(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=False,BatchNorm=False)
IMG_SHAPE = (None, None, 1)
inputs4 = tf.keras.Input(shape=IMG_SHAPE)
predictions4=super_reslution4(inputs4)
full_model_37 = tf.keras.models.Model(inputs=inputs4, outputs=predictions4)


super_reslution5=SR_QP2(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=False,BatchNorm=False)
IMG_SHAPE = (None, None, 1)
inputs5= tf.keras.Input(shape=IMG_SHAPE)
predictions5=super_reslution5(inputs5)
full_model_42 = tf.keras.models.Model(inputs=inputs5, outputs=predictions5)


super_reslution6=SR_QP2(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=False,BatchNorm=False)
IMG_SHAPE = (None, None, 1)
inputs6= tf.keras.Input(shape=IMG_SHAPE)
predictions6=super_reslution6(inputs6)
full_model_17 = tf.keras.models.Model(inputs=inputs6, outputs=predictions6)
full_model_37.summary()
#checkpoint22="checkpoints/BIV_QP22_48_8/cp-0070.ckpt"
checkpoint27="checkpoints/BIV_QP27_48_8/cp-0070.ckpt"
checkpoint32="checkpoints/BIV_QP32_48_8/cp-0070.ckpt"
checkpoint37="checkpoints/BIV_QP37_48_8/cp-0080.ckpt"
checkpoint42="checkpoints/BIV_QP42_48_8/cp-0070.ckpt"
#checkpoint47="checkpoints/BIV_QP47_L1_96/cp-0070.ckpt"


full_model_22.load_weights(checkpoint27)
full_model_27.load_weights(checkpoint32)
full_model_32.load_weights(checkpoint37)
full_model_37.load_weights(checkpoint42)

full_model_37.summary()

orgpath='dataset/01_AI12Off_allframes/*/*/*.png'
out_path='Output_VVC/01_AI12Off_L8_48_8/'
#out_path='Output_VVC/SR_VVC_bic_allframes/'

start = time.time()
preflag = 'A1-01'
flag = 'A1-01'
taken_time=[]
name_seq=[]
name_seq.append(preflag)
for i, image_path in enumerate(sorted(glob.glob(orgpath))):
    name = os.path.basename(image_path)
    path = os.path.dirname(image_path)
    folder_name1 = os.path.basename(path)
    path2 = os.path.dirname(path)
    folder_name2 = os.path.basename(path2)
    filename, extension = os.path.splitext(name)
    image = util.load_image(image_path, print_console=False)

    if 'rec17' == folder_name1:

        out = util.apply_SR(model=full_model_17, input_image=image, scale=2, bit=16)

    elif 'rec22' == folder_name1:


        out = util.apply_SR(model=full_model_22, input_image=image, scale=2, bit=16)

    elif 'rec27' == folder_name1:


        out = util.apply_SR(model=full_model_27, input_image=image, scale=2, bit=16)
    elif 'rec32' == folder_name1:

        out = util.apply_SR(model=full_model_32, input_image=image, scale=2, bit=16)
    elif 'rec37' == folder_name1:

        out = util.apply_SR(model=full_model_37, input_image=image, scale=2, bit=16)
    elif 'rec42' == folder_name1:

        out = util.apply_SR(model=full_model_42, input_image=image, scale=2, bit=16)
    else:
        print('ERROR')
        break
        


    #out = util.resize_image_by_pil(image, scale=2)
    path_out=out_path+folder_name2+'/'+folder_name1+'/'+name
    util.save_image(path_out, out)

    flag=folder_name2
    if preflag!=flag:
        taken_time.append(time.time()-start)
        start = time.time()
        preflag=flag
        name_seq.append(flag)






taken_time.append(time.time()-start)



for i in range(len(name_seq)):
    print("Taken time for {} is {}".format(name_seq[i],taken_time[i]))

os.system("python rgb2yuv_ffmpeg_vvc.py")