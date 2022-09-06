import glob
import os
import numpy as np
import subprocess as sp
import utilty as util
FFMPEG_BIN="ffmpeg"
orgpath='Output_VVC/AI_SR_allframes/*/*/*.png'
out_path='Output_VVC/AI_SR_YUV_each/'
os.makedirs(out_path, exist_ok=True)
size='3840x2160'
#size='1920x1080'
bit=10
up_bic=False
for i, image in enumerate(sorted(glob.glob(orgpath))):
    name = os.path.basename(image)
    path = os.path.dirname(image)
    folder_name = os.path.basename(path)
    print(folder_name.split('AI_SR_allframes')[2])
    out_path_folder = out_path + folder_name
    os.makedirs(out_path_folder, exist_ok=True)