


import cv2
import utilty as util
import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
orgpath="./Input/Prof_data/or/"
srpath="./Input/Prof_data/scaled/"
PSNR_MEAN=[]
scale=2
for j, image in enumerate(sorted(glob.glob(orgpath+"*.png"))):
    name = os.path.basename(image)
    original = util.set_image_alignment(util.load_image(image,print_console=False), scale)
    filename, extension = os.path.splitext(name)
    original_y=util.convert_rgb_to_y(original,bit=16)
    #if not same:
    #    im=name.split('frame')[1]
    #    im2=im.split('.')[0]
    #    contrast = cv2.imread("Image/dog_sr/dog_160x120_"+im2+".png")
    #else:

    contrast = util.load_image(srpath+filename+'.png',print_console=False)
    contrast_y = util.convert_rgb_to_y(contrast, bit=16)
    contrast_y=util.resize_image_by_pil(contrast_y,scale=2)
    ps = peak_signal_noise_ratio(contrast_y, original_y, data_range=65535)

    #ps=cv2.PSNR(contrast2,original,65535)
    print("image {}: {}".format(j + 1, ps))
    #contrast2 = cv2.cvtColor(contrast2, cv2.COLOR_BGR2RGB)
    #save_image('Image/'+name,contrast2)
    PSNR_MEAN.append(ps)

print("MEAN PSNR: {}".format(np.mean(PSNR_MEAN)))
