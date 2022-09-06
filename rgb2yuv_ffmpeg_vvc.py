import glob
import os
import numpy as np
import subprocess as sp
import utilty as util
FFMPEG_BIN="ffmpeg"
orgpath='Output_VVC/01_AI12Off_L8_48_8/*/*'
out_path='../PSNR/VVC_Test/01_AI12Off_L8_48_8/'
os.makedirs(out_path, exist_ok=True)
size='3840x2160'
#size='1920x1080'
bit=10
up_bic=False
for i, image in enumerate(sorted(glob.glob(orgpath))):
    name = os.path.basename(image)
    path = os.path.dirname(image)
    folder_name = os.path.basename(path)


    out_path_folder = out_path + folder_name
    os.makedirs(out_path_folder, exist_ok=True)


    file_out_mp4=out_path_folder + '/' + name + '.mp4'


    ## 10bit
    if bit==10:
        key='yuv420p10le'
    else:
        key='yuv420p'

    command1 = [FFMPEG_BIN,
                '-i', image+'/output%03d.png',
                '-c:v', 'libx265',
                '-x265-params', 'lossless=1',
                '-framerate', '1/1',
                #'-s', size,
                '-pix_fmt', key,
               file_out_mp4]


    #pipe = sp.Popen(command1, stdout=sp.PIPE, bufsize=-1)
    #pipe.wait()
    file_out_yuv = out_path_folder +'/' + name + '.yuv'

    command2 = [FFMPEG_BIN,
                '-i', file_out_mp4,
                file_out_yuv]
    #pipe = sp.Popen(command2, stdout=sp.PIPE, bufsize=-1)
    #pipe.wait()

    os.system(
        'ffmpeg -i {}/output%03d.png -c:v libx265 -x265-params lossless=1 -framerate 1/1 -s {} -pix_fmt {} {}'.format(
            image, size,key, file_out_mp4))

    #os.system(
    #    'ffmpeg -i {}/output001.png -c:v libx265 -x265-params lossless=1 -framerate 1/1 -s {} -pix_fmt {} {}'.format(
    #        image, size, key, file_out_mp4))



    os.system(
        'ffmpeg -i {} {}'.format(file_out_mp4, file_out_yuv))
    os.remove(file_out_mp4)

