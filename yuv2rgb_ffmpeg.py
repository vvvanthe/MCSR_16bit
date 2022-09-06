import glob
import os
import numpy as np
import subprocess as sp
FFMPEG_BIN="ffmpeg"
orgpath='data/SR_VVC/*/*.yuv'
out_path='data/SR_VVC_rgb/'
size='1920x1080'
os.makedirs(out_path, exist_ok=True)
for i, image in enumerate(sorted(glob.glob(orgpath))):

    name = os.path.basename(image)
    path = os.path.dirname(image)
    folder_name = os.path.basename(path)
    filename, extension = os.path.splitext(name)

    out_path_folder = out_path + folder_name
    os.makedirs(out_path_folder, exist_ok=True)
    file_out=out_path_folder+'/'+filename+'.png'
    print(image)
    command = [FFMPEG_BIN,
               '-r', '1/1',
               '-pix_fmt', 'yuv420p10le',
               '-s', size,
               '-i', image,
               '-r', '1/1',
               file_out]

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    pipe.wait()


