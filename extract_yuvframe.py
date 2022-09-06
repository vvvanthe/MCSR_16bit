
import os
import glob
import subprocess as sp
import shutil
inpath='dataset/VVC_AI_ORG/*.yuv'
outpath='dataset/VVC_AI_ORG/'
#size='1920x1080'
size='3840x2160'
FFMPEG_BIN="ffmpeg"
for i, video in enumerate(sorted(glob.glob(inpath))):
    name = os.path.basename(video)
    filename, extension = os.path.splitext(name)
    path = os.path.dirname(video)

    folder_name = os.path.basename(path)
    out_dir=outpath+folder_name+'/'+filename
    os.makedirs(out_dir, exist_ok=True)
    if True:#'rec17' == filename:
        command1 = [FFMPEG_BIN,
                    '-r', '1/1',
                    '-pix_fmt', 'yuv420p10le',
                    '-s', size,
                    '-i', video,
                    '-r', '1/1',
                    '-pix_fmt', 'rgb48be',
                    out_dir + '/output%03d.png']

        # pipe = sp.Popen(command1, stdout=sp.PIPE, bufsize=-1)
        # pipe.wait()
        os.system(
            'ffmpeg -r 1/1 -pix_fmt yuv420p10le -s {} -i {} -r 1/1 -pix_fmt rgb48be {}/output%03d.png'.format(size,
                                                                                                              video,
                                                                                                              out_dir))
    else:
        continue


