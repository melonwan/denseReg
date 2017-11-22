from __future__ import print_function, division, absolute_import
import os
import commands

# check the job id 
gpu_lock_path = '/tmp/lock-gpu*/info.txt'
lock_str = commands.getstatusoutput('cat %s'%gpu_lock_path)
lock_str = lock_str[1]
lock_str = lock_str.split('\n')


# on gpu server, use the gpu for tensorflow
if 'SGE_GPU' in os.environ:
    gpulist = []
    for line in lock_str:
        if line.find('wanc') == -1:
            continue
        line = line.split(' ')
        job_idx = int(line[7])
        gpu_idx = int(line[1])
        gpulist.append((gpu_idx, job_idx))
    gpulist = sorted(gpulist, key=lambda x:x[1])
    gpu_idx,job_idx = gpulist[-1]

    gpu_list = [gpu_idx]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in gpu_list)
    print('use GPU for tensorflow')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpu_list = []
    print('\x1b[0;31;47m use CPU for tensorflow \x1b[0m')

num_gpus = len(gpu_list)
print('available gpu list, ', gpu_list)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
config = tf.ConfigProto()
config.allow_soft_placement = True 
config.gpu_options.allow_growth = True 
