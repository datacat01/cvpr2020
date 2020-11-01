import os
import numpy as np
from motion_algos import blockMotion
import skvideo.io
import skvideo.datasets
import time

VID_NAME = 'video.mp4'
ENCODED_NAME = f'encoded_{VID_NAME[:-4]}.npz'
METHODS = ['4SS', '3SS', 'DS']


def write_time_metric(shape, method, t_enc, t_mot):
    time_metric_file = open('encoder_time.txt', 'a')
    time_metric_file.write(f'\n\nVID SHAPE: {shape}')
    time_metric_file.write(f'\nEncoder time for {method}: {t_enc}')
    time_metric_file.write(f'\nMotion vect computation time for {method}: {t_mot}')
    time_metric_file.write('\n-----------------------------------------------')


def encoder(videodata, method='DS', mbSize=8, p=2, compute_motion_if_exists=False):
    start_time_enc = time.time()

    encoded_name = method + '_' + ENCODED_NAME

    if not(compute_motion_if_exists) and os.path.isfile(encoded_name):
        data_calculated = np.load(encoded_name)
        return (data_calculated['video'], data_calculated['motion'])
    
    start_time_motion = time.time()
    motion = blockMotion(videodata, method=method, mbSize=mbSize, p=p)
    motion_time = time.time() - start_time_motion

    cut_vid = np.delete(videodata, list(range(0, videodata.shape[0], 2)), axis=0)

    encoder_time = time.time() - start_time_enc

    np.savez_compressed(encoded_name, video=cut_vid, motion=motion)

    write_time_metric(videodata.shape, method, encoder_time, motion_time)
    return (cut_vid, motion)



if __name__ == '__main__':
    videodata = skvideo.io.vread(VID_NAME)
    # videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
    print(videodata.shape)
    for method in METHODS:
        encoder(videodata, method=method, mbSize=6)
