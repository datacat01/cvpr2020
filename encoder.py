import os
import numpy as np
from motion_algos import blockMotion
import skvideo.io

VID_NAME = 'video.mp4'
ENCODED_NAME = f'encoded_{VID_NAME[:-4]}.npz'
METHODS = ['4SS', '3SS', 'DS']


def encoder(videodata, method='DS', mbSize=8, p=2, compute_motion_if_exists=False):

    encoded_name = method + '_' + ENCODED_NAME

    if not(compute_motion_if_exists) and os.path.isfile(encoded_name):
        data_calculated = np.load(encoded_name)
        return (data_calculated['video'], data_calculated['motion'])
    
    motion = blockMotion(videodata, method=method, mbSize=mbSize, p=p)
    cut_vid = np.delete(videodata, list(range(0, videodata.shape[0], 2)), axis=0)

    np.savez_compressed(encoded_name, video=cut_vid, motion=motion)

    return (cut_vid, motion)



if __name__ == '__main__':
    videodata = skvideo.io.vread(VID_NAME)
    for method in METHODS:
        encoder(videodata, method=method)
