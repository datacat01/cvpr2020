import os
import numpy as np
import pandas as pd
from motion_algos import blockMotion, blockComp
import skvideo.datasets, skvideo.io


VID_NAME = 'video.mp4'
ENCODED_NAME = f'encoded_{VID_NAME[:-4]}.npz'
MBSIZE = 8
P = 2


def encoder(videodata, method='3SS', mbSize=MBSIZE, p=P, compute_motion_if_exists=False):
    encoded_name = method + '_' + ENCODED_NAME

    if not compute_motion_if_exists and os.path.isfile(encoded_name):
        data_calculated = np.load(encoded_name)
        return data_calculated['motion']

    motion = blockMotion(videodata, method=method, mbSize=mbSize, p=p)
    np.savez_compressed(encoded_name, motion=motion)

    return motion 


def compute_motion_vec_len(mot_vec_matrix):
    vector_norms_framevise = []
    for mot_frame_idx in range(mot_vec_matrix.shape[0]):
        mot_frame = mot_vec_matrix[mot_frame_idx, :, :, :]
        vector_norms_framevise.append(np.linalg.norm(mot_frame, axis=2))
    
    return np.array(vector_norms_framevise)


def filter_slow_or_static(vector_norms, min_criterion=1.2):
    # creates a mask, where True (1) - satisfies blur criteria, False (0) - do not
    # creterion: motion >= min motion in frame * min_criterion if min motion != 0
    # else, take second min and do the same
    mask = np.full_like(vector_norms, False)
    for mot_frame_idx in range(len(vector_norms)):
        mot_norm_frame = vector_norms[mot_frame_idx, :, :]
        mot_min = np.min(mot_norm_frame)
        if mot_min == 0:
            mot_min = np.partition(mot_norm_frame, 1)[1]
        mask[mot_frame_idx, :, :][np.where(mot_norm_frame>=mot_min*(min_criterion))] = True

    return mask


def blur_block(block):
    pass


def blur_vid(videodata, mask):
    pass


if __name__=="__main__":
    videodata = skvideo.io.vread(VID_NAME)
    print(f'vid shape - {videodata.shape}')
    mot_vec = encoder(videodata)
    print(f'mot vec shapes - {mot_vec.shape}')
    vector_norms = compute_motion_vec_len(mot_vec)
    print('describe verc norms:')
    print(np.min(vector_norms), np.max(vector_norms), np.mean(vector_norms), np.std(vector_norms))
    print(f'vect norms shapes - {vector_norms.shape}')
    mask = filter_slow_or_static(vector_norms)
    print(f'mask shape - {mask.shape}')