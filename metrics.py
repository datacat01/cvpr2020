import numpy as np


def match_img(local_hist1, global_hist1, local_hist2, global_hist2):
    resulting_sum = lambda arr1, arr2: np.sum(np.abs(arr1 - arr2))
    distance = resulting_sum(local_hist1, local_hist2) + \
               5 * resulting_sum(global_hist1, global_hist2)
    return distance