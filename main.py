import os
import numpy as np
import cv2 as cv
import pandas as pd
from feature_det_matching import FeatureMatcher

# IMG_FOLDER_TRAIN = 'goose_cup_train/'
# IMG_FOLDER_TEST = 'goose_cup/'
# IMG_FOLDER_TRAIN = 'book_cpp_train/'
# IMG_FOLDER_TEST = 'book_cpp/'
IMG_FOLDER_TRAIN = 'clock_train/'
IMG_FOLDER_TEST = 'clock/'


def go_through_img_and_compute():
    train_imgs = []
    for train_img_name in os.listdir(IMG_FOLDER_TRAIN):
        if train_img_name[-3:] != 'jpg':
            continue
        train_imgs.append(cv.imread(IMG_FOLDER_TRAIN+train_img_name, cv.IMREAD_GRAYSCALE))
    
    img_names = []
    scores = {
            'm1': [], # відносна к-сть прав суміщ ознак
            'm2': [], # похибка локалізації
            'm3': [], # час
            'width_plus_height': []
        }
    for idx, test_img_name in enumerate(os.listdir(IMG_FOLDER_TEST)):
        if test_img_name[-3:] != 'jpg':
            continue

        print(f'{idx+1}. {test_img_name}')

        test_img = cv.imread(IMG_FOLDER_TEST+test_img_name)

        m1_local, m2_local, m3_local = [], [], []
        for train_img in train_imgs:
            fm = FeatureMatcher(train_img, test_img)
            time = fm.time
            matches, m2 = fm.match()

            m1_local.append(len(matches) / len(fm._keypoints[0]))
            m2_local.append(m2)
            m3_local.append([time[1], int(np.sum(test_img.shape[:2]))])
        
        max_score_idx = np.argmax(m1_local)
        scores['m1'].append(m1_local[max_score_idx])
        scores['m2'].append(m2_local[max_score_idx])
        scores['m3'].append(m3_local[max_score_idx][0])
        scores['width_plus_height'].append(m3_local[max_score_idx][1])
        img_names.append(test_img_name[:-4])
    return img_names, scores



if __name__ == '__main__':
    names, scores = go_through_img_and_compute()
    scores_df = pd.DataFrame(scores, index=names)
    scores_df.to_csv('clock_metrics.csv')