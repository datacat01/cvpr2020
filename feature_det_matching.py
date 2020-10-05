import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import timeit

class FeatureDetector:
    def __init__(self, src1, src2, n_features):
        self._srcs = [src1, src2]
        self._descriptors, self._keypoints = [], []
        self.__orb = cv.ORB_create(n_features, scoreType=cv.ORB_FAST_SCORE)
        self.time = [None]*2
    
    
    def __kp_des_for_single_img(self, img, img_idx):
        start_time = timeit.timeit()

        kp = self.__orb.detect(img, None)
        kp, des = self.__orb.compute(img, kp)

        end_time = timeit.timeit()
        self.time[img_idx] = end_time - start_time
        return (kp, des)
    
    
    def __detect_compute_if_empty(self):
        if len(self._keypoints) == 0:
            self.detect_and_compute_orb()
    
    
    def detect_and_compute_orb(self):
        # find the keypoints and descriptors with ORB
        det_comp = lambda src, idx : self.__kp_des_for_single_img(src, idx)
        for i in range(len(self._srcs)):
            kp, des = self.__kp_des_for_single_img(self._srcs[i], i)
            self._keypoints.append(kp)
            self._descriptors.append(des)
        return self._keypoints, self._descriptors
    
    
    def draw_keypoints(self, img_idx=0):
        self.__detect_compute_if_empty()
        
        kp_img = cv.drawKeypoints(self._srcs[img_idx],
                                  self._keypoints[img_idx],
                                  None, color=(0,255,0), flags=4)
        plt.imshow(kp_img)
        plt.show()



class FeatureMatcher(FeatureDetector):
    def __init__(self, src1, src2, n_features=1000):
        FeatureDetector.__init__(self, src1, src2, n_features)
        self._matches = None
        super().detect_and_compute_orb()
    
    
    def match(self, threshold=0.8):
        FLANN_INDEX_LSH = 6
        flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 2) #2
        flann = cv.FlannBasedMatcher(flann_params, {})
        matches = flann.knnMatch(self._descriptors[0], self._descriptors[1], k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        localization_error = 0 # похибка локалізації
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m, n) = m_n
            localization_error += m.distance
            if m.distance < threshold*n.distance:
                good.append(m)
        self._matches = good
        localization_error /= len(matches)
        return self._matches, localization_error
    

    def localize(self):
        if self._matches is None:
            self.match()

        ## extract the matched keypoints
        src_pts  = np.float32([self._keypoints[0][m.queryIdx].pt for m in self._matches]).reshape(-1,1,2)
        dst_pts  = np.float32([self._keypoints[1][m.trainIdx].pt for m in self._matches]).reshape(-1,1,2)

        ## find homography matrix and do perspective transform
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=4.0)
        matches_mask = mask.ravel().tolist()

        h,w = self._srcs[0].shape[:2]
        corners = np.float32([ [0,0], [0,h], [w,h], [w,0] ]).reshape(-1,1,2)

        scene_corners = cv.perspectiveTransform(corners, M)
        return scene_corners, matches_mask


    def draw_matches(self, scene_corners, matches_mask):
        img2 = cv.cvtColor(self._srcs[1], cv.COLOR_GRAY2BGR)
        img_test = cv.polylines(img2, [np.int32(scene_corners)], True, (255, 0, 0), 4, cv.LINE_AA)

        img3 = cv.drawMatches(self._srcs[0], self._keypoints[0],
                            img_test, self._keypoints[1],
                            self._matches, None)

        plt.imshow(img3)
        plt.show()
    
