import cv2
import numpy as np


"""
    The extraction process of this color descriptor consists of four stages:
    - Image partitioning
    - Representative color selection
    - DCT transformation
    - Zigzag scanning
        The purpose of the zigzag is to convert the 2D array of quantized DCT
        coefficients into a 1D array, where the first elements come from the
        upper left, and later elements come from the lower right, of the 2D array
"""


class ColorLayoutDescriptor:

    def __init__(self, img):
        self.img = img
        self.rows = 8
        self.cols = 8


    def _zigzag(self, dct_array):
        pass


    def descript(self):
        img_h, img_w, _ = self.img.shape
        block_h, block_w = int(img_h/self.rows), int(img_w/self.cols)
        representative_colors = np.empty([img_h, img_w, 3])

        for row in range(self.rows):
            for col in range(self.cols):
                # image partitioning
                img_block = self.img[block_h*row : block_h*(row+1), block_w*col : block_w*(col+1)]
                #representative color selection
                average_color_by_row = np.mean(img_block, axis=0)
                average_color_by_rgb = np.round(np.mean(average_color_by_row, axis=0))
                representative_colors[row, col, :] = average_color_by_rgb

        # to YCbCr
        tiny_img_icon = cv2.cvtColor(np.float32(representative_colors), cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(tiny_img_icon)
        # DCT
        perform_dct = lambda x: cv2.dct(np.float32(x))
        transformed = [perform_dct(y), perform_dct(cb), perform_dct(cr)]
        # Zigzag
        res = [self._zigzag(i) for i in transformed]

        return res


if __name__ == "__main__":
    cld = ColorLayoutDescriptor(cv2.imread("test_img.jpg"))
    descriptor_result = cld.descript()
    # print(descriptor_result)