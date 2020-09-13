import cv2
import numpy as np


"""
    The extraction process of this color descriptor consists of four stages:
        - Image partitioning
        - Representative color selection
        - DCT transformation
            The discrete cosine transform (DCT) represents an image as a sum
            of sinusoids of varying magnitudes and frequencies. The DCT has
            the property that, for a typical image, most of the visually significant
            information about the image is concentrated in just a few coefficients
            of the DCT. For this reason, the DCT is often used in
            image compression applications.
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


    def __zigzag_value(self, i, j, n):
        # https://medium.com/100-days-of-algorithms/day-63-zig-zag-51a41127f31
        # upper side of interval
        if i + j >= n:
            return n * n - 1 - self.__zigzag_value(n - 1 - i, n - 1 - j, n)
        # lower side of interval
        k = (i + j) * (i + j + 1) // 2
        return k + i if (i + j) & 1 else k + j


    def _zigzag(self, array):
        n, m = array.shape[0], array.shape[1]
        zigzag_order = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                zigzag_order[i, j] = self.__zigzag_value(i, j, n)
        
        print(array.shape)
        print(zigzag_order.shape)
        print(zigzag_order)

        zigzag_scanning_res = array[zigzag_order]
        
        return zigzag_scanning_res


    def descript(self):
        img_h, img_w, _ = self.img.shape
        block_h, block_w = int(img_h/self.rows), int(img_w/self.cols)
        representative_colors = np.empty([self.rows, self.cols, 3])

        for row in range(self.rows):
            for col in range(self.cols):
                # image partitioning
                img_block = self.img[block_h*row : block_h*(row+1), block_w*col : block_w*(col+1)]
                # representative color selection
                average_color_by_row = np.mean(img_block, axis=0)
                average_color_by_rgb = np.round(np.mean(average_color_by_row, axis=0))
                representative_colors[row, col, :] = average_color_by_rgb

        # to YCbCr
        tiny_img_icon = cv2.cvtColor(np.float32(representative_colors), cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(tiny_img_icon)
        # DCT transformation
        perform_dct = lambda x: cv2.dct(np.float32(x))
        transformed = [perform_dct(y), perform_dct(cb), perform_dct(cr)]
        # Zigzag scanning
        res = [self._zigzag(i) for i in transformed]

        return


if __name__ == "__main__":
    cld = ColorLayoutDescriptor(cv2.imread("test_img.jpg"))
    descriptor_result = cld.descript()
    print(descriptor_result)