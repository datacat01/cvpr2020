import numpy as np
import math
import itertools
import cv2

class EdgeHistogramDescriptor:
    
    def __init__(self, img, threshold=0.1):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.threshold = threshold
        self.rows, self.cols = 4, 4

        self.__block_h = int(self.img.shape[0]/self.rows)
        self.__block_w = int(self.img.shape[1]/self.cols)

        self.__sub_block_i = int(self.__block_h/2)
        self.__sub_block_j = int(self.__block_w/2)
    

    def __filter(self, blocks):
        # apply edge filters to 2x2 sub-blocks of a block
        sqrt2 = math.sqrt(2)
        edge_strengths = []
        edge_masks = {
            'vertical_edge': np.array([1, -1, 1, -1]),
            'horizontal_edge': np.array([1, 1, -1, -1]),
            'dia_edge_45': np.array([sqrt2, 0, 0, -sqrt2]),
            'dia_edge_135': np.array([0, sqrt2, -sqrt2, 0]),
            'nond_edge': np.array([2, -2, -2, 2])
        }

        for block_22 in blocks:
            s = []
            for mask in edge_masks.values():
                s.append(np.abs(np.sum(block_22 * mask)))
            edge_strengths.append(s)
        
        return edge_strengths
    

    def __form_hist(self, edge_strengths):
        bin_block = [0]*5

        for e in edge_strengths:
            m_idx = np.argmax(e)
            if e[m_idx] > self.threshold:
                bin_block[m_idx] += 1

        return bin_block


    def __get_smallest_blocks(self, img_block):
        # divide into 2x2 sub-blocks and calc avg grey
        smallest_blocks = []

        for i in range(self.__sub_block_i):
            for j in range(self.__sub_block_j):
                block_22 = img_block[2*i : 2*(i+1), 2*i : 2*(i+1)]
                smallest_blocks.append(list(itertools.chain.from_iterable(block_22)))

        return smallest_blocks
    

    def descript(self):
        hist = []

        for row in range(self.rows):
            for col in range(self.cols):
                bin_for_block = [0]*5
                img_block = self.img[self.__block_h*row : self.__block_h*(row+1), \
                                     self.__block_w*col : self.__block_w*(col+1)]
                blocks_22 = self.__get_smallest_blocks(img_block)
                edges = self.__filter(blocks_22)
                bin_for_block = self.__form_hist(edges)
                hist.append(bin_for_block)
        
        hist = np.array(hist).astype(int)
        global_hist = np.mean(hist, axis=0)
        return hist, global_hist




if __name__ == "__main__":
    ehd = EdgeHistogramDescriptor(cv2.imread("vertical-lines.jpg"))
    histogram, global_histogram = ehd.descript()
    print(global_histogram)