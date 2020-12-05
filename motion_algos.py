import numpy as np
import os

import skvideo.utils


# we don't need motion data between every 2 frames, so we decided to redefine blockMotion function
# private methods of skvideo.motion were not importable, so I had to copy them too from the
# http://www.scikit-video.org/stable/_modules/skvideo/motion/block.html#blockMotion

def costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))


def minCost(costs):
    h, w = costs.shape
    mi = costs[np.int((h - 1) / 2), np.int((w - 1) / 2)]
    dy = np.int((h - 1) / 2)
    dx = np.int((w - 1) / 2)

    for i in range(h):
        for j in range(w):
            if costs[i, j] < mi:
                mi = costs[i, j]
                dy = i
                dx = j

    return dx, dy, mi


def checkBounded(xval, yval, w, h, mbSize):
    if ((yval < 0) or
            (yval + mbSize >= h) or
            (xval < 0) or
            (xval + mbSize >= w)):
        return False
    else:
        return True


def DS(imgP, imgI, mbSize, p):
    h, w = imgP.shape

    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((9)) * 65537

    L = np.floor(np.log2(p + 1))

    LDSP = []
    LDSP.append([0, -2])
    LDSP.append([-1, -1])
    LDSP.append([1, -1])
    LDSP.append([-2, 0])
    LDSP.append([0, 0])
    LDSP.append([2, 0])
    LDSP.append([-1, 1])
    LDSP.append([1, 1])
    LDSP.append([0, 2])

    SDSP = []
    SDSP.append([0, -1])
    SDSP.append([-1, 0])
    SDSP.append([0, 0])
    SDSP.append([1, 0])
    SDSP.append([0, 1])

    computations = 0

    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[4] = costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            cost = 0
            point = 4
            if costs[4] != 0:
                computations += 1
                for k in range(9):
                    refBlkVer = y + LDSP[k][1]
                    refBlkHor = x + LDSP[k][0]
                    if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    if k == 4:
                        continue
                    costs[k] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                       imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations += 1

                point = np.argmin(costs)
                cost = costs[point]

            SDSPFlag = 1
            if point != 4:
                SDSPFlag = 0
                cornerFlag = 1
                if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                    cornerFlag = 0
                xLast = x
                yLast = y
                x = x + LDSP[point][0]
                y = y + LDSP[point][1]
                costs[:] = 65537
                costs[4] = cost

            while SDSPFlag == 0:
                if cornerFlag == 1:
                    for k in range(9):
                        refBlkVer = y + LDSP[k][1]
                        refBlkHor = x + LDSP[k][0]
                        if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        if k == 4:
                            continue

                        if ((refBlkHor >= xLast - 1) and
                                (refBlkHor <= xLast + 1) and
                                (refBlkVer >= yLast - 1) and
                                (refBlkVer <= yLast + 1)):
                            continue
                        elif ((refBlkHor < j - p) or
                              (refBlkHor > j + p) or
                              (refBlkVer < i - p) or
                              (refBlkVer > i + p)):
                            continue
                        else:
                            costs[k] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                               imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1
                else:
                    lst = []
                    if point == 1:
                        lst = np.array([0, 1, 3])
                    elif point == 2:
                        lst = np.array([0, 2, 5])
                    elif point == 6:
                        lst = np.array([3, 6, 8])
                    elif point == 7:
                        lst = np.array([5, 7, 8])

                    for idx in lst:
                        refBlkVer = y + LDSP[idx][1]
                        refBlkHor = x + LDSP[idx][0]
                        if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        elif ((refBlkHor < j - p) or
                              (refBlkHor > j + p) or
                              (refBlkVer < i - p) or
                              (refBlkVer > i + p)):
                            continue
                        else:
                            costs[idx] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                                 imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1

                point = np.argmin(costs)
                cost = costs[point]

                SDSPFlag = 1
                if point != 4:
                    SDSPFlag = 0
                    cornerFlag = 1
                    if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                        cornerFlag = 0
                    xLast = x
                    yLast = y
                    x += LDSP[point][0]
                    y += LDSP[point][1]
                    costs[:] = 65537
                    costs[4] = cost
            costs[:] = 65537
            costs[2] = cost

            for k in range(5):
                refBlkVer = y + SDSP[k][1]
                refBlkHor = x + SDSP[k][0]

                if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                    continue
                elif ((refBlkHor < j - p) or
                      (refBlkHor > j + p) or
                      (refBlkVer < i - p) or
                      (refBlkVer > i + p)):
                    continue

                if k == 2:
                    continue

                costs[k] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                   imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                computations += 1

            point = 2
            cost = 0
            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]

            x += SDSP[point][0]
            y += SDSP[point][1]

            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [x - j, y - i]

            costs[:] = 65537

    return vectors, computations / ((h * w) / mbSize ** 2)


# Three step search
def ThreeSS(imgP, imgI, mbSize, p):
    h, w = imgP.shape

    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((3, 3), dtype=np.float32) * 65537

    computations = 0

    L = np.floor(np.log2(p + 1))
    stepMax = np.int(2 ** (L - 1))

    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i

            costs[1, 1] = costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            computations += 1

            stepSize = stepMax

            while (stepSize >= 1):
                for m in range(-stepSize, stepSize + 1, stepSize):
                    for n in range(-stepSize, stepSize + 1, stepSize):
                        refBlkVer = y + m
                        refBlkHor = x + n
                        if ((refBlkVer < 0) or
                                (refBlkVer + mbSize > h) or
                                (refBlkHor < 0) or
                                (refBlkHor + mbSize > w)):
                            continue
                        costRow = np.int(m / stepSize) + 1
                        costCol = np.int(n / stepSize) + 1
                        if ((costRow == 1) and (costCol == 1)):
                            continue
                        costs[costRow, costCol] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                                          imgI[refBlkVer:refBlkVer + mbSize,
                                                          refBlkHor:refBlkHor + mbSize])
                        computations = computations + 1
                dx, dy, mi = minCost(costs)  # finds which macroblock in imgI gave us min Cost
                x += (dx - 1) * stepSize
                y += (dy - 1) * stepSize

                stepSize = np.int(stepSize / 2)
                costs[1, 1] = costs[dy, dx]
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [y - i, x - j]

            costs[:, :] = 65537

    return vectors, computations / ((h * w) / mbSize ** 2)


def FourSS(imgP, imgI, mbSize, p):
    # Computes motion vectors using Four Step Search method
    #
    # Input
    #   imgP : The image for which we want to find motion vectors
    #   imgI : The reference image
    #   mbSize : Size of the macroblock
    #   p : Search parameter  (read literature to find what this means)
    #
    # Ouput
    #   motionVect : the motion vectors for each integral macroblock in imgP
    #   SS4computations: The average number of points searched for a macroblock
    h, w = imgP.shape

    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((3, 3), dtype=np.float32) * 65537

    computations = 0
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i

            costs[1, 1] = costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            computations += 1

            for m in range(-2, 3, 2):
                for n in range(-2, 3, 2):
                    refBlkVer = y + m  # row/Vert co-ordinate for ref block
                    refBlkHor = x + n  # col/Horizontal co-ordinate

                    if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue

                    costRow = np.int(m / 2 + 1)
                    costCol = np.int(n / 2 + 1)
                    if ((costRow == 1) and (costCol == 1)):
                        continue
                    costs[costRow, costCol] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                                      imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations = computations + 1
            dx, dy, mi = minCost(costs)  # finds which macroblock in imgI gave us min Cost

            flag_4ss = 0
            if (dx == 1 and dy == 1):
                flag_4ss = 1
            else:
                xLast = x
                yLast = y
                x += (dx - 1) * 2
                y += (dy - 1) * 2

            costs[:, :] = 65537
            costs[1, 1] = mi

            stage = 1

            while (flag_4ss == 0 and stage <= 2):
                for m in range(-2, 3, 2):
                    for n in range(-2, 3, 2):
                        refBlkVer = y + m
                        refBlkHor = x + n
                        if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue

                        if ((refBlkHor >= xLast - 2) and
                                (refBlkHor <= xLast + 2) and
                                (refBlkVer >= yLast - 2) and
                                (refBlkVer >= yLast + 2)):
                            continue

                        costRow = np.int(m / 2) + 1
                        costCol = np.int(n / 2) + 1

                        if (costRow == 1 and costCol == 1):
                            continue

                        costs[costRow, costCol] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                                          imgI[refBlkVer:refBlkVer + mbSize,
                                                          refBlkHor:refBlkHor + mbSize])

                        computations += 1
                dx, dy, mi = minCost(costs)  # finds which macroblock in imgI gave us min Cost

                if (dx == 1 and dy == 1):
                    flag_4ss = 1
                else:
                    flag_4ss = 0
                    xLast = x
                    yLast = y
                    x = x + (dx - 1) * 2
                    y = y + (dy - 1) * 2

                costs[:, :] = 65537
                costs[1, 1] = mi
                stage += 1

            for m in range(-1, 2):
                for n in range(-1, 2):
                    refBlkVer = y + m
                    refBlkHor = x + n

                    if not checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    costRow = m + 1
                    costRow = n + 1
                    if (costRow == 2 and costCol == 2):
                        continue
                    costs[costRow, costCol] = costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                                      imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])

                    computations += 1

            dx, dy, mi = minCost(costs)  # finds which macroblock in imgI gave us min Cost

            x += dx - 1
            y += dy - 1

            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [y - i, x - j]

            costs[:, :] = 65537

    return vectors, computations / ((h * w) / mbSize ** 2)


def blockMotion(videodata, method='DS', mbSize=8, p=2):
    videodata = skvideo.utils.vshape(videodata)
    luminancedata = skvideo.utils.rgb2gray(videodata)

    numFrames, height, width, channels = luminancedata.shape
    assert numFrames > 1, "Must have more than 1 frame for motion estimation!"

    luminancedata = luminancedata.reshape((numFrames, height, width))

    motion_n = numFrames // 2
    if numFrames % 2 != 0:
        motion_n += 1
    motionData = np.zeros((motion_n, np.int(height / mbSize), np.int(width / mbSize), 2), np.int8)
    # print('motion data', motionData.shape)
    if method == "4SS":
        for append_idx, i in enumerate(range(0, numFrames - 1, 2)):
            motion, comps = FourSS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[append_idx, :, :, :] = motion
    elif method == "3SS":
        for append_idx, i in enumerate(range(0, numFrames - 1, 2)):
            motion, comps = ThreeSS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[append_idx, :, :, :] = motion
    elif method == "DS":
        for append_idx, i in enumerate(range(0, numFrames - 1, 2)):
            motion, comps = DS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[append_idx, :, :, :] = motion
    else:
        raise NotImplementedError

    return motionData

def blockComp(framedata, motionVect, mbSize):
    M, N, C = framedata.shape

    compImg = np.zeros((M, N, C))

    for i in range(0, M - mbSize + 1, mbSize):
        for j in range(0, N - mbSize + 1, mbSize):
            dy = motionVect[np.int(i / mbSize), np.int(j / mbSize), 0]
            dx = motionVect[np.int(i / mbSize), np.int(j / mbSize), 1]

            refBlkVer = i + dy
            refBlkHor = j + dx

            # check bounds
            if not checkBounded(refBlkHor, refBlkVer, N, M, mbSize):
                continue

            compImg[i:i + mbSize, j:j + mbSize, :] = framedata[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize, :]
    return compImg