import cv2
import numpy as np
import math
import random

"""
    Встановити OpenCV, зчитати зображення з вебки, відобразити в першому віконці
    та записати його на диск. Після цього зчитати щойно записане зображення з диску,
    конвертувати у відтінки сірого та намалювати на ньому довільних кольорів лінію
    та прямокутник (наприклад червону лінію та синій прямокутник) і відобразити
    у другому віконці. Ні це не психотест, для дебагу це ще й як знадобиться. Бонуси
    за виконання цих кроків для відеоряду і бонуси до бонусів якщо в результаті цих
    кроків замість звалища картинок матимемо відеофайл (наприклад .avi).
"""

VIDEO_PATH_1 = 'initial_video.mp4'
VIDEO_PATH_2 = 'result.mp4'
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 10.0

get_height = lambda capture: math.ceil(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
get_width = lambda capture: math.ceil(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
rand_color = lambda: tuple(np.random.choice(range(256), 3).tolist())


def rand_coord(max_width, max_height, st=0): 
    return (random.randint(st, max_width), random.randint(st, max_height))


def capture_wideocam():
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter(VIDEO_PATH_1, FOURCC, FPS, (get_width(cap), get_height(cap)))

    while(cap.isOpened()):
        ret, frame = cap.read() # If frame is read correctly, ret will be True.
        if not ret:
            break

        out.write(frame)
        cv2.imshow('Press q to stop the recording.',frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def draw_rectangle(path=VIDEO_PATH_1):
    cap = cv2.VideoCapture(path)
    frame_width, frame_height = get_width(cap), get_height(cap)
    out = cv2.VideoWriter(VIDEO_PATH_2, FOURCC, FPS, (frame_width, frame_height))

    colors = [rand_color() for i in range(2)]
    line_coord = [rand_coord(frame_width, frame_height) for i in range(2)]
    rect_coord = [rand_coord(frame_width//2, frame_height//2)]
    rect_coord.append(
        (rect_coord[0][0] + random.randint(0, frame_width//2),
        rect_coord[0][1] + random.randint(0, frame_height//2))
        )

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray_one_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_three_channels = cv2.cvtColor(gray_one_channel, cv2.COLOR_GRAY2BGR)

        cv2.line(gray_three_channels, line_coord[0], line_coord[1], colors[0], 5)
        cv2.rectangle(gray_three_channels, rect_coord[0], rect_coord[1], colors[1], 5)

        cv2.imshow('video result (q to exit)', gray_three_channels)
        cv2.imshow('video original (q to exit)', frame)

        out.write(gray_three_channels)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_wideocam()
    draw_rectangle()