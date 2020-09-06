import cv2
import math

"""
    Встановити OpenCV, зчитати зображення з вебки, відобразити в першому віконці
    та записати його на диск. Після цього зчитати щойно записане зображення з диску,
    конвертувати у відтінки сірого та намалювати на ньому довільних кольорів лінію
    та прямокутник (наприклад червону лінію та синій прямокутник) і відобразити
    у другому віконці. Ні це не психотест, для дебагу це ще й як знадобиться. Бонуси
    за виконання цих кроків для відеоряду і бонуси до бонусів якщо в результаті цих
    кроків замість звалища картинок матимемо відеофайл (наприклад .avi).
"""

def capture_wideocam():

    cap = cv2.VideoCapture(0)

    frame_width = math.ceil(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = math.ceil(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('initial_output.mp4', fourcc, 10.0, (frame_width, frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read() # If frame is read correctly, ret will be True.
        if not ret:
            break

        out.write(frame)
        cv2.imshow('Frame. Press q to exit.',frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def read_vid(path):
    pass


def draw_rectangle():
    pass


if __name__ == "__main__":
    capture_wideocam()