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

VIDEO_PATH_1 = 'initial_output.mp4'
VIDEO_PATH_2 = 'result.mp4'

def capture_wideocam():
    cap = cv2.VideoCapture(0)

    frame_width = math.ceil(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = math.ceil(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_PATH_1, fourcc, 10.0, (frame_width, frame_height))

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


def play_vid(path=VIDEO_PATH_1):
    cap = cv2.VideoCapture(path)
    frame_counter = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('initial frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_rectangle():
    pass


if __name__ == "__main__":
    capture_wideocam()
    play_vid()