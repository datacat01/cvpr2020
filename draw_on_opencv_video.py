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
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

get_height = lambda capture: math.ceil(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
get_width = lambda capture: math.ceil(capture.get(cv2.CAP_PROP_FRAME_WIDTH))


def capture_wideocam():
    cap = cv2.VideoCapture(0)

    out = cv2.VideoWriter(VIDEO_PATH_1, FOURCC, 10.0, (get_width(cap), get_height(cap)))

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
    out = cv2.VideoWriter(VIDEO_PATH_2, FOURCC, 10.0, (get_width(cap), get_height(cap)))

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray_one_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_three_channels = cv2.cvtColor(gray_one_channel, cv2.COLOR_GRAY2BGR)

        cv2.line(gray_three_channels,(0,0),(150,150),(255,255,255),15)
        cv2.rectangle(gray_three_channels,(15,25),(200,150),(0,0,255),15)

        cv2.imshow('video result', gray_three_channels)
        cv2.imshow('video original', frame)

        out.write(gray_three_channels)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_wideocam()
    draw_rectangle()