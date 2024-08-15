# openCV  Practice
# Motion Capture

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import datetime

def get_diff_img(frame_a, frame_b, frame_c, threshold):
    frame_a_gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    frame_b_gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    frame_c_gray = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
    diff_ab = cv2.absdiff(frame_a_gray, frame_b_gray)
    diff_bc = cv2.absdiff(frame_b_gray, frame_c_gray)
    ret, diff_ab_t = cv2.threshold(diff_ab, threshold, 255, cv2.THRESH_BINARY)
    ret, diff_bc_t = cv2.threshold(diff_bc, threshold, 255, cv2.THRESH_BINARY)
    diff = cv2.bitwise_and(diff_ab_t, diff_bc_t)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    diff = cv2.morphologyEX(diff, cv2.MORPH_OPEN, k)
    diff_cnt = cv2.CountNonZero(diff)
    return diff, diff_cnt

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,720)

font = ImageFont.truetype('fonts/SCDream6.otf') #'SCDream6.otf' is not include. you should download it.
fourcc = cv2.VideoWriter_fourcc(*'XVID')

threshold = 40
diff_max = 10

ret, frame_a = capture.read()
ret, frame_b = capture.read()

while True:
    now = datetime.datetime.now()
    nowDatetime = now.strftime("%Y.%m.%d. %H:%M:%S")
    nowDatetime_path = nowstrftime("%Y.%m.%d %H_%M_%S")
    ret, frame_c = capture.read()
    diff, diff_cnt = get_diff_img(frame_a = frame_a, frame_b = frame_b, frame_c = frame_c, threshold = threshold)
    if diff_cnt > diff_max:
        cv2.imwrite("motioncapture"+nowDatetime_path+'.png', frame)
    cv2.imshow("diff",diff)
    frame = np.array(frame_c)
    rectangle = cv2.rectangle(img = frame, pt1 = (15, 30), pt2 = (380, 30), color = (255, 255, 255), thickness = 30)
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    draw.text(xy = (15, 20), text = "motioncapture"+nowDatetime, font = font, fill = (0, 0, 0))
    frame = np.array(frame)
    frame_a = np.array(frame_b)
    frame_c = np.array(frame_c)
    cv2.imshow("motioncapturePRAC", frame)
    if cv2.waitKey(100) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
