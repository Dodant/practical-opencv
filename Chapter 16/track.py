import numpy as np
import argparse
import time
import cv2

ap = argparse.ArgumentParser()
 # ap.add_argument("-v", "--video", required=True, help="Path to the (optional) video file")
args = vars(ap.parse_args())

blueLower = np.array([100,50,0], dtype="uint8")
blueUpper = np.array([255,200,150], dtype="uint8")

cam = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    blue = cv2.inRange(frame, blueLower, blueUpper)
    blue = cv2.GaussianBlur(blue, (3,3), 0)

    (cnts, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame, [rect], -1, (0,255,0), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Binary", blue)

    time.sleep(0.025)

    if cv2.waitKey(0) & 0xff == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
