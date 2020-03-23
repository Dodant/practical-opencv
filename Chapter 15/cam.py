from pyimagesearch.facedetector import FaceDetector
import imutils
import argparse
import cv2

ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade resides")
# ap.add_argument("-v", "--video", required=True, help="Path to the (optional) video file")
args = vars(ap.parse_args())

fd = FaceDetector()

if not args.get("video", False):
    cam = cv2.VideoCapture(0)
else:
    cam = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = cam.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    frameClone = frame.copy()

    for (x,y,w,h) in faceRects:
        cv2.rectangle(frameClone, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Faces", frameClone)

    if cv2.waitKey(0) & 0xff == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
