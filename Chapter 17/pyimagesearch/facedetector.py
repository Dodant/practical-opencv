import cv2

class FaceDetector:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontface.xml')

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
        rects = self.faceCascade.detectMultiScale(image,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=cv2.CASCADE_SCALE_IMAGE)
        return rects
