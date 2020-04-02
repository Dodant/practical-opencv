# import the necessary packages
import numpy as np
import cv2

class CoverDescriptor:
	def __init__(self, useSIFT=False):
		# store the keypoint detection method and descriptor method
		self.useSIFT = useSIFT

	def describe(self, image):
		# detect keypoints in the image
		descriptor = cv2.BRISK_create()

		if self.useSIFT:
			descriptor = cv2.xfeatures2d.SIFT_create()

		(kps, descs) = descriptor.detectAndCompute(image, None)
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and descriptors
		return (kps, descs)
