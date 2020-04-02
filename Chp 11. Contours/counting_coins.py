import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Image", image)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count", len(cnts), "coins in this image")

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)

coins = image.copy()
cv2.drawContours(coins, cnts, 0, (255, 0, 0), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)

coins = image.copy()
cv2.drawContours(coins, cnts, 1, (0, 255, 0), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)

coins = image.copy()
cv2.drawContours(coins, cnts, 2, (0, 0, 255), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)

for (i, c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)

    print("Coin #{}".format(i+1))
    coin = image[y:y+h, x:x+w]
    cv2.imshow("Coin", coin)
    cv2.waitKey(0)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)

    mask = mask[y:y+h, x:x+w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask=mask))
    cv2.waitKey(0)
