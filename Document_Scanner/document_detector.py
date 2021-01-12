# import the necessary packages
from imagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image,height=500)

# convert image to grayscale , blur, and find edges
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray,75,200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image",image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# finding contours in the image , keeping the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_Ccontours(cnts)
cnts  =sorted(cnts,key=cv2.contourAre,reverse=True)[:5]

# loop over the contours
for c in cnts:
	# approximate contour
	peri = cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,0.02*peri,True)

	# if out contour has four points then we can assume it to be paper
	if len(approx)==4:
		screenCnt = approx
		break

# show contour (outline) of the paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply four point transform to obtain top-down view of image
wraped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

# convert wrapped image to grayscale, threshold it and give paper effect
wraped = cv2.cvtColor(wraped,cv2.COLOR_BGR2GRAY)
T = threshold_local(wraped,11,offset = 10,method="gaussian")
wraped = (wraped>T).astype("Unit8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")

cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)







