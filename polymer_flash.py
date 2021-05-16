import cv2
import numpy as np
img = cv2.imread('/home/rohit/Pictures/polymer_flash/170394.png')

img=cv2.resize(img,(600,600))
cv2.imshow('imlg',img)
lower_white = np.array([0,0,0])
higher_white = np.array([70,70,70])
# getting the range of blue color in frame
white_range = cv2.inRange(img, lower_white, higher_white)


contours, hierarchy = cv2.findContours(white_range,
  cv2.RETR_TREE,
  cv2.CHAIN_APPROX_SIMPLE)
     
# Draw the contours (in red) on the original image and display the result
# Input color code is in BGR (blue, green, red) format
# -1 means to draw all contours
with_contours = cv2.drawContours(img, contours, -1, color=(255, 0, 255), thickness=cv2.FILLED)
cv2.imshow('Detected contours', with_contours)

cv2.imshow("White", white_range)
cv2.waitKey(0)



# hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

# # Define lower and uppper limits of what we call "brown"
# brown_lo=np.array([0, 0, 0])
# brown_hi=np.array([10,10, 10])

# # Mask image to only select browns
# mask=cv2.inRange(hsv,brown_lo,brown_hi)

# # Change image to red where we found brown
# image[mask>0]=(0,0,255)

# cv2.imshow('img',image)
# cv2.waitKey(0)



# Color threshold

# original = image.copy()
# blank = np.zeros(image.shape, dtype=np.uint8)
# blur = cv2.GaussianBlur(image, (7,7), 0)
# hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
# lower = np.array([0, 0, 0])
# upper = np.array([179, 93, 97])
# mask = cv2.inRange(hsv, lower, upper)

# # Morph operations
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
# close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# # Find contours and filter using contour approximation + contour area
# cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#     area = cv2.contourArea(c)
#     if len(approx) > 3 and area > 1000:
#         cv2.drawContours(image, [c], -1, (36,255,12), -1)
#         cv2.drawContours(blank, [c], -1, (255,255,255), -1)

# # Bitwise-and for result
# blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
# result = cv2.bitwise_and(original,original,mask=blank)
# result[blank==0] = (255,255,255)

# cv2.imshow('mask', mask)
# cv2.imshow('opening', opening)
# cv2.imshow('close', close)
# cv2.imshow('result', result)
# cv2.imshow('image', image)
# cv2.waitKey()

