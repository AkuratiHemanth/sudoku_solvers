# Imports and libraries
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform



def colr_to_grey(file_bytes):
  # Read the input sudoku image
  img =  cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  # Preprocess the input image and apply adaptive thresholding
  img = cv2.resize(img,(1026,1026))
  imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  imgray =  cv2.GaussianBlur(imgray,(11,11),0)
  thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)

  # Connect the broken grid lines on the threshold image using dilation
  kernel = np.array([0,1,0,1,1,1,0,1,0],dtype=np.uint8).reshape(3,3)
  thresh = cv2.dilate(thresh, kernel,iterations = 2)
  #cv2.imwrite('thresh.jpg', thresh)

  # Find contours in the thresholded image and sort them by size in descending order
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  # Find the largest quadrilateral contour with the help of polygonal approximation
  puzz_cntr = None
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)

    if len(approx) == 4:
      puzz_cntr = approx
      break

  # Apply four point perspective transform and get the warped images
  out_gry=four_point_transform(img, puzz_cntr.reshape(4, 2))
  out_bin=four_point_transform(thresh, puzz_cntr.reshape(4, 2))

  # Dilate the image to fill the cracks
  kernel = np.array([0,1,0,1,1,1,0,1,0],dtype=np.uint8).reshape(3,3)
  out_bin = cv2.dilate(out_bin, kernel,iterations = 1)

  # Resize the binary and grayscale images
  out_bin=cv2.resize(out_bin,(1026,1026))
  out_gry=cv2.resize(out_gry,(1026,1026))

  return out_gry,out_bin