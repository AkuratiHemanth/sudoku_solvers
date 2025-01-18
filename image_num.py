# Imports and libraries
import cv2
import numpy as np





# Crop square ROI from the center of original grayscale image
def crop_center(bin, gry,cropx,cropy):

   y,x = bin.shape[0],bin.shape[1]
   startx = x//2-(cropx//2)
   starty = y//2-(cropy//2)

   crop_bin = bin[starty:starty+cropy,startx:startx+cropx]
   crop_gry = gry[starty:starty+cropy,startx:startx+cropx]

   # Use binary image to find the largest contour
   contours, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   zero=cv2.imread('zero_template.jpg', cv2.IMREAD_GRAYSCALE)

   # Check for blanks in the image
   if len(contours) == 0 or cv2.contourArea( max(contours, key = cv2.contourArea))<250:
     return zero

   # Get contour with maximum area
   cnt = max(contours, key = cv2.contourArea)
   x,y,w,h = cv2.boundingRect(cnt)
   d=(h-w)//2
   c=crop_gry.shape[0]
   ROI = crop_gry[y:y+h, max(0,x-d):min(c,x+w+d)] # Save grayscale image crops

   return ROI



def roi_to_num(out_gry,out_bin):
  # Get crop size for square blocks
  imgheight=out_bin.shape[0]
  imgwidth=out_bin.shape[1]
  H,W = imgheight//9, imgwidth//9

  # For each block crop roi and add them to list
  sudokus=[]
  gry=[]
  for y in range(0,imgheight,H):
      for x in range(0, imgwidth, W):
          y1 = y + H
          x1 = x + W
          tiles_bin = out_bin[y:y+H,x:x+W]
          tiles_gry = cv2.cvtColor(out_gry[y:y+H,x:x+W], cv2.COLOR_BGR2GRAY)

          digits = crop_center(tiles_bin,tiles_gry,81,81)
          digits=cv2.resize(digits,(32,32),cv2.INTER_AREA)

          sudokus.append(digits/255.0)

  # Create a float numpy array with 81 images from list
  sudoku_numbers=np.float32(sudokus).reshape(81,32,32,1)
  #print(sudoku_numbers)
  return sudoku_numbers