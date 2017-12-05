import cv2
from carcode import *

img_path = 'pic.jpg'
img = colorProc(img_path)

cv2.imshow('adf',img)
cv2.waitKey(0)
