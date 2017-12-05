import numpy as np
import cv2 as cv2
from carcode import *
import glob

IMAGES2="./images2/*.jpg"
IMAGES3 = "./images3/*.jpg"

#images2 = [img_path for img_path in glob.glob(IMAGES2)]
images4 = [img_path for img_path in glob.glob(IMAGES4)]

#for img_path in images2:
#    crop(img_path)


for img_path in images3:
    colorProc(img_path)