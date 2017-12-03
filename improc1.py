import numpy as np
import cv2 as cv2
from lane_detection import *
import glob

IMAGES2="./images2/*.jpg"
IMAGES3 = "./images3/*.jpg"

images2 = [img_path for img_path in glob.glob(IMAGES2)]
images3 = [img_path for img_path in glob.glob(IMAGES3)]

#for img_path in images2:
#    crop(img_path)


for img_path in images3:
    process(img_path)
    crop(img_path)