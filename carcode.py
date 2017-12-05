import numpy as np
import cv2




def colorProc(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # limits of yellow mask
    lower_yellow = np.array([50, 50, 50], dtype='uint8')
    upper_yellow = np.array([110, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    neg_yellow_mask = cv2.bitwise_not(mask_yellow)
    neg_yellow = cv2.bitwise_and(gray,neg_yellow_mask)

    # limits of white mask
    lower_white = np.array([0, 0, 0], dtype='uint8')
    upper_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    y = np.zeros(img.shape)
    w = np.zeros(img.shape)

    for i in range(3):
        imslice = img[:,:,i]
        y[:,:,i] = cv2.bitwise_and(imslice,mask_yellow)
        w[:,:,i] = cv2.bitwise_and(imslice,mask_white)

    yw = y + w

    img = yw

    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    h,w,d = img.shape
    mask = np.ones([h,w,d])
    mask[0:vlim1, :,:] = np.zeros([vlim1, w,d])
    mask[vlim2:h, :,:] = np.zeros([(h - vlim2), w,d])
    mask[:, 0:hlim1,:] = np.zeros([h, hlim1,d])
    mask[:, hlim2:w,:] = np.zeros([h, w - hlim2,d])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2,:]
    mask[vlim1:vlim2, hlim1:hlim2,:] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2,:], img)
    img = mask

    return img

def lineProc(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # limits of yellow mask
    lower_white = np.array([0, 0, 0], dtype='uint8')
    upper_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # limits of yellow mask
    lower_yellow = np.array([50, 50, 50], dtype='uint8')
    upper_yellow = np.array([110, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    y = cv2.bitwise_and(gray, mask_yellow)
    w = cv2.bitwise_and(gray, mask_white)
    yw = y + w

    # reduce noise
    gauss_gray = cv2.GaussianBlur(yw, (3, 3), 0)

    # canny edge
    low_threshold = 50
    high_threshold = 160
    img = cv2.Canny(gauss_gray, low_threshold, high_threshold)

    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    h, w = img.shape
    mask = np.ones([h, w])
    mask[0:vlim1, :] = np.zeros([vlim1, w])
    mask[vlim2:h, :] = np.zeros([(h - vlim2), w])
    mask[:, 0:hlim1] = np.zeros([h, hlim1])
    mask[:, hlim2:w] = np.zeros([h, w - hlim2])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2]
    mask[vlim1:vlim2, hlim1:hlim2] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2], img)
    img = mask

    return img


def carcrop(img_path):

    img = cv2.imread(img_path)
    vlim1 = 30
    vlim2 = 105
    hlim1 = 10
    hlim2 = 150

    img = cv2.imread(img_path)
    h, w, d = img.shape
    mask = np.ones([h, w, d])
    mask[0:vlim1, :, :] = np.zeros([vlim1, w, d])
    mask[vlim2:h, :, :] = np.zeros([(h - vlim2), w, d])
    mask[:, 0:hlim1, :] = np.zeros([h, hlim1, d])
    mask[:, hlim2:w, :] = np.zeros([h, w - hlim2, d])

    mask = np.zeros(img.shape, dtype="uint8")
    img = img[vlim1:vlim2, hlim1:hlim2, :]
    mask[vlim1:vlim2, hlim1:hlim2, :] = cv2.bitwise_or(mask[vlim1:vlim2, hlim1:hlim2, :], img)
    img = mask
    return img