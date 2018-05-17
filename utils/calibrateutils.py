import cv2
import numpy as np
import imutils
import os
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from utils.imageutils import DoGrayscaleAndBlur, midpoint, displayImage, crop, detectObject


#def getPreviousCalibratedValues():
#    calibratedValues = {}
#    with open(os.path.realpath('.') + '\\config\\calibrationvalues.txt') as myfile:
#        for line in myfile:
#            name, var = line.partition("=")[::2]
#            calibratedValues[name.strip()] = float(var)
#    return calibratedValues

#def setCalibratedValuesInFile(side_b, side_m, top_b, top_m):
#    calibratedValues = {}
#    calibratedValues.append({'side_b=', side_b}, {'side_m', side_m}, {'top_b', top_b}, {'top_m', top_m})
#    os.makedirs('./config', exist_ok=True)
#    efn = os.path.join('./config', 'calibrationvalues2.txt')
#    with open(efn, "w") as text_file:
#        print(calibratedValues, file=text_file)

def calculateMetric(knownValue, img, bg_img, metric, th):
     
    boxes = detectObject(img, bg_img, th)

    pixelsPerMetric = None
    box = boxes[0]
    #unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        if metric == 'height':
            pixelsPerMetric = dA / knownValue
        else:
            pixelsPerMetric = dB / knownValue

    return pixelsPerMetric, boxes

def calculatePpm(x, m, b):
    ppm = x*m + b
    return ppm

def getCameraParameters():
    side_params = {
                'width': 1500,
                'height': 1000,
                'startX': 500,
                'startY': 400,
                'th': 50
              }

    top_params = {      
                'width': 1050,
                'height': 1300,
                'startX': 500,
                'startY': 400,
                'th': 35
              }
    
    return side_params, top_params

def calibrate(bg_top, top1, top2, top_params, bg_side, side1, side2, side_params, knownHeight, knownWidth):
    
    top1 = imutils.rotate(top1,180)
    top2 = imutils.rotate(top2,180)
    bg_top = imutils.rotate(bg_top, 180)

    side1 = crop(side1, side_params['width'], side_params['height'], side_params['startX'], side_params['startY'])
    side2 = crop(side2, side_params['width'], side_params['height'], side_params['startX'], side_params['startY'])
    bg_side = crop(bg_side, side_params['width'], side_params['height'], side_params['startX'], side_params['startY'])

    top1 = crop(top1, top_params['width'], top_params['height'], top_params['startX'], top_params['startY'])
    top2 = crop(top2, top_params['width'], top_params['height'], top_params['startX'], top_params['startY'])
    bg_top = crop(bg_top, top_params['width'], top_params['height'], top_params['startX'], top_params['startY'])

    hppm1, boxes_s1 = calculateMetric(knownHeight, side1, bg_side, 'height', side_params['th'])
    hppm2, boxes_s2 = calculateMetric(knownHeight, side2, bg_side, 'height', side_params['th'])
    wppm1, boxes_t1 = calculateMetric(knownWidth, top1, bg_top, 'width', top_params['th'])
    wppm2, boxes_t2 = calculateMetric(knownWidth, top2, bg_top, 'width', top_params['th'])

    # side
    b = boxes_t1[0]
    x1 = top1.shape[0]- np.max(b[:,1])
    b = boxes_t2[0]
    x2 = top2.shape[0]- np.max(b[:,1])
    side_b = (hppm2*x1 - hppm1*x2) / (-x2 + x1)
    side_m = (hppm1 - side_b) / x1

    # top
    b = boxes_s1[0]
    x1 = np.min(b[:,1])
    b = boxes_s2[0]
    x2 = np.min(b[:,1])
    top_b = (wppm2*x1 - wppm1*x2) / (-x2 + x1)
    top_m = (wppm1 - top_b) / x1

    return side_b, side_m, top_b, top_m