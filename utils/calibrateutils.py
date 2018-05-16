import cv2
import numpy as np
import imutils
import os
from utils.cttutils import displayImage
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from utils.imageutils import DoGrayscaleAndBlur
from utils.cttutils import midpoint


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
    diff = cv2.absdiff(bg_img, img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    imask =  mask>th
    canvas = np.zeros_like(img, np.uint8)
    canvas[:] = 255
    canvas[imask] = img[imask]
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    result = DoGrayscaleAndBlur(canvas)
    edged = cv2.Canny(result, 180, 200)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)
     # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts, method='left-to-right')
    
    # get areas for all contours
    areas = []
    for c in cnts:
        a = cv2.contourArea(c)
        areas.append(a)
    
    areas = np.array(areas)
    cnts = np.array(cnts)

    # remove small contours
    cnts = cnts[np.where(areas >= 600)[0]]
    
    # remove areas of removed contours
    areas = areas[np.where(areas >= 600)[0]]
    
    # loop over the contours individually and get bounding boxes
    boxes = []
    for c in cnts:
    
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) 
        box = np.array(box, dtype="int")
 
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        boxes.append(box)

    boxes = np.array(boxes)
    object_sizes = []

    pixelsPerMetric = None
    for box in boxes:
        # unpack the ordered bounding box, then compute the midpoint
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

        #object_sizes.append([box, dimB, dimA])

    return pixelsPerMetric, boxes
