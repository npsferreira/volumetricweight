import cv2
import imutils
import numpy as np
from imutils import contours, perspective
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

#def getTreatedBackgroundTopImage():
#    bg = cv2.imread("os.path.realpath('.') + '\\resources\\calibrationImages\\ctt1_bg.jpg")
#    bg = imutils.rotate(bg, 180)
#    
#    width = 1050
#    height = 1080
#    startX = 550
#    startY = 400

#    bg = crop(bg)
#    return bg
    
#def getTreatedBackgroundSideImage():
#    bg = cv2.imread("os.path.realpath('.') + '\\resources\\calibrationImages\\ctt1_bg.jpg")
#    bg = imutils.rotate(bg, 180)
#    
#    width = 1050
#    height = 1080
#    startX = 550
#    startY = 400
#
#    bg = crop(bg)
#    return bg

def DoGrayscaleAndBlur(image):
    # grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def displayImage(image, figSize=(8,8), title=None):
    fig = plt.figure(figsize=figSize)
    plt.axis("off")
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='Greys_r',  interpolation='bicubic')
    if title is not None:
        plt.title(title)
    plt.show()

def crop(image, width, height, startX, startY):
    return image[startY:startY+height, startX:startX+width]

def detectObject(img, bg_img, th):
    diff = cv2.absdiff(bg_img, img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #displayImage(mask,title='Computed difference in grayscale')
    
    imask =  mask>th
    canvas = np.zeros_like(img, np.uint8)
    canvas[:] = 255
    canvas[imask] = img[imask]
    #displayImage(canvas,title='Computed difference in grayscale')
    
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    canvas = DoGrayscaleAndBlur(canvas)
    edged = cv2.Canny(canvas, 20, 200)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)
    #displayImage(edged, title="Edge Detection", figSize=(20,20))
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

    index = np.where(areas == np.max(areas))[0][0]
    cnt = cnts[index]
    
    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box) 
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)

    return np.array([box])

def calculateSize(img, bg_img, ppm, th):
    boxes = detectObject(img, bg_img, th)
    
    # initialize object_sizes
    object_sizes = []
    # copy original image to draw on
    orig = img.copy()
    
    # draw boxes
    for b in boxes:
        cv2.drawContours(orig, [b.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in b:
            cv2.circle(orig, (int(x), int(y)), 10, (0, 0, 255), -1)
    
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
 
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 10, (255,0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 10, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 10, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 10, (255, 0, 0), -1)
 
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 255, 255), 4)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 255, 255), 4)
    
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
 
        # compute the size of the object
        dimA = dA / ppm
        dimB = dB / ppm

        object_sizes.append([box, dimB, dimA])

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
        
    displayImage(orig, figSize=(20,20))
    return object_sizes 

def calculatePixelDistance(img, bg_img, th, mode):
    boxes = detectObject(img, bg_img, th)
    orig = img.copy()
    
    for b in boxes:
        cv2.drawContours(orig, [b.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in b:
            cv2.circle(orig, (int(x), int(y)), 10, (0, 0, 255), -1)    
        
    #displayImage(orig)
    b = boxes[0]
    if mode == 'top': 
        x1 = img.shape[0]- np.max(b[:,1])
    else:
        x1 = np.min(b[:,1])
    return x1