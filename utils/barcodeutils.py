from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
from imutils import resize
import math
from utils.imageutils import displayImage


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    return decodedObjects

def readBarcode(image):
    decodedObjects = decode(image)
    barcodes =[]
    for obj in decodedObjects:
        barcodes.append(obj.data.decode('ascii'))
    return barcodes

def findBarCode(file):
    im = cv2.imread(file)
    
    #doing initial assessment
    barcodes = np.array([])
    for i in [200, 500, 1000]:
        t = resize(im, width=i)
        barcodes= np.unique(np.concatenate((barcodes, readBarcode(t))))
        
    if len(barcodes) == 0:
        print('Doing sliding window:', file)
        im2 = resize(im, 2000)
        slidingWindowSize = 600
        imageWidth = im2.shape[1]
        imageHeight = im2.shape[0]
        steps = math.floor(imageHeight / slidingWindowSize)
        step = 200
        currentY = 0
        currentX = 0

        while (currentY + slidingWindowSize <= imageHeight):
            while(currentX + slidingWindowSize <= imageWidth):
                t = im2[currentY:slidingWindowSize + currentY,currentX:slidingWindowSize + currentX]
                currentX = currentX + step
                barcodes= np.unique(np.concatenate((barcodes, readBarcode(t))))  
            currentY = currentY + step
            currentX = 0
     
    if len(barcodes) == 0:
        im2 = im
        slidingWindowSize = 800
        imageWidth = im2.shape[1]
        imageHeight = im2.shape[0]
        steps = math.floor(imageHeight / slidingWindowSize)
        step = 200
        currentY = 0
        currentX = 0

        while (currentY + slidingWindowSize <= imageHeight):
            while(currentX + slidingWindowSize <= imageWidth):
                t = im2[currentY:slidingWindowSize + currentY,currentX:slidingWindowSize + currentX]
                currentX = currentX + step
                barcodes= np.unique(np.concatenate((barcodes, readBarcode(t))))
            currentY = currentY + step
            currentX = 0

    return barcodes