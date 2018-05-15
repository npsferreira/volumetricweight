from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
from imutils import resize
import math
from utils.cttutils import displayImage
import argparse

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
    
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required = True, help = "path to the image folder")
ap.add_argument("-o", "--output", required = True, help = "path to the output folder")
ap.add_argument("-d", "--debug", required = False, default=False, help = "debug mode (True or False)")
args = vars(ap.parse_args())

folder = args['folder']
output = args['output']

print('[INFO] Opening folder ' + folder)
filelist = os.listdir(folder)

errors = []

for f in filelist:
    print('[INFO] Processing ' + f)
    im = cv2.imread(os.path.join(folder, f))
    if (args['debug']):
        displayImage(im)

    #doing initial assessment
    barcodes = np.array([])
    for i in [200, 500, 1000]:
        t = resize(im, width=i)
        barcodes= np.unique(np.concatenate((barcodes, readBarcode(t))))
        
    if len(barcodes) == 0:
        print('Doing sliding window:', f)
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
        print('Doing sliding window 2:', f)
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

    print('[INFO] Found: ', barcodes)
    if len(barcodes) == 0:
        errors.append(f)

    f = f[:-3] + 'txt'
    ofn = os.path.join(output, f)
    os.makedirs(output,exist_ok=True)
    with open(ofn, "w") as text_file:
         print(barcodes, file=text_file)

print('Writing errors: ', len(errors))
os.makedirs('./errors',exist_ok=True)
efn = os.path.join('./errors', 'errors.txt')
with open(efn, "w") as text_file:
    print(errors, file=text_file)