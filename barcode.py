from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
from imutils import resize
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

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required = True, help = "path to the image folder")
args = vars(ap.parse_args())

folder = args['folder']
print('[INFO] Opening folder ' + folder)
filelist = os.listdir(folder)

for f in filelist:
    print('[INFO] Processing ' + f)
    im = cv2.imread('barcodes/' + f)
    displayImage(im)

    barcodes = np.array([])
    for i in [200, 500, 1000]:
        t = resize(im, width=i)
        barcodes= np.unique(np.concatenate((barcodes, readBarcode(t))))

    print('[INFO] Found: ', barcodes)