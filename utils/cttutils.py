# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csv
import random
from urllib.request import urlopen
import codecs
import os
from utils.calibrateutils import calibrate

def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted

def rotate(image, angle, center=None, scale=1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w // 2, h // 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def plot_histogram(image, title, mask=None):
    # Grab the image channels, initialize the tuple of colors
	# and the figure

	sns.set()

	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")

	# Loop over the image channels
	for (chan, color) in zip(chans, colors):
		# Create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color = color)
		plt.xlim([0, 256])

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image

def calculateVolumeWeight(height, width, length, destination):
    # apply the volume weight formula based on the measures passed
    return round((height * width * length) / getDestinationConversionFactor(destination), 1)

def getDestinationConversionFactor(destination):
    # CTT conversion factor for the destination
    switcher = {
        "Portugal": 6000,
        "Spain": 4000,
        "RestOfTheWorld": 5000,
    }
    conversionFactor = switcher.get(destination, 6000)
    return conversionFactor
    
def findPackageInformation(barcode):
    with open(os.path.realpath('.') + '\\resources\\addresses\\Moradas.csv', 'r', encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        randomNum = random.randint(1,3000)
    
        idx = 0
        for row in spamreader:
            idx += 1
            if(idx == randomNum):
                return row
            

def calibrateCamerasForVolumeWeight():
	
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
                'th': 30
              }
	
	bg_side = cv2.imread("resources/calibrationImages/bg_side.jpg")
	side1 = cv2.imread("resources/calibrationImages/side1.jpg")
	side2 = cv2.imread("resources/calibrationImages/side2.jpg")

	bg_top = cv2.imread("resources/calibrationImages/bg_top.jpg")
	top1 = cv2.imread("resources/calibrationImages/top1.jpg")
	top2 = cv2.imread("resources/calibrationImages/top2.jpg")

	knownHeight = 19.5
	knownWidth = 20

	vals = calibrate(bg_top, top1, top2, top_params, bg_side, side1, side2, side_params, knownHeight, knownWidth)
	#save values in file
	return vals