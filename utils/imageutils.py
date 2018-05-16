import cv2
import imutils

 

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

def crop(image, width, height, startX, startY):
    return image[startY:startY+height, startX:startX+width]