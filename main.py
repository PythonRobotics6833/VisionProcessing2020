import cv2
import numpy as np
import sys
import logging
import time
from networktables import NetworkTables


vid = cv2.VideoCapture(1)


logging.basicConfig(level=logging.DEBUG)
minpixels = 200
minpixels_cube = 800

# while not NetworkTables.isConnected():
#print (NetworkTables.initialize(server='192.168.0.110'))
#time.sleep(1)
#statustable = NetworkTables.getTable("status")
#statustable.putBoolean('booted', True)
#vtargetobj = NetworkTables.getTable("vtargetobj")
#cubetarget = NetworkTables.getTable("cubetarget")
counter=0
while (True):

    #if statustable.getEntry('powerstatus').getBoolean(True) == False:
    #    break
    ret, frame = vid.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # covert to hsv space
    gBlurImg = cv2.GaussianBlur(hsv, (9, 9), 1.7)  # gaussian blur for noise reduction

    # cv2.imshow('orig',hsv)
    cv2.imshow("Frame", frame)
    #print(gBlurImg[0,0])
    cv2.waitKey(1)
    print("hello")
    lower_green = np.array([30, 100, 100])
    upper_green =   np.array([65, 230, 200])

    mask = cv2.inRange(gBlurImg, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    targetlistx = []
    targetlisty = []
    targetlistw = []
    targetlisth = []
    if len(contours) != 0:
        i = 0
        while (i < len(contours)):
            if (cv2.contourArea(contours[i]) > minpixels):
                comp = hierarchy[0, i, 3]
                if (comp == -1):
                    x, y, w, h = cv2.boundingRect(contours[i])
                    targetlistx.append(x)
                    targetlisty.append(y)
                    targetlistw.append(w)
                    targetlisth.append(h)
                    cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)
            i = i + 1

    #cv2.imshow('fff', res)
    #vtargetobj.putNumber('objcount', len(targetlistx))
    #vtargetobj.putNumberArray('vtargetobjx', targetlistx)
    #vtargetobj.putNumberArray('vtargetobjy', targetlisty)
    #vtargetobj.putNumberArray('vtargetobjw', targetlistw)
    #vtargetobj.putNumberArray('vtargetobjh', targetlisth)

    # cube detection

    lower_yellow = np.array([18, 150, 50])
    upper_yellow = np.array([26, 255, 255])


    mask_yellow = cv2.inRange(gBlurImg, lower_yellow, upper_yellow)
    #cv2.imshow("mask",mask_yellow)
    res_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)
    #res_yellow = cv2.dilate(res_yellow, np.ones((9, 9), np.uint8), iterations=1)
    #res_yellow = cv2.erode(res_yellow, np.ones((9, 9), np.uint8), iterations=1)
    res_yellow = cv2.cvtColor(res_yellow, cv2.COLOR_HSV2BGR)
    imgray_yellow= cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY)

    #ret_yellow, thresh_yellow = cv2.threshold(imgray_yellow, 127, 255, 0)

    
    contours_yellow, hierarchy_yellow = cv2.findContours(imgray_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #power port data array
    #It is a 2 dementional array with the first being the object number
    #The second is the data of the rectangle with the values being [x,y,w,h]
    powerCells = np.array([])
    cubelistpixels = []
    if len(contours_yellow) != 0:
        for i in range(0,len(contours_yellow)):
            if (cv2.contourArea(contours_yellow[i]) > minpixels_cube):
                comp = hierarchy_yellow[0, i, 3]
                if (comp == -1):
                    x, y, w, h = cv2.boundingRect(contours_yellow[i])
                    roi = imgray_yellow[x:x + w, y:y + h]
                    pixels = w * h - cv2.countNonZero(roi)
                    cubelistpixels.append(pixels)
                    powerCells= np.vstack([x,y,w,h])
                    cv2.rectangle(res_yellow, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('cubes', res_yellow)
    print(powerCells[:][:])
    #cubetarget.putNumber('objcount', len(cubelistx))
    #cubetarget.putNumberArray('cubelistpixels', cubelistpixels)
    #cubetarget.putNumberArray('cubetargetx', cubelistx)
    #cubetarget.putNumberArray('cubetargety', cubelisty)
    #cubetarget.putNumberArray('cubetargetw', cubelistw)
    #cubetarget.putNumberArray('cubetargeth', cubelisth)

    counter+=1
    if counter >5:
        break


#statustable2 = NetworkTables.getTable("status")
#statustable2.putBoolean('booted', False)

print("break")
vid.release()
cv2.destroyAllWindows()
