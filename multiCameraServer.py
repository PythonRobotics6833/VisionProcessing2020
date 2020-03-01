#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.
#----------------------------------------------------------------------------

import json
import time
import sys
import cv2
import numpy as np
import logging
import cvsink 

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from networktables import NetworkTablesInstance
import ntcore

#   JSON format:
#   {
#       "team": <team number>,
#       "ntmode": <"client" or "server", "client" if unspecified>
#       "cameras": [
#           {
#               "name": <camera name>
#               "path": <path, e.g. "/dev/video0">
#               "pixel format": <"MJPEG", "YUYV", etc>   // optional
#               "width": <video mode width>              // optional
#               "height": <video mode height>            // optional
#               "fps": <video mode fps>                  // optional
#               "brightness": <percentage brightness>    // optional
#               "white balance": <"auto", "hold", value> // optional
#               "exposure": <"auto", "hold", value>      // optional
#               "properties": [                          // optional
#                   {
#                       "name": <property name>
#                       "value": <property value>
#                   }
#               ],
#               "stream": {                              // optional
#                   "properties": [
#                       {
#                           "name": <stream property name>
#                           "value": <stream property value>
#                       }
#                   ]
#               }
#           }
#       ]
#       "switched cameras": [
#           {
#               "name": <virtual camera name>
#               "key": <network table key used for selection>
#               // if NT value is a string, it's treated as a name
#               // if NT value is a double, it's treated as an integer index
#           }
#       ]
#   }

configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []
switchedCameraConfigs = []
cameras = []

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readCameraConfig(config):
    """Read single camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get("stream")

    cam.config = config

    cameraConfigs.append(cam)
    return True

def readSwitchedCameraConfig(config):
    """Read single switched camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read switched camera name")
        return False

    # path
    try:
        cam.key = config["key"]
    except KeyError:
        parseError("switched camera '{}': could not read key".format(cam.name))
        return False

    switchedCameraConfigs.append(cam)
    return True

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    # switched cameras
    if "switched cameras" in j:
        for camera in j["switched cameras"]:
            if not readSwitchedCameraConfig(camera):
                return False

    return True

def startCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    inst = CameraServer.getInstance()
    camera = UsbCamera(config.name, config.path)
    
    server = inst.startAutomaticCapture(camera=camera, return_server=True)

    camera.setConfigJson(json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen)

    if config.streamConfig is not None:
        server.setConfigJson(json.dumps(config.streamConfig))

        
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
            #ret, frame = vid.read()

            ret, frame = camera.grabFrame()

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

            #print("break")
            #vid.release()
            #cv2.destroyAllWindows()



    return camera

def startSwitchedCamera(config):
    """Start running the switched camera."""
    print("Starting switched camera '{}' on {}".format(config.name, config.key))
    server = CameraServer.getInstance().addSwitchedCamera(config.name)

    def listener(fromobj, key, value, isNew):
        if isinstance(value, float):
            i = int(value)
            if i >= 0 and i < len(cameras):
              server.setSource(cameras[i])
        elif isinstance(value, str):
            for i in range(len(cameraConfigs)):
                if value == cameraConfigs[i].name:
                    server.setSource(cameras[i])
                    break

    NetworkTablesInstance.getDefault().getEntry(config.key).addListener(
        listener,
        ntcore.constants.NT_NOTIFY_IMMEDIATE |
        ntcore.constants.NT_NOTIFY_NEW |
        ntcore.constants.NT_NOTIFY_UPDATE)

    return server

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read uration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)

    # start cameras
    for config in cameraConfigs:
        cameras.append(startCamera(config))

    # start switched cameras
    for config in switchedCameraConfigs:
        startSwitchedCamera(config)

    # loop forever
    while True:
        time.sleep(10)
