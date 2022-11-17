#!/usr/bin/python3
from threading import Thread
import cv2

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30, source=0):
        # Initialize the PiCamera and the camera image stream
        self.source = source
        self.stream = cv2.VideoCapture(source)
        ret = self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        ret = self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,resolution[0])
        ret = self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,resolution[1])
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'JPEG'))
        ret = self.stream.set(cv2.CAP_PROP_FPS, framerate)
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    # def start(self):
	# # Start the thread that reads frames from the video stream
    #     Thread(target=self.update,args=()).start()
    #     return self

    # def replay(self):
    #     self.stream = cv2.VideoCapture(self.source)
    #     (self.grabbed, self.frame) = self.stream.read()

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while not self.stopped:
            # If the camera is stopped, stop the thread
            if not self.grabbed:
                # Close camera resources
                self.stop()
                self.stream.release()
            else:
                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True