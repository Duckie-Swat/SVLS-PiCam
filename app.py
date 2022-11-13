#!/usr/bin/python3
from flask import Flask, render_template, Response
import os
import cv2
import numpy as np
import time
import importlib.util
from my_videostream import VideoStream
from my_plate_recognition import PlateRecognition
from my_util import save, get_current_gps, convert_gps_to_address
from datetime import datetime
import json
import pytesseract
from threading import Thread

app=Flask(__name__)
plateRecognition = PlateRecognition()

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter    

CWD_PATH = os.getcwd()
min_conf_threshold = 0.5
imW, imH = 1280, 720
MODEL_NAME = 'models'
GRAPH_NAME = 'detect_quant_v2.tflite'
LABELMAP_NAME = 'labelmap.txt'
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to .tflite file, which contains the model that is used for classification
PATH_TO_CKPT_CLS = os.path.join(CWD_PATH,MODEL_NAME,'mobilenetv2_model_classification.tflite')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

LOST_VEHICLE_LIST = os.path.join(CWD_PATH, 'saved', 'data.json')


# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
print("Starting video capturing .....")
time.sleep(1)

def check_plate_number_belong_lost_vehicle(plate_number, frame):
    with open(LOST_VEHICLE_LIST, 'r') as f:
        try:
            lost_vehicle_list = json.loads(f.read())["items"]
            for lv in lost_vehicle_list:
                lv_plate_number = lv["plateNumber"]
                lv_request = lv["id"]

                lv_plate_number = lv_plate_number.replace("-", "")
                lv_plate_number = lv_plate_number.replace(".", "")
                            
                if lv_plate_number == plate_number:
                    print("detected")
                    current_gps = get_current_gps()
                    current_address = convert_gps_to_address(current_gps)
                    with open(os.path.join(CWD_PATH, 'saved', f'{lv_plate_number}.txt'), 'w') as file:
                        file.write(json.dumps({
                            "current_gps": current_gps,
                            "current_address": current_address
                        }))
                        save(frame, fileName=f'{plate_number}_{lv_request}.jpg')
        except Exception as e:
            print(f'Exception {e}')


def gen_frames():
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    
    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2
    
    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        
         # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                # label = '%s: %d%% ' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                label = '{{plate_number}}: {{plate_number_percentagle}}% '

                # label = label.replace('{{plate_number}}', object_name)
                label = label.replace('{{plate_number_percentagle}}', f'%.2f' % (scores[i]*100) )

                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

                # Plate recognition

                cropped_image = frame[ymin:ymax, xmin:xmax]
                # plate_number = plateRecognition.recognize_svm(cropped_image)
                plate_number = plateRecognition.recognize_ssd(cropped_image)
                label = label.replace('{{plate_number}}', plate_number)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                Thread(target=check_plate_number_belong_lost_vehicle, args=(plate_number, frame, )).start()
                
                    
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
         # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=False)