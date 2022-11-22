#!/usr/bin/python3
from datetime import datetime
import cv2
import os
import numpy as np
import pytesseract
import string
from tensorflow.lite.python.interpreter import Interpreter
import math

class PlateRecognition():
    def __init__(self) -> None:
        self.char_list =  string.digits + string.ascii_uppercase
        self.digit_w = 30 # Kich thuoc ki tu
        self.digit_h = 60 # Kich thuoc ki tu
        self.model_svm = cv2.ml.SVM_load(os.path.join(os.getcwd(), 'models', 'svm.xml'))
        with open(os.path.join(os.getcwd(), 'models', 'plate_recognition_labelmap.txt'), 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.num_threads = 4

    def recognize_ssd(self, image, min_conf=0.5) -> str:
        interpreter = Interpreter(model_path=os.path.join(os.getcwd(), 'models', 'mobilenetv2_model_plate_recognition_quant.tflite'), num_threads=self.num_threads)
        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        float_input = (input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5


        outname = output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else: # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        num_list = []
        for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))
              
            #   cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
              # Draw label
              object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            #   cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            #   cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
              length = ymax - ymin
              num_list.append([xmin, ymin, object_name, length])

        return self.get_license_plate(num_list)

    def recognize_svm(self, img) -> str:
         # Chuyen anh bien so ve gray
        img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(img_gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]

        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""
        for c in reversed(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w 
            if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                if h/img.shape[0]>=0.2: # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(self.digit_w, self.digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num,dtype=np.float32)
                    curr_num = curr_num.reshape(-1, self.digit_w * self.digit_h)

                    # Dua vao model SVM
                    result = self.model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result<=9: # Neu la so thi hien thi luon
                        result = str(result)
                    else: #Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info +=result
        # print(f'Dectected: {plate_info}')
        return plate_info

    # Ham sap xep contour tu trai sang phai
    def sort_contours(self, cnts):
        reverse = True
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

    # Ham fine tune bien so, loai bo cac ki tu khong hop ly
    def fine_tune(self, lp) -> str:
        newString = ""
        for i in range(len(lp)):
            if lp[i] in self.char_list:
                newString += lp[i]
        return newString

    def get_license_plate(self, num_list):
        def min_Y(array_2d):
            res = float('inf')
            for a in array_2d:
                if a[1] < res:
                    res = a[1]
            return res
        def avg_length(array_2d):
            s = 0
            length_array_2d = len(array_2d)
            for a in array_2d:
                s += a[3]
            return s / length_array_2d if length_array_2d > 0 else 0
        min_y = min_Y(num_list)    
        line1 = []
        line2 = []
        THRESH_HOLD = 0.6 * avg_length(num_list)
        for a in num_list:
            if abs(a[1] - min_y) < THRESH_HOLD:
                line1.append(a)
            else:
                line2.append(a)

        line1 = sorted(line1, key=lambda e: e[0])
        line2 = sorted(line2, key=lambda e: e[0])

        if len(line2) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[2]) for ele in line1])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[2]) for ele in line1])  + "".join([str(ele[2]) for ele in line2])
        return license_plate