#!/usr/bin/python3
from datetime import datetime
import cv2
import os
import numpy as np
import pytesseract
import string

class PlateRecognition():
    def __init__(self) -> None:
        self.char_list =  string.digits + string.ascii_uppercase
        self.digit_w = 30 # Kich thuoc ki tu
        self.digit_h = 60 # Kich thuoc ki tu
        self.model_svm = cv2.ml.SVM_load(os.path.join(os.getcwd(), 'models', 'svm.xml'))

    
    def recognize(self, img) -> str:
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