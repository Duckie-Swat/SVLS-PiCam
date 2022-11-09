import os
from datetime import datetime
import cv2
import geocoder

def save(img, currentDir=os.getcwd(),fileName=f'{datetime.timestamp(datetime.now())}.jpg') -> None:
    savedDir = os.path.join(currentDir, 'saved')
    os.makedirs(savedDir, exist_ok=True)
    cv2.imwrite(os.path.join(savedDir, fileName), img)
    print(f'saved {fileName} to {savedDir}')

def get_current_gps():
    return geocoder.ip('me')