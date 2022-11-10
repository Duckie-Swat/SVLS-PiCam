import os
from datetime import datetime
import cv2
import geocoder
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

def save(img, currentDir=os.getcwd(),fileName=f'{datetime.timestamp(datetime.now())}.jpg') -> None:
    savedDir = os.path.join(currentDir, 'saved')
    os.makedirs(savedDir, exist_ok=True)
    cv2.imwrite(os.path.join(savedDir, fileName), img)
    print(f'saved {fileName} to {savedDir}')

def get_current_gps():
    return geocoder.ip('me').latlng

def convert_gps_to_address(gps):
    gps_str = ", ".join(map(lambda p: str(p), gps))
    address = geolocator.geocode(geolocator.reverse(gps_str), addressdetails=True)
    return address.raw['address']['city_district']
