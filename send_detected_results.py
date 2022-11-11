
import requests
import os
import json
import base64

WORK_DIR = os.getcwd()
BASE_URL = 'https://svls-api.duckieswat.com/api/v1'
IMG_EXTENSION = '.jpg'

'''
    each 1p send 1 list
'''
def send_detected_results(url, payload):
    print(type(payload))
    response = requests.post(url=url, json=payload)
    print(response.content)

send_data = os.path.join(WORK_DIR, 'saved', 'send_data.json')
SAVED_DIR = os.path.join(WORK_DIR, 'saved')

request_payload = []
list_image_remove = []

def obj_dict(obj):
    return obj.__dict__

# Handle send all detected image
for file in os.listdir(SAVED_DIR):
    if file.endswith(IMG_EXTENSION):
        plate_num, request_id = file.split("_")
        request_id = request_id.split('.jpg')[0]
        file_name = plate_num
        file_url = os.path.join(SAVED_DIR, file)
        text_file_url = os.path.join(SAVED_DIR, f'{file_name}.txt')
        
        # Get GPS and address from txt file
        with open(text_file_url, 'r') as txt_file:
            json_content = json.loads(txt_file.read())
            current_gps, current_address = json_content["current_gps"], json_content["current_address"]
        
        # Encode image
        with open(file_url, "rb") as image_file:
            photo = base64.b64encode(image_file.read()).decode('utf-8')
            

        request_payload.append(
            {
                    "cameraId": "f5a5d695-eb56-4070-b262-76066b63bc46",
                    "lostVehicleRequestId": request_id,
                    "latitude": current_gps[0],
                    "longitude": current_gps[1],
                    "location": current_address,
                    "photo": photo,
                    "plateNumber": plate_num
            }
        )
        list_image_remove.append(file_url)
        list_image_remove.append(text_file_url)

if len(request_payload) > 0:
    print(type(request_payload))

    send_detected_results(f'{BASE_URL}/camera-detected-result/List', payload=request_payload)
    # Handle remove all image
    for file in list_image_remove:
        os.remove(file)
    print(f"Send a POST request to server with payload {request_payload}")
