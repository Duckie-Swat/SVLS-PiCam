
import requests
import os
import json

WORK_DIR = os.getcwd()
BASE_URL = 'https://svls-api.duckieswat.com/api/v1'

'''
    each 1p send 1 list
'''
def send_detected_results(url, payload):
    requests.post(url=url,  headers={
        "Content-Type":"application/json"
    }, body=payload)
    

send_data = os.path.join(WORK_DIR, 'saved', 'send_data.json')
SAVED_DIR = os.path.join(WORK_DIR, 'saved')

for file in SAVED_DIR:
    pass

# with open(send_data, 'r') as file:
#     send_detected_results(f'{BASE_URL}/camera-detected-result', body=json.load(file.read()))

