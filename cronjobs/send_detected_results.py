
import requests
import os


WORK_DIR = os.getcwd()
BASE_URL = 'https://svls-api.duckieswat.com/api/v1'


def send_detected_results(url, payload):
    requests.post(url=url,  headers={
        "Content-Type":"application/json"
    }, body=payload)
    




send_detected_results(f'{BASE_URL}/camera-detected-result', body=[

])