import requests
import os

WORK_DIR = os.getcwd()
BASE_URL = 'https://svls-api.duckieswat.com/api/v1'



def fetch_lost_vehicles_requests(url, limit):
    response = requests.get(url=url, params={
        limit: limit
    })
    saveFile = os.path.join(WORK_DIR, 'saved', 'data.json')
    
    with open(saveFile, 'wb') as file:
        file.write(response.json())

fetch_lost_vehicles_requests(f'{BASE_URL}//lost-vehicle-requests/find', 10000)