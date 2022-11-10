import requests
import os
import json
from my_util import get_current_gps, convert_gps_to_address

WORK_DIR = os.getcwd()
BASE_URL = 'https://svls-api.duckieswat.com/api/v1'



def fetch_lost_vehicles_requests(url, keyword, limit):
    print(f'Request to {url} with keyword {keyword}')
    response = requests.get(url=url, params={
        keyword: keyword,
        limit: limit
    })
    print('received data from api')
    saveFile = os.path.join(WORK_DIR, 'saved', 'data.json')
    with open(saveFile, 'w') as file:
        file.write(json.dumps(response.json()))
        print('---------Done---------')

location = convert_gps_to_address(get_current_gps())

fetch_lost_vehicles_requests(f'{BASE_URL}/lost-vehicle-requests/find',
                             location,
                              10000)


