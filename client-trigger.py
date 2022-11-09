import requests
import time

# Delay 10s 
time.sleep(10)

print('---sent a request---')
response = requests.get('http://127.0.0.1:5000/video_feed')
print(response)