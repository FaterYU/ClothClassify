import cv2
from ultralytics import YOLO
import numpy as np


categoryL = {'0': 'negative', '1': 'short sleeve top', '2': 'long sleeve top', '3': 'short sleeve outwear', '4': 'long sleeve outwear', '5': 'vest', '6': 'sling',
             '7': 'shorts', '8': 'trousers', '9': 'skirt', '10': 'short sleeve dress', '11': 'long sleeve dress', '12': 'vest dress', '13': 'sling dress'}


segment_model = YOLO('models/segment.pt')

for i in range(1, 10):
    image = cv2.imread('input/' + str(i) + '.jpg')
    segment_results = segment_model(image)

    image = segment_results[0].plot()

    cv2.imwrite('output/' + str(i) + '_segment.jpg', image)
