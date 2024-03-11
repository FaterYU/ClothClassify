import cv2
from ultralytics import YOLO
import numpy as np




categoryL = {'1': 'short sleeve top', '2': 'long sleeve top', '3': 'short sleeve outwear', '4': 'long sleeve outwear', '5': 'vest', '6': 'sling',
             '7': 'shorts', '8': 'trousers', '9': 'skirt', '10': 'short sleeve dress', '11': 'long sleeve dress', '12': 'vest dress', '13': 'sling dress'}


category_model = YOLO('models/category.pt')
style_model = YOLO('models/style.pt')

for i in range(1,10):
    image = cv2.imread('input/' + str(i) + '.jpg')
    category_results = category_model(image)
    style_results = style_model(image)

    category_top5 = category_results[0].probs.top5
    category_top5conf = category_results[0].probs.top5conf

    # put text on image
    for j in range(5):
        cv2.putText(image, categoryL[str(category_top5[j]+1)] + ' ' + str(round(category_top5conf[j].item(
        ), 2)), (10, 30 + 30*j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite('output/' + str(i) + '_category.jpg', image)
