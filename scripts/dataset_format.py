import cv2
import os
import json
import tqdm

dataset_path = 'DeepFashion/'
style_dataset_path = 'DeepFashion-style/'
category_dataset_path = 'DeepFashion-category/'

if not os.path.exists(style_dataset_path + 'train'):
    os.makedirs(style_dataset_path + 'train')
    os.makedirs(style_dataset_path + 'validation')
    os.makedirs(style_dataset_path + 'test')

if not os.path.exists(category_dataset_path + 'train'):
    os.makedirs(category_dataset_path + 'train')
    os.makedirs(category_dataset_path + 'validation')
    os.makedirs(category_dataset_path + 'test')

styles = {}
categories = {}

operations = ['train', 'validation', 'test']

for operation in operations:
    if not os.path.exists(dataset_path + operation):
        continue
    images = os.listdir(dataset_path + operation + '/image/')
    print('Processing ' + operation + ' dataset, total images: ' + str(len(images)))
    
    for image_name in tqdm.tqdm(images):
        annos_name = image_name.replace('jpg', 'json')

        image = cv2.imread(dataset_path + operation + '/image/' + image_name)
        annos = json.load(open(dataset_path + operation + '/annos/' + annos_name))

        del annos['source']
        del annos['pair_id']
        for key in annos.keys():
            bounding_box = annos[key]['bounding_box']
            roi = image[int(bounding_box[1]):int(bounding_box[3]),
                        int(bounding_box[0]):int(bounding_box[2])]
            style = annos[key]['style']
            category = annos[key]['category_id']
            
            # set new dataset
            style_path = style_dataset_path + operation + '/' + str(style) + '/'
            category_path = category_dataset_path + operation + '/' + str(category) + '/'
            
            if styles.get(style) is None:
                styles[style] = 1
                os.makedirs(style_path)
            if categories.get(category) is None:
                categories[category] = 1
                os.makedirs(category_path)
            
            cv2.imwrite(style_path + image_name, roi)
            cv2.imwrite(category_path + image_name, roi)
