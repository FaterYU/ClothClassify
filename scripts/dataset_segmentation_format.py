import cv2
import os
import json
import tqdm
import shutil

dataset_path = 'DeepFashion/'
segment_dataset_path = 'DeepFashion-segment/'

styles = {}
categories = {}

operations = ['train', 'validation']

if not os.path.exists(segment_dataset_path + 'images'):
    os.makedirs(segment_dataset_path + 'images')
    for operation in operations:
        if not os.path.exists(segment_dataset_path + 'images/' + operation):
            os.makedirs(segment_dataset_path + 'images/' + operation)
if not os.path.exists(segment_dataset_path + 'labels'):
    os.makedirs(segment_dataset_path + 'labels')
    for operation in operations:
        if not os.path.exists(segment_dataset_path + 'labels/' + operation):
            os.makedirs(segment_dataset_path + 'labels/' + operation)


for operation in operations:
    if not os.path.exists(dataset_path + operation):
        continue
    labels = os.listdir(dataset_path + operation + '/annos/')
    print('Processing ' + operation +
          ' dataset, total images: ' + str(len(labels)))

    for annos_name in tqdm.tqdm(labels):
        annos = json.load(
            open(dataset_path + operation + '/annos/' + annos_name))
        image_name = annos_name.replace('json', 'jpg')
        image = cv2.imread(dataset_path + operation + '/image/' + image_name)
        shape = image.shape

        del annos['source']
        del annos['pair_id']
        
        label = []
        
        for key in annos.keys():
            category = annos[key]['category_id']
            segmentation = annos[key]['segmentation'][0]
            
            # normalize segmentation
            for i in range(len(segmentation)):
                if i % 2 == 0:
                    segmentation[i] = segmentation[i] / shape[1]
                else:
                    segmentation[i] = segmentation[i] / shape[0]
                if segmentation[i] > 1:
                    segmentation[i] = 1

            label.append([category]+segmentation)
            
        try:
            assert os.path.exists(dataset_path + operation + '/image/' + image_name)
            shutil.copy(dataset_path + operation + '/image/' + image_name, segment_dataset_path + 'images/' + operation + '/' + image_name)
            with open(segment_dataset_path + 'labels/' + operation + '/' + annos_name.replace('json', 'txt'), 'w') as f:
                for l in label:
                    f.write(' '.join(map(str, l)) + '\n')
        except:
            print('Error in ' + image_name)
            continue
print('Done')
