import os
import sys
import pathlib
import json

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy
import scipy.io
import cv2
#from skimage import measure, io

from sklearn.model_selection import train_test_split


from pycocotools import mask

np.random.seed(seed=42)

DOWNLOAD_DIR = '/media/hdd/aron/egohands/'
DATASET_PATH = os.path.join(DOWNLOAD_DIR, 'all_frames')


# Get all directories with samples
sample_directories = [f for f in pathlib.Path(DATASET_PATH).iterdir() if f.is_dir()]
print('{} folders found.'.format(len(sample_directories)))



hands_info = dict(description= 'Hands', url= 'http://vision.soic.indiana.edu/projects/egohands/', version= '0.1', year= 2020, contributor= 'Indiana', date_created= '2015 00:55:41.903634')
hands_licenses = [dict(url= 'https://creativecommons.org/licenses/by/4.0/', id= 1, name= 'Creative Commons Attribution 4.0 License')]
hands_categories = [dict(supercategory= 'object', id= 1, name ='hand')]

 


def annotation_data(folders):
    vid_id = 1
    ann_id = 1   

    hand_data = dict(info=hands_info, 
                    licenses=hands_licenses,
                    categories=hands_categories,
                    videos=[],
                    annotations=[])
    
    
    for directory in folders:

        frames = sorted(directory.glob('*.jpeg'))
        print(frames[0])
        im_path = str(frames[0])
        img = cv2.imread(im_path)
        
        video = dict(width= img.shape[1],
                     length= len(frames),
                     date_captured= '',
                     license= '',
                     flickr_url= '',
                     file_names= [],
                     id= vid_id,
                     coco_url= '',
                     height=img.shape[0]) 
        
        annotations = {}
        for i in range(4):
            annotations[ann_id] = dict(height= img.shape[0],
                                    width= img.shape[1],
                                    length= 1,
                                    category_id= 1,
                                    segmentations= [],
                                    bboxes= [],
                                    video_id= vid_id,
                                    iscrowd= False,
                                    id= ann_id,
                                    areas= [])
            
            ann_id += 1
            
                
        for frame_path in frames:
            file_name = str(frame_path).split(os.sep)
            
            file_name = os.path.join(*file_name[-2:])
            
            video['file_names'].append(file_name)
            
            
                            
        for _, ann in annotations.items():
            hand_data['annotations'].append(ann)
        
        hand_data['videos'].append(video)
        vid_id += 1
        
        
    return hand_data


annotation = annotation_data(sample_directories[40:])

with open('/media/hdd/aron/egohands/annotations/instances_test.json', 'w') as outfile:            
    json.dump(annotation, outfile)

