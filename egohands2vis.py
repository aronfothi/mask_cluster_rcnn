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
DATASET_PATH = os.path.join(DOWNLOAD_DIR, '_LABELLED_SAMPLES')
TRAINING_FILE = 'train_files.csv'
TESTING_FILE = 'test_files.csv'

# Get all directories with samples
sample_directories = [f for f in pathlib.Path(DATASET_PATH).iterdir() if f.is_dir()]
print('{} folders found.'.format(len(sample_directories)))


# Get all samples for each directory
def get_path_polygons(directory):
    annotation_path = directory.joinpath('polygons.mat')
    mat = scipy.io.loadmat(annotation_path.resolve())
    # Load polygons data structure

    
    polygons = mat['polygons'][0]
    
    return polygons


hands_info = dict(description= 'Hands', url= 'http://vision.soic.indiana.edu/projects/egohands/', version= '0.1', year= 2020, contributor= 'Indiana', date_created= '2015 00:55:41.903634')
hands_licenses = [dict(url= 'https://creativecommons.org/licenses/by/4.0/', id= 1, name= 'Creative Commons Attribution 4.0 License')]
hands_categories = [dict(supercategory= 'object', id= 1, name ='hand')]

def annotation_data(folders, vid_id, ann_id, exists=False):
    hand_data = dict(info=hands_info, 
                    licenses=hands_licenses,
                    categories=hands_categories,
                    videos=[],
                    annotations=[])
    
    
    for directory in folders:
        
        
        print(sorted(directory.glob('*.jpg'))[0])
        im_path = str(sorted(directory.glob('*.jpg'))[0])
        img = cv2.imread(im_path)
        
        video_polygons = get_path_polygons(directory)
        
        video = dict(width= img.shape[1],
                     length= len(sorted(directory.glob('*.jpg'))),
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
            
        if not exists:           
            for polygons, frame_path in zip(video_polygons, sorted(directory.glob('*.jpg'))):
                file_name = str(frame_path).split(os.sep)
                
                file_name = os.path.join(*file_name[-2:])
                
                video['file_names'].append(file_name)
                
                for inst_id, polygon in zip(annotations, list(polygons)):
                    
                    if polygon.shape[0]>1:
                        #polygon = polygon.astype(int).astype(float)


                        #polygon[:, 0], polygon[:, 1] = polygon[:, 1], polygon[:, 0].copy()
                        
                        
                        polygon = polygon.transpose()
                        

                        contour = [j for i in zip(polygon[0],polygon[1]) for j in i]


                        rles = mask.frPyObjects([contour],img.shape[0],img.shape[1])

                        
                        rle = mask.merge(rles)
                        area = mask.area(rle)
                        bounding_box = mask.toBbox(rle)
                        annotations[inst_id]['bboxes'].append(bounding_box.tolist())
                        annotations[inst_id]['areas'].append(int(area))

                        rle['counts'] = rle['counts'].decode('ascii') 
                        annotations[inst_id]['segmentations'].append(rle)
                        
                    else:
                        annotations[inst_id]['segmentations'].append(None)
                        annotations[inst_id]['bboxes'].append(None)
                        annotations[inst_id]['areas'].append(None)

                            
        for _, ann in annotations.items():
            hand_data['annotations'].append(ann)
        
        hand_data['videos'].append(video)
        vid_id += 1
        
        
    return hand_data, vid_id, ann_id

def transform(json_root, sample_directories):
    vid_id = 1
    ann_id = 1    

    for s, sample in enumerate(sample_directories):

        if not os.path.exists(os.path.join(json_root, '{}.json'.format(s))):
            annotation, vid_id, ann_id = annotation_data([sample], vid_id, ann_id)
            with open(os.path.join(json_root, '{}.json'.format(s)), 'w') as outfile:            
                json.dump(annotation, outfile)
        else:
            _, vid_id, ann_id = annotation_data([sample], vid_id, ann_id, exists=True)

        print(vid_id)

json_train = '/home/fothar/mask_cluster_rcnn/jsons/train'
json_val = '/home/fothar/mask_cluster_rcnn/jsons/val'

transform(json_train, sample_directories[:40])
transform(json_val, sample_directories[40:])

def concat(json_root, input_files, out_path):
    hand_data = dict(info=hands_info, 
                        licenses=hands_licenses,
                        categories=hands_categories,
                        videos=[],
                        annotations=[])

    for vide_json in input_files:
        print(vide_json)
        with open(os.path.join(json_root, vide_json), 'r') as f:
            video_data = json.load(f)
            assert len(video_data['videos']) == 1 and len(video_data['annotations']) == 4
            hand_data['videos'] += video_data['videos']
            hand_data['annotations'] += video_data['annotations']


    with open(out_path, 'w') as outfile:            
        json.dump(hand_data, outfile)


concat(json_train, os.listdir(json_train), '/media/hdd/aron/egohands/annotations/instances_train.json')

concat(json_val, os.listdir(json_val), '/media/hdd/aron/egohands/annotations/instances_val.json')
    