import sys
import os
import cv2
import json
import numpy as np
from os import listdir

path = sys.argv[1]
dest_path = sys.argv[2]

images_path = dest_path + 'images'

if ('mp4' in path.split('/')[-1]) or ('avi' in path.split('/')[-1]):
    print("Video processing....")

    try:
        os.mkdir(images_path)
    except OSError:
        print ("Creation of the directory %s failed" % images_path) 
    else:
        print ("Successfully created the directory %s " % images_path)

    cap = cv2.VideoCapture(path)
    folder_num = 0
    picture_num = 0
    success,image = cap.read()
    count = 0
    success = True
    while(success):
        if picture_num == 0:
            try:
                curr_path = images_path+'/'+str(folder_num)
                os.mkdir(curr_path)
                folder_num += 1
            except OSError:
                print ("Creation of the directory failed")
            else:
                print ("Successfully created the directory")

        success,frame = cap.read()
        #640:420
    
        if folder_num == 10:
            break

        cv2.imwrite(curr_path + '/' + str(picture_num) + '.png', frame[160:580, 300:940])

        picture_num += 1

        if picture_num == 201:
            picture_num = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    images_path = path


rat_info = dict(description= 'Rats', url= 'https://rats.org/home', version= '0.1', year= 2020, contributor= 'ELTE', date_created= '2020-01-11 00:55:41.903634')
rat_licenses = [dict(url= 'https://creativecommons.org/licenses/by/4.0/', id= 1, name= 'Creative Commons Attribution 4.0 License')]
rat_categories = [dict(supercategory= 'object', id= 1, name ='rat')]

def imFunc(e):
  return int(e[:-4])

def get_folders(path):
    f = np.array(listdir(path))
    g = np.char.find(f, 'aug')
    c = np.array(f[g == -1])
    
    folders.sort()
    folders = [path + '/' + str(folder) for folder in folders]
    return folders


def get_pngs(path, scene):
    l = listdir(path)
    l.sort(key=imFunc)
    l = [os.path.join(scene, ll) for ll in l]
    return l

rat_data = dict(info=rat_info, 
                    licenses=rat_licenses,
                    categories=rat_categories,
                    videos=[],
                    annotations=[])

scenes = [scene for scene in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, scene))]
scenes.sort(key=int)

for e, scene in enumerate(scenes):
    all_paths = get_pngs(os.path.join(images_path, scene), scene)

    print(scene)
    video = dict(width= 640,
                        length= len(all_paths),
                        date_captured= '',
                        license= '',
                        flickr_url= '',
                        file_names= [],
                        id= scene,
                        coco_url= '',
                        height=420)

    video['file_names'] = list(all_paths)

    rat_data['videos'].append(video)

    

with open(os.path.join(dest_path, 'ann.json'), 'w') as outfile:
    json.dump(rat_data, outfile)






