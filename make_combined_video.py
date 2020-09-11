import cv2
import os
import numpy as np

image_folder = 'viss'

mt_image_folder = '/home/fothar/MaskTrackRCNN/None/'



def imFunc(e):
  return int(e[:-4])

def sceneFunc(e):
  return int(e)


scenes = [scene for scene in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, scene))]
scenes.sort(key=sceneFunc)
for scene in scenes:
    print(scene)
    video = cv2.VideoWriter(os.path.join(image_folder, 'combined' + scene + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 24, (3*549, 360))
    scene_dir = os.path.join(image_folder, scene)

    images = [img for img in os.listdir(scene_dir) if img.endswith(".png")]
    
    
    images.sort(key=imFunc)

    for image in images:

        clus_img = cv2.imread(os.path.join(scene_dir, image))

        orig_img = clus_img[360:, :549]
        clus_res = clus_img[:360, :549]
        mt_img = cv2.imread(os.path.join(mt_image_folder, scene, image))

        mt_res = mt_img[:360]

        img = np.concatenate((orig_img, mt_res, clus_res), axis=1)


        video.write(img)
    '''
    for i in range(10):
        video.write(np.zeros((720, 1098, 3), dtype=np.uint8))
    '''
    cv2.destroyAllWindows()
    video.release()
