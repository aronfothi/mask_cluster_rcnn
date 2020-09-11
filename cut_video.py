import cv2
import os
import numpy as np

image_folder = '/media/hdd/aron/rats/test_images/'
inp_video = '/media/hdd/aron/rats/Exp_2019_03_22_retrograd_study_20190415_160817.avi'

cap = cv2.VideoCapture(inp_video)

start_frame_idx = 30000
end_frame_idx = 60000

window = 2000
CUT_BBOX_YXHW = (160, 320, 420, 640)

f_idx = 0
w_idx = 0

while(cap.isOpened() and f_idx < end_frame_idx):
  ret, frame = cap.read()

  if not ret:
    break

  if start_frame_idx <= f_idx and f_idx < end_frame_idx:
    
    if start_frame_idx == f_idx or (f_idx-start_frame_idx) % window == 0:
      print('w_idx:', w_idx)
      if w_idx > 0:
        video.release()      

      scene_folder = os.path.join(image_folder, str(w_idx))
      os.makedirs(scene_folder, exist_ok=True)
      video = cv2.VideoWriter(os.path.join(image_folder, str(w_idx) + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 48, (640, 420))
      w_idx += 1

    frame = frame[CUT_BBOX_YXHW[0]:CUT_BBOX_YXHW[0]+CUT_BBOX_YXHW[2], \
      CUT_BBOX_YXHW[1]:CUT_BBOX_YXHW[1]+CUT_BBOX_YXHW[3],:]

    cv2.imwrite(os.path.join(scene_folder, str((f_idx-start_frame_idx) % window) + '.png'), frame)
    video.write(frame)

  #print(f_idx)
  f_idx += 1

cap.release()

