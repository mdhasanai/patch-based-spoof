import random
import glob
import os
import cv2
import sys, time
# from facerec.arcface_detection import FaceDetection
from facerec.retinaface_detection import RetinaDetector
import numpy as np
import argparse
path = '/home/ec2-user/SageMaker/dataset/spoof-data/nogod_images'
save_dir = '/home/ec2-user/SageMaker/dataset/spoof-data/nogod_frames_bck/batch_3'
fake = glob.glob(f"{path}/batch_3/fake/*")
real = glob.glob(f"{path}/batch_3/real/*")
# print(save_dir)
def rotate_image(face):
    out = cv2.transpose(face)
    return cv2.flip(out,flipCode=1)
def align_face(det, face, image_size):
    try:
#         face = rotate_image(true_face)
        aligned = det.get_input(face)
        aligned = np.transpose(aligned[0][0], (1, 2, 0))
#         aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        return aligned
    except Exception as e:
        print(f'could not detect face: {e}')
        

        

def save_vid_frames(detector, file_names):
#     dtype = file_names[0].split("/")[-2]
    frame_cnt = 0
    not_detected = 0
    for i, im_file in enumerate(file_names):
        start = time.time()
        dtype_cnt = 0
        label = "real" if im_file.split("/")[-2] == 'real' else "spoof"
     
        save_path = f"{save_dir}/{label}"
           
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            frame_cnt = len(os.listdir(save_path))
            
        
        
        frame = cv2.imread(im_file)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'random/{random.randint(1, 20)}.jpg', frame)
        frame_cnt += 1
        aligned = align_face(detector, frame, '112,112')
        
        if aligned is not None:
            dtype_cnt += 1
            cv2.imwrite(f'{save_path}/{os.path.split(im_file)[-1]}', aligned)
#             print(f"saved {dtype_cnt} frames")
        else:
            not_detected += 1
            
    print(f"Total Not detected images: {not_detected}")
#       print()  
#     print(f'DONE WITH {dtype}')

    
parser = argparse.ArgumentParser(description='Define Data type...')
parser.add_argument('--dtype', default="fake")
parser.add_argument('--gpu', type=int, default=6)
args = parser.parse_args()

fake, real = sorted(fake), sorted(real)
detector = RetinaDetector(args.gpu)
# print(eval(args.dtype))
save_vid_frames(detector, eval(args.dtype))
# for files in [front, real, back]:
#     save_vid_frames(detector, files)

