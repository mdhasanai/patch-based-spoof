import os
import argparse

parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--train_dataset', help='path to dev.csv is required', default="msu")
args = vars(parser.parse_args())

# os.system('python eval_keypoint.py --eval_dataset msu --model_type msu --image_size 96')
# print("MSU evaluated")

# os.system('python eval_keypoint.py --eval_dataset oulu --model_type msu --image_size 96')
# print("OULU evaluated")

#protocol

os.system('python eval_keypoint.py --eval_dataset msu --model_type oulu --protocol Protocol_1 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_1 --data_protocol Protocol_1 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_1 --data_protocol Protocol_2 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_1 --data_protocol Protocol_3 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_1 --data_protocol Protocol_4 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset mobile_replay --model_type oulu --protocol Protocol_1 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset nagod --model_type oulu --protocol Protocol_1 --image_size 48')
print("OULU evaluated")



os.system('python eval_keypoint.py --eval_dataset msu --model_type oulu --protocol Protocol_2 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_2 --data_protocol Protocol_1 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_2 --data_protocol Protocol_2 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_2 --data_protocol Protocol_3 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_2 --data_protocol Protocol_4 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset mobile_replay --model_type oulu --protocol Protocol_2 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset nagod --model_type oulu --protocol Protocol_2 --image_size 48')
print("OULU evaluated")





os.system('python eval_keypoint.py --eval_dataset msu --model_type oulu --protocol Protocol_3 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_3 --data_protocol Protocol_1 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_3 --data_protocol Protocol_2 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_3 --data_protocol Protocol_3 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_3 --data_protocol Protocol_4 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset mobile_replay --model_type oulu --protocol Protocol_3 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset nagod --model_type oulu --protocol Protocol_3 --image_size 48')
print("OULU evaluated")




os.system('python eval_keypoint.py --eval_dataset msu --model_type oulu --protocol Protocol_4 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_4 --data_protocol Protocol_1 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_4 --data_protocol Protocol_2 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_4 --data_protocol Protocol_3 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset oulu --model_type oulu --protocol Protocol_4 --data_protocol Protocol_4 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset mobile_replay --model_type oulu --protocol Protocol_4 --image_size 48')
print("OULU evaluated")

os.system('python eval_keypoint.py --eval_dataset nagod --model_type oulu --protocol Protocol_4 --image_size 48')
print("OULU evaluated")






