import cv2
import time
#from rknnpool import rknnPoolExecutor
from rknnpool import *

from func import myFunc
import numpy as np
import sys


output_save=None
#cap = cv2.VideoCapture('./AVM_5th_q_3_drive.mp4')
cap = cv2.VideoCapture(11)
modelPath = "./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn"

TPEs = 3
rknn_lite = initRKNN(modelPath,0)
# pool = rknnPoolExecutor(
#     rknnModel=modelPath,
#     TPEs=TPEs,
#     func=myFunc)

#error 검사
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            # del pool
            exit(-1)
        # pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    
    if not ret:
        break

    frame, outputs = myFunc(rknn_lite,frame)
 
    outputs = []
    outputs.append(np.arange(1632000).reshape(1, 255, 80, -1))
    outputs.append(np.arange(1632000).reshape(1, 255, 80, -1))
    outputs.append(np.arange(1632000).reshape(1, 255, 80, -1))
    
    print(outputs[0].shape)  #(1,255,80,80) 

    print(list(outputs[0].shape[-2:]))#(80,80)
    print(outputs[0].reshape([3,-1])) # 크게 3 개로 나눔

    input=outputs[0].reshape([3,-1]+list(outputs[0].shape[-2:]))
    print(input.shape)  #(3, 85, 80, 80)
    
   

    sys.exit(1)
    
    
    
    # if flag == False:
        # break
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30 FPS average frame rate :\t", 30 / (time.time() - loopTime), " frame")
        loopTime = time.time()

print("Overall average frame rate : ", frames / (time.time() - initTime))
cap.release()
cv2.destroyAllWindows()
# pool.release()
