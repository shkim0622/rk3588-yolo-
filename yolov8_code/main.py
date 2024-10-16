import cv2
from yolo_map_test_rknn import myfunc
from common.framework_executor.rknn_executor import RKNN_model_container

model_path = "./yolov8s_rknnopt_RK3588_fp.rknn"
cap = cv2.VideoCapture(11)

TPEs = 3
rknn_executor = RKNN_model_container(
    model_path=model_path,
    TPEs=TPEs, 
    func = myfunc)

if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del rknn_executor
            exit(-1)
        rknn_executor.put(frame)

while (cap.isOpened()):
    success, frame = cap.read()
    if not success:
        break

    rknn_executor.put(frame)
    frame,flag= rknn_executor.get()
   
    if flag == False:
        print("get false")
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
rknn_executor.release()
    
