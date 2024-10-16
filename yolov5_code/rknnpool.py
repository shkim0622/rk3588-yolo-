from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed

#"./rknnModel/yolov5s.rknn" 존재하지않음
#def initRKNN(rknnModel="./rknnModel/yolov5s.rknn", id=0):
def initRKNN(rknnModel="./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn",id=0):
    #initRKNN(rknnModel=./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn",id=1)
    #initRKNN(rknnModel=./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn",id=2)
    
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    
    #yolov5s.rknn를 'rb'로 read
    #load_rknn : success(0)
    #thread = id  
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
        
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0) #1
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)#2
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)#4
        #여기까지 사용
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)#7
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")

    return rknn_lite

#"./rknnModel/yolov5s.rknn" 존재하지않음
# def initRKNNs(rknnModel="./rknnModel/yolov5s.rknn", TPEs=1):
def initRKNNs(rknnModel="./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn",TPEs=3):
    rknn_list = []
    for i in range(TPEs):# i = (0,1,2)
        rknn_list.append(initRKNN(rknnModel, i % 3))
        #initRKNN(rknnModel,0) 
        #initRKNN(rknnModel,1)
        #initRKNN(rknnModel,2)
        
    return rknn_list
    #rknn_list = [rknn_lite, rknn_lite, rknn_lite]

class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs #TPES = 3
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        
        # 비동기식 병렬 처리  max_workers =  3 
        self.pool = ThreadPoolExecutor(max_workers=TPEs) 
        
        #frame으로 detection 
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1
           
    def get(self):
        #비어있으면 실패
        if self.queue.empty():
            return None, False
        
        #put() 가져옴
        fut = self.queue.get()
        return fut.result(), True
        # queue 의 값을 popleft return해준다.
    
    def get_save(self):
        #비어있으면 실패
        if self.queue.empty():
            return None
        
        #put() 가져옴
        fut = self.queue.get()
        return fut.result() 
    
   
    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
