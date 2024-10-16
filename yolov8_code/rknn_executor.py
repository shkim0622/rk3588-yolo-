#from rknn.api import RKNN
from rknnlite.api import RKNNLite
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

def initRKNN(rknnModel="./yolov8s_rknnopt_RK3588_fp.rknn",id=0):

    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
 
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

def initRKNNs(rknnModel="./yolov8s_rknnopt_RK3588_fp.rknn",TPEs=3):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))

    return rknn_list


class RKNN_model_container():
    def __init__(self, model_path, TPEs, func) -> None:  
    #def __init__(self,func,TPEs) :  
        
        self.TPEs = TPEs 
        self.queue = Queue()
        self.rknnPool = initRKNNs(model_path, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs) 
        self.func = func
        self.num = 0
        
    def put(self, frame):
        self.queue.put(self.pool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame))
        
        #future=self.pool.submit(self.func,self.num % self.TPEs)
        self.num += 1
        #self.queue.put(future)
    
    def get(self):
        if self.queue.empty():
            print("get empty")
            return None, False

        fut = self.queue.get()
        print(fut)
        return fut.result(), True
        # return fut, True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
