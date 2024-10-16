import numpy as np

def myFunc(rknn_lite, IMG):
    outputs = rknn_lite.inference(inputs=[IMG])
    input0_data = outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:]))
    input1_data = outputs[1].reshape([3, -1]+list(outputs[1].shape[-2:]))
    input2_data = outputs[2].reshape([3, -1]+list(outputs[2].shape[-2:]))
 

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
    
    print(input_data)