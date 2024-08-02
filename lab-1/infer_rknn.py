from rknnlite.api import RKNNLite
import numpy as np

N = 16
rknn_lite = RKNNLite()

# Load RKNN model
ret = rknn_lite.load_rknn(f'./models/matmul_model_{N}.rknn')
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)

# Initialize runtime
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    print('Init runtime failed')
    exit(ret)

# Get input shapes and prepare inputs
input_shapes = rknn_lite.get_input_shape()
print("Expected input shapes:", input_shapes)

input1 = np.random.rand(N, N).astype(np.float32)
input2 = np.random.rand(N, N).astype(np.float32)

# Perform inference
outputs = rknn_lite.inference(inputs=[input1, input2])
if outputs is not None:
    print(outputs[0])
else:
    print("Inference failed")

# Release resources
rknn_lite.release()
