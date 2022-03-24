import numpy as np
import onnx
import onnxruntime

file_path = (
    "./save_logs/03_Wed_19_49_15/model.onnx"
)
ort_session = onnxruntime.InferenceSession(file_path)
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: np.random.randn(1, 6, 64).astype(np.float32)}
ort_outs = ort_session.run(None, ort_inputs)[0]
print(ort_session.get_inputs()[0])
print(ort_session.get_outputs()[0])
print(ort_outs) # will need to use argmax to get class
print(type(ort_outs)) # numpy array
print(ort_outs.shape) # (1, num_class)
