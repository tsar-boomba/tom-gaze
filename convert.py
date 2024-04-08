import onnx
from safetensors import torch as stt

MODEL = 'L2CSNet_gaze360'

model = onnx.helper.make_model(onnx.helper.make_model())
load_file(model, f'{MODEL}.safetensors')
onnx.save_model(model, 'model.onnx')
