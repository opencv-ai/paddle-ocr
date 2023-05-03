import onnx
import torch
import torch.nn as nn
from fix_recognizer import fix_reduce_mean
from onnx import shape_inference
class Dummy(nn.Module):
    def forward(self, x):
        mean = torch.mean(x, dim=2, keepdim=True)
        diff = x - mean
        # mean2 = torch.mean(torch.pow(diff, 2), dim=2, keepdim=True) + 1e-6
        # mean2 = torch.sqrt(mean2)
        # out = diff / mean2

        return diff + 1 - 1


data = torch.rand((1, 152, 120))
model = Dummy()
torch.onnx.export(model, data, "dummy.onnx", input_names=["x"], output_names=["y"], opset_version=11)
model = onnx.load("dummy.onnx")
fix_reduce_mean(model)
for i in range(len(model.graph.value_info)):
    del model.graph.value_info[-1]
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model, True)
onnx.save(model, "dummy_fixed.onnx")
