import torch


model = torch.load('runway_segmentation_best_model.pth')
model.eval()

input = torch.randn(3,640, 360) 


torch.onnx.export(
    model,
    input,
    "runway_segmentation_best_model.onnx",
    input_names=["input"]
  

)

print("Model converted successfully to ONNX format.")