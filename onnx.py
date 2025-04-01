import torch


model = torch.load('erfe_best_model.pth')
model.eval()

input = torch.randn(3,640, 360) 


torch.onnx.export(
    model,
    input,
    "erfe.onnx",
    input_names=["input"]
  

)

print("Model converted successfully to ONNX format.")