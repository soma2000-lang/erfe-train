import torch
from model import ERFE  
import config


dummy_input = torch.randn(3, 3, 360, 640)


model = ERFE(
    #input_shape=(3, 640, 360),
    num_seg_classes=config.NUM_SEG_CLASSES,
    num_line_classes=config.NUM_LINE_CLASSES
)

model.eval()

with torch.no_grad():
    output = model(dummy_input)


print("Output keys:", output.keys())


# LRASPP output
seg_output = output["segmentation"]
print("Segmentation output shape:", seg_output.keys())
if isinstance(seg_output, dict) and "out" in seg_output:
    seg_tensor = seg_output["out"]
    print("Segmentation output shape:", seg_tensor.shape)
else:
    print("Segmentation output shape:", seg_output.shape)


print("Line probability output shape:", output["line_probability"].shape)