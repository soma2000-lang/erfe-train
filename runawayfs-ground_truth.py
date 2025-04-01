import torch
from model import ERFE
import config
import os
from PIL import Image
import numpy as np

def verify_model_output_shape_with_single_images(image_path, mask_path):

    print(f"Loading image from: {image_path}")
    print(f"Loading mask from: {mask_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    

    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path)
    

    sample_image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0 # pytorch tensor
    sample_mask = torch.tensor(np.array(mask)).long() # segmentationn masks expect long only
    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")
    

    unique_classes = torch.unique(sample_mask)
    print(f"Unique classes in mask: {unique_classes}")
    print(f"Number of unique classes: {len(unique_classes)}")

    sample_image = sample_image.unsqueeze(0)
    

    model = ERFE(
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_line_classes=config.NUM_LINE_CLASSES
    )
    model.eval()
    

    with torch.no_grad():
        output = model(sample_image)
    
    print("\n--- Model Output Information ---")
    print("Output keys:", output.keys())
    

    seg_output = output["segmentation"]
    if isinstance(seg_output, dict) and "out" in seg_output:
        seg_tensor = seg_output["out"]
        print(f"Segmentation output shape: {seg_tensor.shape}")
        

        expected_classes = config.NUM_SEG_CLASSES + 1
        if seg_tensor.shape[1] == expected_classes:
            print(f"✓ Output channels ({seg_tensor.shape[1]}) match expected classes ({expected_classes})")
        else:
            print(f"✗ Output channels ({seg_tensor.shape[1]}) don't match expected classes ({expected_classes})")
        

        expected_h, expected_w = sample_image.shape[2], sample_image.shape[3]
        if seg_tensor.shape[2] == expected_h and seg_tensor.shape[3] == expected_w:
            print(f"✓ Output spatial dimensions ({seg_tensor.shape[2]}×{seg_tensor.shape[3]}) match input dimensions")
        else:
            print(f"✗ Output dimensions ({seg_tensor.shape[2]}×{seg_tensor.shape[3]}) don't match input ({expected_h}×{expected_w})")
    

    line_prob = output["line_probability"]
    print(f"Line probability output shape: {line_prob.shape}")
    expected_line_classes = config.NUM_LINE_CLASSES
    if line_prob.shape[1] == expected_line_classes:
        print(f"✓ Line probability channels ({line_prob.shape[1]}) match expected ({expected_line_classes})")
    else:
        print(f"✗ Line probability channels ({line_prob.shape[1]}) don't match expected ({expected_line_classes})")

if __name__ == "__main__":

    image_path = "/home/delen018/erfe/data/4AK606_1_1LDImage1_actual_train.png" 
    mask_path = "/home/delen018/erfe/data/4AK606_1_1LDImage1_area_label.png"   
    
    verify_model_output_shape_with_single_images(image_path, mask_path)