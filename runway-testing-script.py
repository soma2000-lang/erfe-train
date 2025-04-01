import cv2
import numpy as np
import json
import time
import os
import onnxruntime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from glob import glob
from model import ERFE
from config import DEVICE, INPUT_SHAPE, NUM_SEG_CLASSES, NUM_LINE_CLASSES

def folder_check(folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_onnx_session(onnx_model_path, use_gpu=True):
    """Create an optimized ONNX runtime session with GPU support if available"""

    providers = []
    print(onnxruntime.get_available_providers())
    if use_gpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print("Using CUDA for inference")
    else:
        if use_gpu:
            print("CUDA requested but not available. Falling back to CPU.")
        providers.append('CPUExecutionProvider')
        print("Using CPU for inference")
    
   
    session_options = onnxruntime.SessionOptions()
    

    session = onnxruntime.InferenceSession(
        onnx_model_path,
        sess_options=session_options,
        providers=providers
    )
    
    return session

def load_and_preprocess_image(image_path, input_shape):
    """Load and preprocess an image for the model"""

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape

    transform = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed = transform(image=image)
    processed_image = transformed["image"]
    
    # Convert to tensor and add batch dimension
    processed_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float()
    
    return image, processed_tensor, original_shape

def process_pytorch_model(model, image_tensor, device):
    """Process an image through the PyTorch model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        seg_pred, line_pred = model(image_tensor)
        

        seg_probs = F.softmax(seg_pred, dim=1)
        seg_pred_class = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
        
    return seg_pred_class, seg_probs.cpu().numpy(), line_pred.cpu().numpy()

def process_onnx_model(session, image_tensor):
    """Process an image through the ONNX model"""

    input_name = session.get_inputs()[0].name
    input_data = image_tensor.cpu().numpy()
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = time.time() - start_time
    
    # Processing  outputs (assuming segmentation output is first)
    seg_logits = outputs[0]  # Shape: [batch, classes, height, width]
    seg_probs = np.exp(seg_logits) / np.sum(np.exp(seg_logits), axis=1, keepdims=True)
    seg_pred_class = np.argmax(seg_probs, axis=1).squeeze()
    
    # If there's also line prediction output
    if len(outputs) > 1:
        line_pred = outputs[1]
    else:
        line_pred = None
    
    return seg_pred_class, seg_probs, line_pred, inference_time

def resize_to_original(mask, original_shape):
    """Resize mask to original image dimensions"""
    resized_mask = cv2.resize(
        mask.astype(np.uint8), 
        (original_shape[1], original_shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask

def calculate_iou(pred_mask, gt_mask, class_idx):
    """Calculate IoU for a specific class"""
    pred_class = (pred_mask == class_idx).astype(np.uint8)
    gt_class = (gt_mask == class_idx).astype(np.uint8)
    
    intersection = np.logical_and(pred_class, gt_class).sum()
    union = np.logical_or(pred_class, gt_class).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def process_batch_pytorch(model, image_paths, mask_paths, input_shape, device, batch_size=4):
    """Process a batch of images using PyTorch model"""
    model.eval()
    
    total_images = len(image_paths)
    class_ious = {i: [] for i in range(NUM_SEG_CLASSES)}
    all_ious = []
    processing_times = []
    
   
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_image_paths = image_paths[start_idx:end_idx]
        batch_mask_paths = mask_paths[start_idx:end_idx]
        current_batch_size = len(batch_image_paths)

        batch_images = []
        batch_shapes = []
        
        for img_path in batch_image_paths:
            image, processed_tensor, original_shape = load_and_preprocess_image(img_path, input_shape)
            batch_images.append(processed_tensor)
            batch_shapes.append(original_shape)
        

        batch_tensor = torch.cat(batch_images, dim=0)

        with torch.no_grad():
            start_time = time.time()
            batch_tensor = batch_tensor.to(device)
            seg_preds, _ = model(batch_tensor)
            seg_probs = F.softmax(seg_preds, dim=1)
            seg_pred_classes = torch.argmax(seg_probs, dim=1).cpu().numpy()
            batch_time = time.time() - start_time
            

            time_per_image = batch_time / current_batch_size
            processing_times.append(time_per_image)

        for i in range(current_batch_size):
   
            seg_pred = seg_pred_classes[i]
   
            resized_pred = resize_to_original(seg_pred, batch_shapes[i])
            
            gt_mask = cv2.imread(batch_mask_paths[i], cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Warning: Could not load mask at {batch_mask_paths[i]}")
                continue
            
            # Calculate IoU for each class
            image_iou = 0
            valid_classes = 0
            
            for class_idx in range(NUM_SEG_CLASSES):
                iou = calculate_iou(resized_pred, gt_mask, class_idx)
                class_ious[class_idx].append(iou)
                
                # Only count classes that appear in the ground truth
                if (gt_mask == class_idx).any():
                    image_iou += iou
                    valid_classes += 1
            
            # Average IoU for this image
            if valid_classes > 0:
                image_iou /= valid_classes
                all_ious.append(image_iou)
    
    # Calculate average IoU for each class and mean IoU
    class_miou = {class_idx: np.mean(ious) if ious else 0 for class_idx, ious in class_ious.items()}
    mean_iou = np.mean(all_ious) if all_ious else 0
    average_time = np.mean(processing_times) if processing_times else 0
    fps = 1.0 / average_time if average_time > 0 else 0
    
    return class_miou, mean_iou, fps

def visualize_segmentation_result(image, pred_mask, gt_mask, save_path=None):
    """Visualize segmentation prediction vs ground truth"""
    # Define color map for visualization
    colors = [
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Runway Area - red
        [255, 255, 0],    # Aiming Point Marking - yellow
        [0, 255, 0]       # Threshold Marking - green
    ]
    colors = np.array(colors)
    
    # Create colored masks
    pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    
    for class_idx in range(len(colors)):
        pred_colored[pred_mask == class_idx] = colors[class_idx]
        gt_colored[gt_mask == class_idx] = colors[class_idx]
    
    # Create blended overlay with original image
    alpha = 0.5
    pred_overlay = cv2.addWeighted(image, 1-alpha, pred_colored, alpha, 0)
    gt_overlay = cv2.addWeighted(image, 1-alpha, gt_colored, alpha, 0)
    
    # Create visualization figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_and_visualize(model, test_image_paths, test_mask_paths, input_shape, output_dir, device, num_samples=5):
    """Test model and visualize results for a few samples"""
    model.eval()
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    folder_check(vis_dir)
    
    # Randomly select samples to visualize
    num_test_images = min(len(test_image_paths), len(test_mask_paths))
    if num_samples > num_test_images:
        num_samples = num_test_images
    
    sample_indices = np.random.choice(num_test_images, num_samples, replace=False)
    
    for idx in sample_indices:
        img_path = test_image_paths[idx]
        mask_path = test_mask_paths[idx]
        
        # Load image and preprocess
        image, processed_tensor, original_shape = load_and_preprocess_image(img_path, input_shape)
        
        # Process through model
        with torch.no_grad():
            processed_tensor = processed_tensor.to(device)
            seg_pred, _ = model(processed_tensor)
            seg_prob = F.softmax(seg_pred, dim=1)
            seg_pred_class = torch.argmax(seg_prob, dim=1).squeeze().cpu().numpy()
        
        # Resize to original image dimensions
        resized_pred = resize_to_original(seg_pred_class, original_shape)
        
        # Load ground truth mask
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: Could not load mask at {mask_path}")
            continue
        
        # Visualize
        base_filename = os.path.basename(img_path)
        save_path = os.path.join(vis_dir, f'vis_{base_filename}')
        visualize_segmentation_result(image, resized_pred, gt_mask, save_path)
        
        # Calculate and display metrics for this sample
        sample_ious = []
        for class_idx in range(NUM_SEG_CLASSES):
            iou = calculate_iou(resized_pred, gt_mask, class_idx)
            sample_ious.append(iou)
        
        print(f"Sample {base_filename} IoUs:")
        class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
        for class_idx, iou in enumerate(sample_ious):
            print(f"  {class_names[class_idx]}: {iou:.4f}")
        print(f"  Mean IoU: {np.mean(sample_ious):.4f}")
        print()

def main():
    # Configuration
    model_path = 'runway_segmentation_best_model.pth'  # Path to the trained PyTorch model
    onnx_model_path = 'runway_segmentation_model.onnx'  # Path to exported ONNX model (if available)
    test_images_dir =   # Directory containing test images
    test_masks_dir = 'dataset/masks/test'  # Directory containing test segmentation masks
    output_dir = 'test_results'  # Directory to save results
    batch_size = 8  # Adjust based on your GPU memory
    use_gpu = True  # Set to False to force CPU execution
    use_onnx = False  # Set to True to use ONNX model instead of PyTorch
    num_visualization_samples = 5  # Number of test samples to visualize
    "/home/AD/smajumder/runaway/640x360_dataset/640x360/train"
    # Create output directory
    folder_check(output_dir)
    
    # Get paths of test images and masks
    test_image_paths = sorted(glob(os.path.join(test_images_dir, '*.png')) + 
                             glob(os.path.join(test_images_dir, '*.jpg')))
    
    test_mask_paths = []
    for img_path in test_image_paths:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.jpg', '_area_label.png').replace('.png', '_area_label.png')
        mask_path = os.path.join(test_masks_dir, mask_name)
        if os.path.exists(mask_path):
            test_mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask not found for {img_path}")
    
            test_image_paths.remove(img_path)
    
    if len(test_image_paths) == 0:
        raise ValueError("No valid test images found!")
    
    print(f"Found {len(test_image_paths)} test images with corresponding masks")
    
  
    device = torch.device(DEVICE if use_gpu else "cpu")

    if use_onnx:
        print("Using ONNX model for inference")
        session = get_onnx_session(onnx_model_path, use_gpu=use_gpu)
        # We'll need to test each image individually for ONNX
        # Implementation continues below if needed
    else:
        print("Using PyTorch model for inference")
        model = ERFE(input_shape=INPUT_SHAPE, num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # Process test images in batches
        print("Testing model on dataset...")
        class_miou, mean_iou, fps = process_batch_pytorch(
            model, 
            test_image_paths, 
            test_mask_paths, 
            INPUT_SHAPE, 
            device, 
            batch_size
        )
        
        # Print metrics
        print("\nSegmentation Performance:")
        class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
        for class_idx, miou in class_miou.items():
            if class_idx < len(class_names):
                print(f"{class_names[class_idx]} IoU: {miou:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"FPS: {fps:.2f}")
        
        # Save metrics
        metrics = {
            'class_miou': {str(k): float(v) for k, v in class_miou.items()}, #may not be required also
            'mean_iou': float(mean_iou),
            'fps': float(fps)
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Visualize a few test samples
        print(f"\nGenerating visualizations for {num_visualization_samples} random samples...")
        test_and_visualize(
            model,
            test_image_paths,
            test_mask_paths,
            INPUT_SHAPE,
            output_dir,
            device,
            num_visualization_samples
        )
    
    print("Testing complete!")

if __name__ == "__main__":
    main()