import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import time
import os
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob


from config import DEVICE, INPUT_SHAPE, NUM_SEG_CLASSES, NUM_LINE_CLASSES
from model import ERFE

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
    import albumentations as A

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
    
   
    processed_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float()
    
    return image, processed_tensor, original_shape

def process_pytorch_model(model, image_tensor, device):
    """Process an image through the PyTorch model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        model_output = model(image_tensor)
        
  
        if isinstance(model_output, dict) and 'segmentation' in model_output:
            seg_output = model_output['segmentation']
            if isinstance(seg_output, dict) and 'out' in seg_output:
                seg_pred = seg_output['out']
            else:
                raise ValueError("Unexpected segmentation output structure")
                
            seg_probs = F.softmax(seg_pred, dim=1)
            seg_pred_class = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
            
        
            line_pred = None
            if 'line_probability' in model_output:
                line_pred = model_output['line_probability'].cpu().numpy()
        else:
            raise ValueError("Model output is not in expected format")
        
    return seg_pred_class, seg_probs.cpu().numpy(), line_pred

def process_onnx_model(session, image_tensor):
    """Process an image through the ONNX model"""
    input_name = session.get_inputs()[0].name
    input_data = image_tensor.cpu().numpy()
    
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = time.time() - start_time
    

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

def calculate_tp_tn_rate(pred_mask, gt_mask, class_idx):
    """Calculate TP+TN rate for a specific class"""
    pred_class = (pred_mask == class_idx).astype(np.uint8)
    gt_class = (gt_mask == class_idx).astype(np.uint8)
    
    # True Positives: Pixels correctly predicted as this class
    tp = np.logical_and(pred_class, gt_class).sum()
    
    # True Negatives: Pixels correctly predicted as NOT this class
    tn = np.logical_and(np.logical_not(pred_class), np.logical_not(gt_class)).sum()
 
    total_pixels = pred_class.size
    
    # TP+TN Rate
    if total_pixels == 0:
        return 0.0
    
    return (tp + tn) / total_pixels

def process_batch_pytorch(model, image_paths, mask_paths, input_shape, device, batch_size=4):
    """Process a batch of images using PyTorch model"""
    model.eval()
    
    total_images = len(image_paths)
    class_stats = {cls: {'total_pixels': 0, 'correct_pixels': 0} for cls in range(NUM_SEG_CLASSES)}
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
            
         
            model_output = model(batch_tensor)
            
            if isinstance(model_output, dict) and 'segmentation' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    print("Error: Unexpected segmentation output structure")
                    continue
            else:
                print("Error: Model output is not in expected format")
                continue
                
            seg_probs = F.softmax(seg_pred, dim=1)
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
            
           
            for class_idx in range(NUM_SEG_CLASSES):

                true_mask = (gt_mask == class_idx)
                pred_mask = (resized_pred == class_idx)
                
        
                tp = np.logical_and(true_mask, pred_mask).sum()
                
              
                tn = np.logical_and(np.logical_not(true_mask), np.logical_not(pred_mask)).sum()
                
               
                class_stats[class_idx]['total_pixels'] += true_mask.size
                class_stats[class_idx]['correct_pixels'] += (tp + tn)
    
  
    class_tp_tn_rates = {class_idx: 0 for class_idx in range(NUM_SEG_CLASSES)}
    for class_idx in range(NUM_SEG_CLASSES):
        if class_stats[class_idx]['total_pixels'] > 0:
            class_tp_tn_rates[class_idx] = class_stats[class_idx]['correct_pixels'] / class_stats[class_idx]['total_pixels']
    

    mean_tp_tn_rate = sum(class_tp_tn_rates.values()) / len(class_tp_tn_rates) if class_tp_tn_rates else 0
    
    # Calculate FPS
    average_time = np.mean(processing_times) if processing_times else 0
    fps = 1.0 / average_time if average_time > 0 else 0
    
    return class_tp_tn_rates, mean_tp_tn_rate, fps

def visualize_segmentation_result(image, pred_mask, gt_mask, save_path=None):
    """Visualize segmentation prediction vs ground truth"""
 
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
    
    # Creating blended overlay with original image
    alpha = 0.5
    pred_overlay = cv2.addWeighted(image, 1-alpha, pred_colored, alpha, 0)
    gt_overlay = cv2.addWeighted(image, 1-alpha, gt_colored, alpha, 0)
    
    # Creating visualization figure
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
    

    vis_dir = os.path.join(output_dir, 'visualizations')
    folder_check(vis_dir)
    
   
    num_test_images = min(len(test_image_paths), len(test_mask_paths))
    if num_samples > num_test_images:
        num_samples = num_test_images
    
    sample_indices = np.random.choice(num_test_images, num_samples, replace=False)
    
    for idx in sample_indices:
        img_path = test_image_paths[idx]
        mask_path = test_mask_paths[idx]
     
        image, processed_tensor, original_shape = load_and_preprocess_image(img_path, input_shape)
        

        with torch.no_grad():
            processed_tensor = processed_tensor.to(device)
         
            model_output = model(processed_tensor)
            

            if isinstance(model_output, dict) and 'segmentation' in model_output:
                seg_output = model_output['segmentation']
                if isinstance(seg_output, dict) and 'out' in seg_output:
                    seg_pred = seg_output['out']
                else:
                    print(f"Error: Unexpected segmentation output structure for {img_path}")
                    continue
            else:
                print(f"Error: Model output is not in expected format for {img_path}")
                continue
                
            seg_prob = F.softmax(seg_pred, dim=1)
            seg_pred_class = torch.argmax(seg_prob, dim=1).squeeze().cpu().numpy()
        
        # Resize to original image dimensions
        resized_pred = resize_to_original(seg_pred_class, original_shape)
        
        # Load ground truth mask
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: Could not load mask at {mask_path}")
            continue
        

        base_filename = os.path.basename(img_path)
        save_path = os.path.join(vis_dir, f'vis_{base_filename}')
        visualize_segmentation_result(image, resized_pred, gt_mask, save_path)
        
        # Calculate and display TP+TN rates for this sample
        sample_tp_tn_rates = []
        for class_idx in range(NUM_SEG_CLASSES):
            tp_tn_rate = calculate_tp_tn_rate(resized_pred, gt_mask, class_idx)
            sample_tp_tn_rates.append(tp_tn_rate)
        
        print(f"Sample {base_filename} TP+TN Rates:")
        class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
        for class_idx, rate in enumerate(sample_tp_tn_rates):
            print(f"  {class_names[class_idx]}: {rate:.4f}")
        print(f"  Mean TP+TN Rate: {np.mean(sample_tp_tn_rates):.4f}")
        print()

def visualize_tp_tn_rates(class_tp_tn_rates, output_dir):

    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]

    plt.figure(figsize=(10, 6))

    classes = list(class_tp_tn_rates.keys())
    rates = [class_tp_tn_rates[cls] for cls in classes]
    class_labels = [class_names[cls] for cls in classes]
    
    bars = plt.bar(class_labels, rates, color='skyblue')
    plt.ylim(0, 1.0)
    plt.ylabel('TP+TN Rate')
    plt.title('Segmentation TP+TN Rates')
    

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tp_tn_rates.png'))
    plt.close()

def main():
    # Configuration
    model_path = 'runway_segmentation_best_model.pth'  
    onnx_model_path = 'runway_segmentation_model.onnx'  
    test_images_dir = '/home/AD/smajumder/runaway/640x360_dataset/640x360/test'  
    test_masks_dir = '/home/AD/smajumder/runaway/resizzed_images_test_640x360'  
    output_dir = 'test_results' 
    batch_size = 3 
    use_gpu = True 
    use_onnx = False 
    num_visualization_samples = 5  


    
    
    # Create output directory
    folder_check(output_dir)
    
    # Get paths of test images and masks
    test_image_paths = sorted(glob(os.path.join(test_images_dir, '*.png')))
                          
    
    test_mask_paths = []
  
    for img_path in test_image_paths:
        img_name = os.path.basename(img_path)

        mask_name = img_name
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
        # Implementation for ONNX would need to be updated to match the TP+TN metrics approach
    else:
        print("Using PyTorch model for inference")
        model = ERFE(input_shape=INPUT_SHAPE, num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # Process test images in batches
        print("Testing model on dataset...")
        class_tp_tn_rates, mean_tp_tn_rate, fps = process_batch_pytorch(
            model, 
            test_image_paths, 
            test_mask_paths, 
            INPUT_SHAPE, 
            device, 
            batch_size
        )
        

        print("\nSegmentation Performance:")
        class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
        for class_idx, tp_tn_rate in class_tp_tn_rates.items():
            if class_idx < len(class_names):
                print(f"{class_names[class_idx]} TP+TN Rate: {tp_tn_rate:.4f}")
        print(f"Mean TP+TN Rate: {mean_tp_tn_rate:.4f}")
        print(f"FPS: {fps:.2f}")
        
        # Save metrics
        metrics = {
            'class_tp_tn_rates': {str(k): float(v) for k, v in class_tp_tn_rates.items()},
            'mean_tp_tn_rate': float(mean_tp_tn_rate),
            'fps': float(fps)
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Visualize TP+TN rates
        visualize_tp_tn_rates(class_tp_tn_rates, output_dir)
        
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