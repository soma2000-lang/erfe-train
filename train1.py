import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from config import (DEVICE, INPUT_SHAPE, NUM_SEG_CLASSES, NUM_LINE_CLASSES, 
                   BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS)
from model import ERFE
from loss import CombinedLoss,SegmentationLoss
from dataloader import RunwayDataset

def folder_check(folder_path):
 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def train(model, dataloader, device, optimizer, criterion):

    model.train()
    running_loss = 0
    counter = 0
    
    for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
                                                    desc="Training loop", 
                                                    total=len(dataloader)):
        counter += 1
        images = images.to(device)
        seg_true = seg_true.to(device)
        line_true = line_true.to(device)
        
        optimizer.zero_grad()
        seg_pred, line_pred = model(images)
        loss = criterion((seg_pred, line_pred), (seg_true, line_true))
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    
    training_loss = running_loss / counter
    return training_loss

def eval(model, dataloader, device, criterion):

    model.eval()
    running_loss = 0
    counter = 0
    
    with torch.no_grad():
        for idx, (images, (seg_true, line_true)) in tqdm(enumerate(dataloader), 
                                                        desc="Validation loop", 
                                                        total=len(dataloader)):
            counter += 1
            images = images.to(device)
            seg_true = seg_true.to(device)
            line_true = line_true.to(device)
            
            seg_pred, line_pred = model(images)
            loss = criterion((seg_pred, line_pred), (seg_true, line_true))
            
            running_loss += loss.item()
    
    validation_loss = running_loss / counter
    return validation_loss

def training_loop(epochs, model, train_loader, val_loader, device, optimizer, criterion, scheduler=None):
 
    
    train_loss_history = []
    valid_loss_history = []
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Creating checkpoint directory
    checkpoint_dir = 'checkpoints'
    folder_check(checkpoint_dir)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
  
        train_epoch_loss = train(model, train_loader, device, optimizer, criterion)
        valid_epoch_loss = eval(model, val_loader, device, criterion)
      
        train_loss_history.append(train_epoch_loss)
        valid_loss_history.append(valid_epoch_loss)
        
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")
        

        if scheduler is not None:
            scheduler.step(valid_epoch_loss)
        
        # Saving best model
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'{checkpoint_dir}/runway_segmentation_epoch_{epoch}_valLoss_{valid_epoch_loss:.3f}.pth')
            torch.save(model.state_dict(), 'runway_segmentation_best_model.pth')
            print(f"\nModel saved at epoch: {epoch + 1} \n")
        
        print(f"------ End of Epoch {epoch + 1} -------")
    
    # Load best model
    model.load_state_dict(torch.load('runway_segmentation_best_model.pth'))
    
    return model, train_loss_history, valid_loss_history, best_val_loss, best_epoch

def loss_plot(train_loss, valid_loss):
    """Plot and save training and validation loss curves."""
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('runway_segmentation_loss.png')
    plt.close()

def evaluate_tp_tn_rate(model, val_loader, device, threshold=0.1):
    """
    Evaluate the segmentation performance with TP+TN rate as described in the paper.
    This calculates the rate of correctly classified pixels (both true positives and true negatives).
    """
    model.eval()
    
    # Define class names for printing results
    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
    
    # Initialize counters for each class
    class_stats = {cls: {'total_pixels': 0, 'correct_pixels': 0} for cls in range(NUM_SEG_CLASSES)}
    
    # For line detection if applicable
    line_stats = {cls: {'total_samples': 0, 'correct_samples': 0} for cls in range(NUM_LINE_CLASSES)} if NUM_LINE_CLASSES > 0 else {}
    
    with torch.no_grad():
        # Process segmentation metrics
        for images, (seg_true, line_true) in tqdm(val_loader, desc="Evaluating TP+TN Rate"):
            images = images.to(device)
            seg_true = seg_true.to(device)
            
            # Get model predictions
            seg_pred, line_pred = model(images)
            
            # Process segmentation predictions
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred_classes = torch.argmax(seg_pred, dim=1)
            
            # Calculate TP+TN for each class in the batch
            batch_size = images.size(0)
            
            for i in range(batch_size):
                for cls in range(NUM_SEG_CLASSES):
                    # Getting binary masks for this class  we could have used the rle also inplace of this
                    true_mask = (seg_true[i] == cls)
                    pred_mask = (seg_pred_classes[i] == cls)
                    
                    # Calculating TP (correctly predicted class pixels)
                    tp = torch.logical_and(true_mask, pred_mask).sum().item()
                    
                    # Calculating  TN (correctly predicted non-class pixels)
                    tn = torch.logical_and(~true_mask, ~pred_mask).sum().item()
                    
                  
                    class_stats[cls]['total_pixels'] += true_mask.numel()
                    class_stats[cls]['correct_pixels'] += (tp + tn)
            
            # Process line predictions if applicable and if line_true is provided
            if NUM_LINE_CLASSES > 0 and line_true is not None:
                # Threshold the line probability maps
                for cls in range(NUM_LINE_CLASSES):
                    for i in range(batch_size):
                        # Check if line exists (max value exceeds threshold)
                        if hasattr(line_true[i, cls], 'max'):  # Check if it's a tensor
                            ground_truth_line = line_true[i, cls].max() > threshold
                            line_detected = line_pred[i, cls].max() > threshold
                            
                            line_stats[cls]['total_samples'] += 1
                            
                            # Count TP and TN
                            if (line_detected and ground_truth_line) or (not line_detected and not ground_truth_line):
                                line_stats[cls]['correct_samples'] += 1
    
    # Calculate TP+TN rates for segmentation
    class_tp_tn_rates = {}
    for cls in range(NUM_SEG_CLASSES):
        if class_stats[cls]['total_pixels'] > 0:
            tp_tn_rate = class_stats[cls]['correct_pixels'] / class_stats[cls]['total_pixels']
            class_tp_tn_rates[cls] = tp_tn_rate
    
    # Calculate TP+TN rates for line detection not required
    line_tp_tn_rates = {}
    if NUM_LINE_CLASSES > 0:
        line_class_names = ["Left Edge", "Center Line", "Right Edge", "Aiming Point", "Threshold"]
        for cls in range(NUM_LINE_CLASSES):
            if cls in line_stats and line_stats[cls]['total_samples'] > 0:
                tp_tn_rate = line_stats[cls]['correct_samples'] / line_stats[cls]['total_samples']
                line_tp_tn_rates[cls] = tp_tn_rate
    
    # Print results
    print("\nSegmentation TP+TN Rates:")
    for cls in range(NUM_SEG_CLASSES):
        if cls in class_tp_tn_rates:
            print(f"{class_names[cls]}: {class_tp_tn_rates[cls]:.4f}")
    
    if NUM_LINE_CLASSES > 0 and line_tp_tn_rates:
        print("\nLine Detection TP+TN Rates:")
        for cls in range(NUM_LINE_CLASSES):
            if cls in line_tp_tn_rates:
                line_name = line_class_names[cls] if cls < len(line_class_names) else f"Line {cls}"
                print(f"{line_name}: {line_tp_tn_rates[cls]:.4f}")
    
    # Calculate overall rates
    overall_seg_tp_tn_rate = sum(class_tp_tn_rates.values()) / len(class_tp_tn_rates) if class_tp_tn_rates else 0
    overall_line_tp_tn_rate = sum(line_tp_tn_rates.values()) / len(line_tp_tn_rates) if line_tp_tn_rates else 0
    
    print(f"\nOverall Segmentation TP+TN Rate: {overall_seg_tp_tn_rate:.4f}")
    if NUM_LINE_CLASSES > 0 and line_tp_tn_rates:
        print(f"Overall Line Detection TP+TN Rate: {overall_line_tp_tn_rate:.4f}")
    
    return class_tp_tn_rates, line_tp_tn_rates

def visualize_tp_tn_rates(class_tp_tn_rates, line_tp_tn_rates=None, epoch=None):
    """
    Visualize TP+TN rates using bar charts similar to the paper's figures.
    """
    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
    line_class_names = ["Left Edge", "Center Line", "Right Edge", "Aiming Point", "Threshold"]
    
    # Set up the figure
    num_plots = 2 if line_tp_tn_rates else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]  # Make it iterable for single plot case
    
    # Plot segmentation TP+TN rates
    ax = axes[0]
    classes = list(class_tp_tn_rates.keys())
    rates = [class_tp_tn_rates[cls] for cls in classes]
    class_labels = [class_names[cls] for cls in classes]
    
    bars = ax.bar(class_labels, rates, color='skyblue')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('TP+TN Rate')
    ax.set_title(f'Segmentation TP+TN Rates{" (Epoch "+str(epoch)+")" if epoch else ""}')
    
    # Add rate values on top of the bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Plot line detection TP+TN rates if available
    if line_tp_tn_rates:
        ax = axes[1]
        line_classes = list(line_tp_tn_rates.keys())
        line_rates = [line_tp_tn_rates[cls] for cls in line_classes]
        line_labels = [line_class_names[cls] if cls < len(line_class_names) else f"Line {cls}" 
                      for cls in line_classes]
        
        bars = ax.bar(line_labels, line_rates, color='lightgreen')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('TP+TN Rate')
        ax.set_title(f'Line Detection TP+TN Rates{" (Epoch "+str(epoch)+")" if epoch else ""}')
        
        # Add rate values on top of the bars
        for bar, rate in zip(bars, line_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_name = f'tp_tn_rates{"_epoch_"+str(epoch) if epoch else ""}.png'
    plt.savefig(save_name)
    plt.close()
    
    return save_name

def visualize_segmentation_results(model, val_loader, device, num_samples=3):
    """
    Visualize segmentation predictions for a few validation samples.
    """
    model.eval()
    
    # Getting a few samples from validation set
    samples = []
    with torch.no_grad():
        for images, (seg_true, _) in val_loader:
            if len(samples) >= num_samples:
                break
            
  
            batch_size = min(num_samples - len(samples), images.size(0))
            for i in range(batch_size):
                samples.append((images[i:i+1], seg_true[i:i+1]))
    
  
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    # Class colors for visualization a given in the dataset runaway fs
    colors = [
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Runway Area - red
        [255, 255, 0],    # Aiming Point Marking - yellow
        [0, 255, 0]       # Threshold Marking - green
    ]
    colors = np.array(colors)
    
    with torch.no_grad():
        for i, (image, seg_true) in enumerate(samples):
            image = image.to(device)
            seg_true = seg_true.to(device)
            

            seg_pred, _ = model(image)
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred = torch.argmax(seg_pred, dim=1)
            image = image.cpu().numpy()[0]
            image = np.transpose(image, (1, 2, 0))
            # Denormalize image
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            
            seg_true = seg_true.cpu().numpy()[0]
            seg_pred = seg_pred.cpu().numpy()[0]
            
            # Create colored segmentation maps
            true_colored = np.zeros((seg_true.shape[0], seg_true.shape[1], 3), dtype=np.uint8)
            pred_colored = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
            
            for cls in range(NUM_SEG_CLASSES):
                true_colored[seg_true == cls] = colors[cls]
                pred_colored[seg_pred == cls] = colors[cls]
            
            # Plot
            axs[i, 0].imshow(image)
            axs[i, 0].set_title('Input Image')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(true_colored)
            axs[i, 1].set_title('Ground Truth')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(pred_colored)
            axs[i, 2].set_title('Prediction')
            axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png')
    plt.close()

if __name__ == "__main__":
    # Define paths to data
    image_dir = "/home/AD/smajumder/runaway/640x360_dataset/640x360/train"
    mask_dir = "/home/AD/smajumder/runaway/resizded_trainimages_640x360"
    line_paths_dir = "/home/AD/smajumder/runaway/train_labels_640x360.json"
    
    # Get all image paths - ensure we're handling paths correctly
    image_paths = []
    for fname in os.listdir(image_dir):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            image_paths.append(os.path.join(image_dir, fname))
    
    # Get corresponding mask paths - adjust pattern based on your actual naming convention
    mask_paths = []
    valid_image_paths = []
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.png', '_area_label.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
            valid_image_paths.append(img_path)
        else:
            print(f"Warning: Mask not found for {img_path}")
    

    image_paths = valid_image_paths
    
    # Since we're dealing with semantic segmentation only, we'll use empty line paths
    # The dataloader will need to be adjusted to handle this
    line_paths = [line_paths_dir] * len(image_paths) if os.path.exists(line_paths_dir) else [""] * len(image_paths)
    
    # Split into train and validation
    train_idx = int(0.8 * len(image_paths))
    
    train_dataset = RunwayDataset(
        image_paths[:train_idx],
        mask_paths[:train_idx],
        line_paths[:train_idx],
        augment=True
    )
    
    val_dataset = RunwayDataset(
        image_paths[train_idx:],
        mask_paths[train_idx:],
        line_paths[train_idx:]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    
    
    model = ERFE(input_shape=INPUT_SHAPE, num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
    model = model.to(DEVICE)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    

    model, train_loss, valid_loss, best_val_loss, best_epoch = training_loop(
        epochs=NUM_EPOCHS, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=DEVICE, 
        optimizer=optimizer, 
        criterion=criterion,
        scheduler=scheduler
    )

    loss_plot(train_loss, valid_loss)

    class_tp_tn_rates, line_tp_tn_rates = evaluate_tp_tn_rate(model, val_loader, DEVICE)
    
    # Visualize the TP+TN rates
    visualize_tp_tn_rates(class_tp_tn_rates, line_tp_tn_rates)
    
    # Visualize some results from the validation set
    visualize_segmentation_results(model, val_loader, DEVICE, num_samples=3)
    
    print("Training complete!")