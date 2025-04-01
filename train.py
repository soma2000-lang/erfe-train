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
from loss import CombinedLoss
from dataloader import RunwayDataset

def folder_check(folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def train(model, dataloader, device, optimizer, criterion):
    """Single epoch training function."""
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
    """Evaluation function."""
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
    """Complete training loop for all epochs."""
    
    train_loss_history = []
    valid_loss_history = []
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Create checkpoint directory
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
        
        # Adjust learning rate if scheduler provided
        if scheduler is not None:
            scheduler.step(valid_epoch_loss)
        
        # Save best model
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

def evaluate_segmentation(model, val_loader, device):
    """Evaluate the segmentation performance with IoU metric."""
    model.eval()
    total_iou = 0
    class_iou = {i: 0 for i in range(NUM_SEG_CLASSES)}
    class_counts = {i: 0 for i in range(NUM_SEG_CLASSES)}
    
    with torch.no_grad():
        for images, (seg_true, _) in tqdm(val_loader, desc="Evaluating Segmentation"):
            images = images.to(device)
            seg_true = seg_true.to(device)
            
            seg_pred, _ = model(images)
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred = torch.argmax(seg_pred, dim=1)
            
            # Calculate IoU for each class
            batch_size = images.size(0)
            
            for i in range(batch_size):
                for cls in range(NUM_SEG_CLASSES):
                    pred_mask = (seg_pred[i] == cls)
                    true_mask = (seg_true[i] == cls)
                    
                    # Only calculate IoU if this class exists in the ground truth
                    if true_mask.sum() > 0:
                        intersection = torch.logical_and(pred_mask, true_mask).sum().item()
                        union = torch.logical_or(pred_mask, true_mask).sum().item()
                        
                        iou = intersection / (union + 1e-10)  # Add small epsilon to prevent division by zero
                        total_iou += iou
                        class_iou[cls] += iou
                        class_counts[cls] += 1
    
    # Calculate mean IoU
    mean_iou = total_iou / sum(class_counts.values())
    
    # Calculate class-wise IoU
    class_names = ["Background", "Runway Area", "Aiming Point Marking", "Threshold Marking"]
    print("\nClass-wise IoU:")
    for cls in range(NUM_SEG_CLASSES):
        if class_counts[cls] > 0:
            cls_iou = class_iou[cls] / class_counts[cls]
            print(f"{class_names[cls]}: {cls_iou:.4f}")
    
    return mean_iou

def visualize_segmentation_results(model, val_loader, device, num_samples=3):
    """
    Visualize segmentation predictions for a few validation samples.
    """
    model.eval()
    
    # Get a few samples from validation set
    samples = []
    with torch.no_grad():
        for images, (seg_true, _) in val_loader:
            if len(samples) >= num_samples:
                break
            
            # Get batch results
            batch_size = min(num_samples - len(samples), images.size(0))
            for i in range(batch_size):
                samples.append((images[i:i+1], seg_true[i:i+1]))
    
    # Create figure
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    # Class colors for visualization
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
            
            # Forward pass
            seg_pred, _ = model(image)
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred = torch.argmax(seg_pred, dim=1)
            
            # Convert to numpy for visualization
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
    # Define your training and validation paths
    # Construct the dataset paths
    # Modify these paths to match your actual dataset structure
    image_dir = "dataset/images/train"
    mask_dir = "dataset/masks/train"
    
    # Get all image paths
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) 
                  if fname.endswith('.png') or fname.endswith('.jpg')]
    
    # Get corresponding mask paths - adjust pattern based on your actual naming convention
    mask_paths = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.jpg', '_area_label.png').replace('.png', '_area_label.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask not found for {img_path}")
            # Remove the corresponding image if mask doesn't exist
            image_paths.remove(img_path)
    
    # Since we're dealing with semantic segmentation only, we'll use empty line paths
    # The dataloader will need to be adjusted to handle this
    line_paths = [""] * len(image_paths)
    
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
    
    # Create data loaders
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
    
    # Set number of segmentation classes based on your usecase
    # Make sure this matches your config.py
    # NUM_SEG_CLASSES should be 4 (Background, Runway Area, Aiming Point Marking, Threshold Marking)
    
    model = ERFE(input_shape=INPUT_SHAPE, num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES)
    model = model.to(DEVICE)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    
    # Train the model
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

    # Plot and save loss curves
    loss_plot(train_loss, valid_loss)
    
    # Evaluate segmentation performance on validation set
    mean_iou = evaluate_segmentation(model, val_loader, DEVICE)
    print(f"Mean IoU on validation set: {mean_iou:.4f}")

    # Visualize some results from the validation set
    visualize_segmentation_results(model, val_loader, DEVICE, num_samples=3)
    
    print("Training complete!")