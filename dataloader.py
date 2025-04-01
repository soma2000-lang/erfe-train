import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from util import gaussian_line_probability
from config import NUM_SEG_CLASSES, NUM_LINE_CLASSES

class RunwayDataset(Dataset):
    """
    Dataset for the ERFE model.
    """
    def __init__(self, image_paths, seg_paths, line_paths, input_shape=(3, 640, 360),
                 num_seg_classes=NUM_SEG_CLASSES, num_line_classes=NUM_LINE_CLASSES, augment=False):
        self.image_paths = image_paths
        self.seg_paths = seg_paths
        self.line_paths = line_paths
        self.input_shape = input_shape
        self.num_seg_classes = NUM_SEG_CLASSES
        self.num_line_classes = NUM_LINE_CLASSES
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # as has been mentioned in paper
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load segmentation mask
        seg = cv2.imread(self.seg_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Create one-hot encoded segmentation creates an array of shape (4, 640, 360) as I have 4 classes at max
        seg_one_hot = np.zeros((self.num_seg_classes, *seg.shape), dtype=np.float32)
        for i in range(self.num_seg_classes):
            seg_one_hot[i, :, :] = (seg == i).astype(np.float32)  #Pixels equal to that class ID become 1.0 and other 0.All other pixels become 0.0
            
        # Initialize line probability maps
        line_prob = np.zeros((self.num_line_classes, *img.shape[:2]), dtype=np.float32)
        
        # # Load line annotations
        # try:
        #     with open(self.line_paths[idx], "r") as f:
        #         line_data = json.load(f)
            
        #     # Process line annotations
        #     for line_idx, (key, annotations) in enumerate(line_data.items()):
        #         if line_idx >= self.num_line_classes:
        #             break
                    
        #         if isinstance(annotations, list):
        #             for annotation in annotations:
        #                 if isinstance(annotation, dict) and "points" in annotation:
        #                     points = annotation["points"]
        #                     if len(points) >= 2:
        #                         coords = (
        #                             int(points[0][0]),
        #                             int(points[0][1]),
        #                             int(points[1][0]),
        #                             int(points[1][1])
        #                         )
                                
        #                         prob_map = gaussian_line_probability(
        #                             coords, img.shape[:2], std=0.01, epsilon=1e-6
        #                         )
        #                         line_prob[line_idx] = np.maximum(line_prob[line_idx], prob_map)
        # except Exception as e:
        #     print(f"Warning: Error loading line data for {self.line_paths[idx]}: {e}")
            

        if self.augment and np.random.rand() > 0.5:
        
            img = np.fliplr(img)
            seg_one_hot = np.fliplr(seg_one_hot)
            #line_prob = np.fliplr(line_prob)
            
        # Convert to tensors - using ascontiguousarray to prevent negative stride issues
        img_tensor = self.transform(img.copy())
        seg_tensor = torch.from_numpy(np.ascontiguousarray(seg_one_hot))
        line_tensor = torch.from_numpy(np.ascontiguousarray(line_prob))
        
        return img_tensor, (seg_tensor, line_tensor)