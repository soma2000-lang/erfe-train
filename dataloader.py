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
from config import NUM_SEG_CLASSES,NUM_LINE_CLASSES

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # as has ben mentioned in paper
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#To get label IDs as single channel
        seg = cv2.imread(self.seg_paths[idx], cv2.IMREAD_GRAYSCALE)
    #Converts the segmentation mask (seg) into a one-hot encoded format. Each class in the segmentation mask is 
    # represented as a separate binary map, where each map highlights the pixels belonging to that particular class.

    # basically to have three separate masks in seg_one_hot.
        seg_one_hot = np.zeros((self.num_seg_classes, *seg.shape), dtype=np.float32)
        for i in range(self.num_seg_classes):
            seg_one_hot[i, :, :] = (seg == i).astype(np.float32)
        
        # Loading the  line coordinates
        line_prob = np.zeros((self.num_line_classes, *img.shape[:2]), dtype=np.float32)
        # with open(self.line_paths[idx], 'r') as f:
        #     for line in f:
        #     #line_data = f.readlines()
        #         line_data = json.loads(line)
        #         print(line_data)
        with open(self.line_paths[idx], "r") as f:
            line_data = json.load(f)

        for annotations in line_data.items():
            for annotation in annotations:
                # Check the type of annotation
                print(f"Annotation type: {type(annotation)}")
                
                # If annotation is a dictionary
                if isinstance(annotation, dict) and "points" in annotation:
                    coords = (
                        int(annotation["points"][0][0]),
                        int(annotation["points"][0][1]),
                        int(annotation["points"][1][0]),
                        int(annotation["points"][1][1])
                    )
                    print(f"Points: {coords}")
   
                    if len(coords) == 4:  #"points": [
        #   [
        #     287.5,
        #     162.7
        #   ],
        #   [
        #     300,
        #     165
        #   ]
        # ],
        
                        prob_map = gaussian_line_probability(
                            coords, img.shape[:2], std=0.01, epsilon=1e-6
                        )
                #line_prob[line_idx, :, :] = prob_map

        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            img = np.fliplr(img)
            seg_one_hot = np.fliplr(seg_one_hot)
            line_prob = np.fliplr(line_prob)
        
        # Convert to tensors
        img_copy = img.copy()
        img_tensor = self.transform(img_copy)
        seg_one_hot=seg_one_hot.copy()
        seg_tensor = torch.from_numpy(seg_one_hot)
        line_tensor=line_prob.copy()
        line_tensor = torch.from_numpy(line_prob)
        
        return img_tensor, (seg_tensor, line_tensor)