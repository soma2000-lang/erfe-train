import cv2

import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from config import NUM_SEG_CLASSES, NUM_LINE_CLASSES, DEVICE

# Mathematical Formulation

# The weighted line regression minimizes:
# Error=∑wi[xisin⁡(θ)−yicos⁡(θ)+C]2
# Error=∑wi​[xi​sin(θ)−yi​cos(θ)+C]2

# The closed-form solutions derived from minimizing this weighted squared error are solved through the quadratic equation:
# αtan⁡2(θ)+βtan⁡(θ)+γ=0
# αtan2(θ)+βtan(θ)+γ=0

# where:

#     α=(sum_w⋅sum_wxy−sum_wx⋅sum_wy)α=(sum_w⋅sum_wxy−sum_wx⋅sum_wy)
#     β=(sum_w⋅sum_wx2_y2−sum_wy2−sum_wx2)β=(sum_w⋅sum_wx2_y2−sum_wy2−sum_wx2)
#     γ=(sum_wx⋅sum_wy−sum_w⋅sum_wxy)γ=(sum_wx⋅sum_wy−sum_w⋅sum_wxy)

# Finally, the offset CC is found as:
# C=∑wi(yicos⁡θ−xisin⁡θ)∑wi
# C=∑wi​∑wi​(yi​cosθ−xi​sinθ)​

def gaussian_line_probability(line_coords, shape, std=0.01, epsilon=1e-6, line_length=None):
    """
    Generate Gaussian probability map for a line.
    
    Args:
        line_coords: tuple of (x1, y1, x2, y2) line coordinates
        shape: tuple of (height, width) for the output map
        std: standard deviation for the Gaussian distribution
        epsilon: small constant to prevent division by zero
        line_length: length of the line to scale std
        
    Returns:
        Probability map with Gaussian distribution along the line
    """
    x1, y1, x2, y2 = line_coords
    height, width = shape
    
    # we are first creating coordinate grid
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    if line_length is None:
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # as said in paper
    adjusted_std = std * line_length + epsilon
    # Distance = |Ax + By + C| / sqrt(A^2 + B^2)
    # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    
    # Calculate distance
    distance = np.abs(A * x_grid + B * y_grid + C) / np.sqrt(A**2 + B**2 + epsilon)
    # Applying the Gaussian function
    probability = np.exp(-(distance**2) / (2 * adjusted_std**2))
    return probability


def weighted_line_regression(probability_map):
    """
    Fit a line to the probability map using weighted regression.
    Implements the algorithm from the paper.
    
    Args:
        probability_map: 2D array with probability values
        
    Returns:
        Tuple of (theta, C) representing the line parameters
    """
    # Extracts the dimensions (height and width) of the input 2D probability map array.
    height, width = probability_map.shape


    #Generates two coordinate grids (x_grid and y_grid) representing the x and y coordinates of every pixel in the probability map.
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    # we are falttening  2D coordinate grids and the probability map into 1D arrays (x, y, and w)
    x = x_grid.flatten()
    y = y_grid.flatten()
    w = probability_map.flatten()
    
    # sum_w=∑wi​
    # sum_wx=∑wixisum_wx=∑wi​xi​
    # sum_wy=∑wiyisum_wy=∑wi​yi​
    # sum_wxy=∑wixiyisum_wxy=∑wi​xi​yi​
    # sum_wx2_y2=∑wi(xi2−yi2)sum_wx2_y2=∑wi​(xi2​−yi2​)
    # to calcualte the weighted summations
    sum_w = np.sum(w)
    sum_wx = np.sum(w * x)
    sum_wy = np.sum(w * y)
    sum_wxy = np.sum(w * x * y)
    sum_wx2_y2 = np.sum(w * (x**2 - y**2))
    
    # Calculate coefficients for the quadratic equation
    #αtan2(θ)+βtan(θ)+γ=0 - this equatio as stated in the paper
    alpha = (sum_w * sum_wxy) - (sum_wx * sum_wy)
    beta = (sum_w * sum_wx2_y2) - (sum_wy**2) - (sum_wx**2)
    gamma = (sum_wx * sum_wy) - (sum_w * sum_wxy)
    
    # Solving  quadratic equation to find tan(theta)
    discriminant = beta**2 - 4 * alpha * gamma
    
    if discriminant < 0:
  
        theta = np.pi / 2
    else:

        tan_theta1 = (-beta + np.sqrt(discriminant)) / (2 * alpha)
        tan_theta2 = (-beta - np.sqrt(discriminant)) / (2 * alpha)
        
      
        theta1 = np.arctan(tan_theta1)
        theta2 = np.arctan(tan_theta2)
        
        # Calculate C for both solutions
        C1 = ((sum_wy * np.cos(theta1)) - (sum_wx * np.sin(theta1))) / sum_w
        C2 = ((sum_wy * np.cos(theta2)) - (sum_wx * np.sin(theta2))) / sum_w
        

        error1 = np.sum(w * (x * np.sin(theta1) - y * np.cos(theta1) + C1)**2)
        error2 = np.sum(w * (x * np.sin(theta2) - y * np.cos(theta2) + C2)**2)
        
        # Choose the solution with minimum error
        if error1 <= error2:
            theta = theta1
        else:
            theta = theta2
    
    #C=∑wi(yicos⁡θ−xisin⁡θ)∑wi
    #C=∑wi​∑wi​(yi​cosθ−xi​sinθ)​
    C = ((sum_wy * np.cos(theta)) - (sum_wx * np.sin(theta))) / sum_w
    
    return theta, C

# Benchmark Error=N1​i=1∑N​[xgt,i​sin(θpred​)−ygt,i​cos(θpred​)+Cpred​]2

def benchmark_error(gt_line, pred_line, shape, num_samples=100):
    """
    Calculate the benchmark error between ground truth and predicted lines.
    
    Args:
        gt_line: tuple of (theta_gt, C_gt) for ground truth line
        pred_line: tuple of (theta_pred, C_pred) for predicted line
        shape: tuple of (height, width) for the image shape
        num_samples: number of points to sample on the ground truth line
        
    Returns:
        Benchmark error value
    """
    theta_gt, C_gt = gt_line
    theta_pred, C_pred = pred_line
    height, width = shape
    
    # Parametric form of ground truth line
    # x_gt = a1*t + b1, y_gt = a2*t + b2
    sin_gt = np.sin(theta_gt)
    cos_gt = np.cos(theta_gt)
    
    # Finding line endpoints
    if abs(sin_gt) > abs(cos_gt):
        # More vertical line
        t_values = np.linspace(0, height-1, num_samples)
        a1 = -C_gt / sin_gt if sin_gt != 0 else 0
        b1 = 0
        a2 = 1
        b2 = 0
    else:
        # More horizontal line
        t_values = np.linspace(0, width-1, num_samples)
        a1 = 1
        b1 = 0
        a2 = C_gt / cos_gt if cos_gt != 0 else 0
        b2 = 0
    
    # Parametric form of predicted line
    sin_pred = np.sin(theta_pred)
    cos_pred = np.cos(theta_pred)
    
    # Calculating squared error for each point on the ground truth line
    squared_error = 0
    for t in t_values:
        # Ground truth point
        x_gt = a1 * t + b1
        y_gt = a2 * t + b2
        
        # Finding corresponding point on predicted line
        # Perpendicular distance equation from the paper
        if abs(sin_pred * cos_gt - cos_pred * sin_gt) < 1e-6:
            # Lines are parallel
            dx = x_gt * sin_pred + C_pred if sin_pred != 0 else 0
            dy = y_gt * cos_pred + C_pred if cos_pred != 0 else 0
        else:
            # Calculating corresponding parameter for predicted line
            t_pred = ((x_gt - b1) * a1 + (y_gt - b2) * a2) / (a1**2 + a2**2)
            
            # Predicted point
            x_pred = a1 * t_pred + b1
            y_pred = a2 * t_pred + b2
            
            # Distance
            dx = x_gt - x_pred
            dy = y_gt - y_pred
        
     
        squared_error += dx**2 + dy**2
    return squared_error / num_samples
