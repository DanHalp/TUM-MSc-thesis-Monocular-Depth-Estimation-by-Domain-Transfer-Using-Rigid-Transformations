import torch
import numpy as np

import torch.nn.functional as F

def coral(source, target):
    B, C, D = source.size()  # batch_size, channels, dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    # loss = torch.norm(source_c - target_c, p='fro', dim=(-2, -1)) ** 2
    loss = F.mse_loss(source_c, target_c) / 4

    # loss = loss / (4 * D * D)
    return loss

def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data for each batch
    """
    B, C, D = input_data.size()  # batch_size, channels, dim vector

    # Check if using gpu or cpu
    device = input_data.device

    id_row = torch.ones(C).resize(1, C).to(device=device)
    sum_column = torch.matmul(id_row, input_data)
    mean_column = sum_column / C
    term_mul_2 = torch.matmul(mean_column.transpose(1, 2), mean_column)
    d_t_d = torch.matmul(input_data.transpose(1, 2), input_data)
    c = (d_t_d - term_mul_2) / (C - 1)

    return c


def riemannian_distance_gaussians(point_cloud1, point_cloud2, epsilon=1e-5):
    """
    Calculate the Riemannian distance between two batches of point clouds treated as Gaussians.
    
    point_cloud1, point_cloud2: Tensors of shape (B, N, D) where B is batch size,
                                N is the number of points, and D is the dimensionality.
    epsilon: Regularization parameter for the covariance matrix.
    
    Returns: Riemannian distance between the two distributions, of shape (B,).
    """
    
    def calculate_mean(X):
        """Calculate the mean vector for a batch of data points."""
        return X.mean(dim=-2)
    
    def calculate_regularized_covariance(X, epsilon):
        """Calculate the regularized covariance matrix for a batch of data points."""
        X_mean = X.mean(dim=-2, keepdim=True)
        X_centered = X - X_mean
        
        # Compute covariance matrix
        cov_matrix = torch.matmul(X_centered.transpose(-2, -1), X_centered) / (X.size(-2) - 1)
        
        # Add epsilon * Identity matrix for regularization
        identity = torch.eye(X.size(-1)).to(X.device)  # Create identity matrix of shape (D, D)
        regularized_cov_matrix = cov_matrix + epsilon * identity
        
        return regularized_cov_matrix
    
    # point_cloud1 = F.normalize(point_cloud1, p=2, dim=-1)
    # point_cloud2 = F.normalize(point_cloud2, p=2, dim=-1)
    

    # mean_diff = torch.norm(point_cloud1.mean(dim=-2) - point_cloud2.mean(dim=-2), dim=-1).mean()
    mean_diff = torch.tensor(0.0).to(point_cloud1.device)
    
    # Covariance difference (Frobenius norm of the difference between covariance matrices)
    sigma_diff = coral(point_cloud1, point_cloud2)
    
    # Riemannian distance is the sum of the mean difference and covariance difference
    distance = mean_diff + sigma_diff
    
    return distance






