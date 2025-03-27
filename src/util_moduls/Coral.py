import torch
import numpy as np

import torch.nn.functional as F

def coral(source, target):

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)
    loss = F.mse_loss(source_c, target_c) / 4

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






