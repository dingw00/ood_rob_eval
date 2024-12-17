import torch 
import math
import torch.nn.functional as F
import os

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return torch.FloatTensor(size=size).uniform_(**args)

def theta2affine(theta):
    bs = theta.size()[0]
    rotation = math.pi * theta[:, 0]
    tx, ty = theta[:, 1], theta[:, 2]
    cx, cy = theta[:, 3], theta[:, 4]
    rotation_matrix = torch.stack([
        torch.stack([torch.cos(rotation), -torch.sin(rotation), torch.zeros_like(rotation)], dim=1),
        torch.stack([torch.sin(rotation), torch.cos(rotation), torch.zeros_like(rotation)], dim=1),
        torch.stack([torch.zeros_like(rotation), torch.zeros_like(rotation), torch.ones_like(rotation)], dim=1)
    ], dim=2)
    scaling_matrix = torch.stack([
        torch.stack([cx, torch.zeros_like(rotation), torch.zeros_like(rotation)], dim=1),
        torch.stack([torch.zeros_like(rotation), cy, torch.zeros_like(rotation)], dim=1),
        torch.stack([torch.zeros_like(rotation), torch.zeros_like(rotation), torch.ones_like(rotation)], dim=1)
    ], dim=2)
    translation_matrix = torch.stack([
        torch.stack([torch.ones_like(rotation), torch.zeros_like(rotation), torch.zeros_like(rotation)], dim=1),
        torch.stack([torch.zeros_like(rotation), torch.ones_like(rotation), torch.zeros_like(rotation)], dim=1),
        torch.stack([tx, ty, torch.ones_like(rotation)], dim=1)
    ], dim=2)
    affine_matrix = torch.bmm(torch.bmm(rotation_matrix, scaling_matrix), translation_matrix)
    return affine_matrix[:,:2,:]

def transform(x, rots, txs, scales):
    assert x.shape[2] == x.shape[3]

    # print(x.shape[2], x.shape[3])
    with torch.no_grad():
        rots = rots / 180 
        rots = rots.unsqueeze(dim=1)
        theta = torch.cat((rots, txs, scales), axis = 1)
        affine = theta2affine(theta)
        grid = F.affine_grid(affine,
               x.size(),align_corners=True)
        new_image = F.grid_sample(x, grid, align_corners=True)
        return new_image