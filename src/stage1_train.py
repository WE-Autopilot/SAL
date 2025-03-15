import numpy as np
import torch as pt
from PIL import Image
from get_overlap_distance_2 import get_overlap_distance_2
from model import Model
from placeholders import image_lidar, sal, penalty


# training loop for stage1
def train_stage1(model, optimizer, num_epochs, data_loader, a=1.0, b=1.0, c=1.0, d=1.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for image, position, theta, path, target in data_loader:
            optimizer.zero_grad()
            lidar_image = image_lidar(image, position, theta, scale=1.0)
            saliency = sal(lidar_image)
            loss = stage1_penalty(image, position, theta, path, target, a, b, c, d)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
