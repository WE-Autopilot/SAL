import numpy as np
import torch as pt
from PIL import Image
from get_overlap_distance_2 import get_overlap_distance_2
from model import Model
from placeholders import image_lidar, sal, penalty


def stage1_penalty(image, position, theta, path, target, a, b, c, d):
    """
    Computes the Stage 1 penalty function based on:
      - d_o: Total out-of-bounds distance (penalizes collisions).
      - d_t: Distance from the end of the path to the target (goal proximity).
      - d_c: Total path length (penalizes excessive travel).
      - d_p: Mean second-order differences (penalizes sharp turns).

    Returns the weighted sum:
        penalty = a*d_o + b*d_t + c*d_c + d*d_p

    Args:
        image (PIL.Image): Map where black pixels represent obstacles.
        position (np.array): Initial (x, y) position of the car.
        theta (float): Car heading in radians.
        path (np.array): List of (dx, dy) steps defining movement.
        target (np.array): Target location (x, y).
        a, b, c, d (float): Weights for each penalty component.

    Returns:
        float: The total penalty score.
    """

    # Compute the total out-of-bounds distance using the overlap function
    d_o = get_overlap_distance_2(image, position, theta, path)

    # Compute the Euclidean distance from the final path position to the target
    final_pos = position + path.sum(axis=0)
    d_t = np.linalg.norm(final_pos - target)

    # Compute the total path distance (sum of step lengths)
    step_lengths = np.linalg.norm(path, axis=1)
    d_c = step_lengths.sum()

    # Compute mean second-order differences to penalize sharp turns
    if len(path) < 3:
        d_p = 0.0  # Not enough points to compute second-order differences
    else:
        first_diff = path[1:] - path[:-1]  # First-order differences (n-1 vectors)
        second_diff = first_diff[1:] - first_diff[:-1]  # Second-order differences (n-2 vectors)
        d_p = np.mean(np.linalg.norm(second_diff, axis=1))  # Mean magnitude of second differences

    # Return the weighted penalty sum (smaller penalty is better)
    return pt.tensor(a * d_o + b * d_t + c * d_c + d * d_p, dtype=pt.float32)


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