import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch as pt


def get_absolute_path(path, position, angle, scale):
    """
    Converts relative movements into absolute positions with scaling and rotation.
    """
    path = path.reshape(len(path), -1, 2) * scale
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    path = np.tensordot(path, rotation_matrix, axes=(2, 1))
    absolute_path = np.cumsum(np.concatenate((position[:, None, :], path), axis=1), axis=1)
    return absolute_path


def get_track_array(track_images):
    track_arrays = [np.array(track_image.convert("L")) for track_image in track_images]
    track_array = np.array(track_arrays)
    return track_array


def draw_lines_on_bitmaps(points, height, width, radius):
    batch_size = points.shape[0]
    output_bitmaps = []
    points = points.astype(np.int32)

    for i in range(batch_size):
        # Create a blank black image for the current set of points
        bitmap = np.zeros((height, width), dtype=np.uint8)
        current_points = points[i]

        # Draw lines between consecutive points
        for j in range(current_points.shape[0] - 1):
            start_point = tuple(current_points[j])
            end_point = tuple(current_points[j + 1])
            cv2.line(bitmap, start_point, end_point, color=255, thickness=radius)

        output_bitmaps.append(bitmap)

    # Concatenate all the bitmaps into a single tensor
    result_array = np.stack(output_bitmaps, axis=0)
    return result_array


def get_overlap(track_array, absolute_path, car_radius):
    """
    Computes the overlap area by masking the path on an inverted, normalized map.
    :param track_image: PIL image of the map
    :param absolute_path: Array of absolute path waypoints
    :param car_radius: Radius of the car to determine path thickness
    :return: Overlap area (sum of out-of-bounds pixels)
    """
    
    # Create a blank black image (same size as track)
    path_only_image = draw_lines_on_bitmaps(absolute_path, track_array.shape[-2], track_array.shape[-1], car_radius)
    
    # Invert and normalize the track image
    inverted_track_array = 255 - track_array
    normalized_track_array = inverted_track_array / 255.0
    
    # Convert path-only image to an array and normalize
    path_mask_array = np.array(path_only_image) / 255.0
    
    # Apply the mask: Keep only the path pixels on the inverted map
    masked_map = normalized_track_array * path_mask_array
    
    # Compute overlap area
    overlap_area = np.sum(masked_map, axis=(-1, -2))
    return overlap_area


def stage1_penalty(paths, positions, angles, scales, track_arrays, car_radius, target, a=1, b=1, c=0.1, d=0.5):
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

    # Compute absolute path waypoints
    absolute_path = get_absolute_path(path, position, angle, scale)

    # Compute the total out-of-bounds distance using the overlap function
    d_o = get_overlap(track_arrays, absolute_path, car_radius).mean()

    # Compute the Euclidean distance from the final path position to the target
    final_pos = absolute_path[:, -1]
    d_t = np.linalg.norm(final_pos - target).mean()

    # Compute the total path distance (sum of step lengths)
    step_lengths = np.linalg.norm(path, axis=-1)
    d_c = step_lengths.sum(axis=-1)

    # Compute mean second-order differences to penalize sharp turns
    if path.shape[1] < 3:
        d_p = 0.0  # Not enough points to compute second-order differences
    else:
        first_diff = np.diff(path, axis=-2)  # First-order differences (n-1 vectors)
        second_diff = np.diff(first_diff, axis=-2)  # Second-order differences (n-2 vectors)
        d_p = np.linalg.norm(second_diff, axis=-1).mean()  # Mean magnitude of second differences

    # Return the weighted penalty sum (smaller penalty is better)
    return pt.tensor(a * d_o + b * d_t + c * d_c + d * d_p, dtype=pt.float32)


if __name__ == "__main__":
    track_images = [Image.open("map.png") for _ in range(2)]

    path = np.array([[ -5,  -3],
       [ -8,  -8],
       [ -8,  -2],
       [ -4,  -2],
       [ -9,   1],
       [ -6,  -1],
       [ -8,   2],
       [ -7,  -1],
       [ -2,  -5],
       [ -3,  -5],
       [ -1,  -6],
       [  1,  -6],
       [  2,  -6],
       [ -1,  -6],
       [ -1,  -6],
       [  1, -12]])
    path = np.tile(path[None, ...], (2, 1, 1))

    position = np.array([[452, 224], [452, 224]])

    angle = 0

    scale = 1

    car_radius = 3

    track_array = get_track_array(track_images)
    absolute_path = get_absolute_path(path, position, angle, scale)
    target = absolute_path[:, -1] + 1
    overlap = get_overlap(track_array, absolute_path, car_radius)

    loss = stage1_penalty(path, position, angle, scale, track_array, car_radius, target, 1, 1, 1, 1)
