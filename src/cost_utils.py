import numpy as np
from PIL import Image, ImageDraw


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


def get_overlap(track_array, absolute_path, car_radius):
    """
    Computes the overlap area by masking the path on an inverted, normalized map.
    :param track_image: PIL image of the map
    :param absolute_path: Array of absolute path waypoints
    :param car_radius: Radius of the car to determine path thickness
    :return: Overlap area (sum of out-of-bounds pixels)
    """
    
    # Create a blank black image (same size as track)
    path_only_image = Image.new("L", (track_array.shape[1], track_array.shape[0]), 0)
    draw = ImageDraw.Draw(path_only_image)
    
    # Draw the path in white (255) with a thickness based on car_radius
    for i in range(len(absolute_path) - 1):
        x1, y1 = absolute_path[i]
        x2, y2 = absolute_path[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=255, width=car_radius * 2)
    
    # Invert and normalize the track image
    inverted_track_array = 255 - track_array
    normalized_track_array = inverted_track_array / 255.0
    
    # Convert path-only image to an array and normalize
    path_mask_array = np.array(path_only_image) / 255.0
    
    # Apply the mask: Keep only the path pixels on the inverted map
    masked_map = normalized_track_array * path_mask_array
    
    # Ensure the masked map is correctly normalized
    assert masked_map.min() >= 0 and masked_map.max() <= 1, "Masked map is not normalized!"
    
    # Compute overlap area
    overlap_area = np.sum(masked_map)
    return overlap_area


def stage1_penalty(path, position, angle, scale, track_image, car_radius, target, a, b, c, d):
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
    d_o = get_overlap(track_image, absolute_path, car_radius)

    # Compute the Euclidean distance from the final path position to the target
    final_pos = absolute_path[-1]
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


