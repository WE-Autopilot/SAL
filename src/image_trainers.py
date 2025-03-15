import os
import numpy as np
import h5py
import yaml
from PIL import Image
from f110_gym.envs import laser_models as lidar

# Parameter: use every nth waypoint.
waypointJump = 16

def get_track_names(maps_folder):
    """
    Returns a list of available track names based on files in the maps folder.
    Expects each track to have a PNG file, a YAML file, and a CSV file.
    """
    tracks = []
    for file in os.listdir(maps_folder):
        if file.endswith(".png"):
            track_name = file.replace(".png", "")
            csv_path = os.path.join(maps_folder, f"{track_name}.csv")
            yaml_path = os.path.join(maps_folder, f"{track_name}.yaml")
            if os.path.exists(csv_path) and os.path.exists(yaml_path):
                tracks.append((track_name, csv_path, yaml_path))
    return tracks

def find_waypoint_angle(waypoints, index):
    """
    Computes the heading angle from the waypoint at the given index to the next waypoint.
    Returns a pose: [x, y, angle].
    """
    if index < 0 or index >= len(waypoints) - 1:
        print("Invalid index: no next waypoint available.")
        return None
    
    current = waypoints[index]
    next_point = waypoints[index + 1]
    dx = next_point[0] - current[0]
    dy = next_point[1] - current[1]
    angle = np.arctan2(dy, dx)
    pose = np.array([current[0], current[1], angle])
    return pose

def get_origin_from_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config.get("origin")

def simulate_lidar(png_path, yaml_path, pose, num_beams=540, fov=(2*np.pi), eps=0.0001, theta_dis=2000, max_range=30.0, resolution=0.05):
    """
    Simulates a 2D LIDAR scan on a given map (PNG file) at a specified pose.
    """
    # Load the PNG as a grayscale image.
    img = Image.open(png_path).convert('L')
    bitmap = np.array(img).astype(np.float64)
    bitmap[bitmap < 128] = 0.
    bitmap[bitmap >= 128] = 255.
    
    # Compute distance transform.
    dt = lidar.get_dt(bitmap, resolution)
    height, width = bitmap.shape
    
    # Load origin from YAML.
    origin = get_origin_from_yaml(yaml_path)
    orig_x = origin[0]
    orig_y = origin[1]
    orig_theta = origin[2]
    orig_c = np.cos(orig_theta)
    orig_s = np.sin(orig_theta)
    
    # Precompute discretized angle array.
    theta_arr = np.linspace(0.0, 2*np.pi, num=theta_dis, endpoint=False)
    sines = np.sin(theta_arr)
    cosines = np.cos(theta_arr)
    
    angle_increment = fov / (num_beams - 1)
    theta_index_increment = theta_dis * angle_increment / (2*np.pi)
    
    scan = lidar.get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps,
                          orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)
    return scan

def create_lidar_image(scan, image_height=100, max_range=30.0):
    """
    Converts a 1D LIDAR scan array into a 2D bitmap image.
    We map range values [0, max_range] linearly to [255, 0] (closer objects are brighter),
    then tile the 1D array vertically.
    """
    # Normalize: closer ranges give higher brightness.
    # Clip scan to max_range.
    scan_clipped = np.clip(scan, 0, max_range)
    # Map to [0,255] (here 0 = max range (dark), 255 = very close (bright))
    # You can invert if desired.
    lidar_line = ((max_range - scan_clipped) / max_range * 255).astype(np.uint8)
    # Tile vertically.
    lidar_image = np.tile(lidar_line, (image_height, 1))
    return lidar_image

def crop_map_image(png_path, yaml_path, pose, crop_size=200):
    """
    Crops the map image around the given pose.
    The pose is in world coordinates. The YAML provides origin and resolution.
    
    Args:
        png_path (str): Path to the PNG map.
        yaml_path (str): Path to the YAML file.
        pose (np.ndarray): [x, y, theta] of the pose in world coordinates.
        crop_size (int): Crop size in pixels (square crop).
        
    Returns:
        cropped (np.ndarray): Cropped map bitmap as a uint8 array.
    """
    # Load map metadata.
    origin = get_origin_from_yaml(yaml_path)
    orig_x = origin[0]
    orig_y = origin[1]
    resolution = None
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        resolution = config.get("resolution", 0.1)  # default resolution if missing

    # Open the map image.
    img = Image.open(png_path).convert('L')
    # (Assume the image is already oriented correctly.)
    img_np = np.array(img)
    height, width = img_np.shape

    # Convert pose world coordinates to pixel coordinates.
    # According to ROS convention, the origin in the YAML is the bottom-left corner.
    # Our map image (if not flipped) may have origin at the top-left.
    # Many times the map image is flipped vertically relative to the YAML.
    # Here we assume that the PNG is oriented with the origin at the bottom-left.
    # If not, you might need to flip the image.
    pixel_x = (pose[0] - orig_x) / resolution
    pixel_y = (pose[1] - orig_y) / resolution
    # If the image is stored with (0,0) at top-left, then:
    pixel_y = height - pixel_y

    # Define crop box centered at (pixel_x, pixel_y).
    left = int(pixel_x - crop_size/2)
    upper = int(pixel_y - crop_size/2)
    right = left + crop_size
    lower = upper + crop_size

    # Ensure the box is within the image boundaries.
    left = max(0, left)
    upper = max(0, upper)
    right = min(width, right)
    lower = min(height, lower)

    cropped = img.crop((left, upper, right, lower))
    return np.array(cropped, dtype=np.uint8)

# --- Main loop to generate dataset ---

lidar_images = []
map_crops = []

maps_folder = "../assets/maps"  # Adjust folder path as needed
tracks = get_track_names(maps_folder)
print("Found tracks:", tracks)

for track_name, csv_path, yaml_path in tracks:
    png_path = os.path.join(maps_folder, f"{track_name}.png")
    # Load waypoints from CSV.
    waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=1)
    num_waypoints = len(waypoints)
    print(f"Track: {track_name}, Num waypoints: {num_waypoints}")
    
    # For every nth waypoint
    for i in range(0, num_waypoints - 1, waypointJump):
        pose = find_waypoint_angle(waypoints, i)
        if pose is None:
            continue
        print(f"Processing {track_name} waypoint index {i}: pose {pose}")
        
        # Simulate lidar scan at this pose.
        scan = simulate_lidar(png_path, yaml_path, pose)
        # Convert the 1D scan into a 2D lidar image.
        lidar_img = create_lidar_image(scan, image_height=100, max_range=30.0)
        
        # Crop the map image around this pose.
        map_crop = crop_map_image(png_path, yaml_path, pose, crop_size=200)
        
        # Append results.
        lidar_images.append(lidar_img)
        map_crops.append(map_crop)

# Convert lists to numpy arrays.
lidar_images = np.array(lidar_images)
map_crops = np.array(map_crops)

print("Lidar images shape:", lidar_images.shape)
print("Map crops shape:", map_crops.shape)

# Store results in an HDF5 file.
import h5py as hp
with hp.File("dataset.h5", "w") as file:
    file.create_dataset("lidar", data=lidar_images, chunks=None, dtype='uint8')
    file.create_dataset("map", data=map_crops, chunks=None, dtype='uint8')

print("Dataset stored in dataset.h5")
