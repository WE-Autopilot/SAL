import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##########################
# 1. LOCAL PIXEL DISPLACEMENTS
##########################
local_waypoints_pixels = np.array([
    [10,  0],
    [10,  2],
    [10,  4],
    [10,  6],
    [10,  8],
    [10, 10],
    [10, 12],
    [10, 14],
    [10, 16],
    [10, 18],
    [10, 20],
    [10, 22],
    [10, 24],
    [10, 26],
    [10, 28],
    [10, 30],
], dtype=float)

##########################
# 2. GLOBAL CONVERSION UTILITY
##########################
def convert_local_pixels_to_global(rel_waypoints_px, car_x, car_y, heading_rad, scale_m_per_px):
    """
    Converts local pixel displacements to global coordinates in meters:
      1) Scale pixel -> meters
      2) Cumulative sum (tip-to-tail)
      3) Rotate by heading_rad
      4) Translate by (car_x, car_y)
    """
    # Scale
    scaled = rel_waypoints_px * scale_m_per_px
    # Cumulative sum
    cumsum = np.cumsum(scaled, axis=0)
    # Rotate
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    R = np.array([[cos_h, -sin_h],
                  [sin_h,  cos_h]])
    rotated = cumsum @ R.T
    # Translate
    global_waypoints = rotated + np.array([car_x, car_y])
    return global_waypoints

##########################
# 3. MAIN SCRIPT
##########################
def main():
    # --- Car's pose from config.yaml ---
    car_x = 0.7
    car_y = 0.0
    car_heading = 1.37079632679  # radians
    scale = 0.0625               # m/px

    # --- Map info from map.yaml ---
    resolution = 0.0625
    origin_x, origin_y = -78.21853769831466, -44.37590462453829

    # Load the map image
    map_path = os.path.join("..", "assets", "example_map.png")  # Adjust path if needed
    img = mpimg.imread(map_path)
    if img.ndim == 2:
        H, W = img.shape
    else:
        H, W, _ = img.shape

    # ----------------------------------------------------------------
    # A) Compute global waypoints from local pixel displacements
    # ----------------------------------------------------------------
    global_waypoints = convert_local_pixels_to_global(
        local_waypoints_pixels,
        car_x, car_y,
        car_heading,
        scale
    )

    # ----------------------------------------------------------------
    # B) Compute local waypoints in "pixel space" with the same orientation
    # ----------------------------------------------------------------
    # 1) Cumulative sum of the raw pixel displacements
    cumsum_px = np.cumsum(local_waypoints_pixels, axis=0)

    # 2) Rotate them by the same heading (so orientation matches the global plot)
    cos_h, sin_h = np.cos(car_heading), np.sin(car_heading)
    R_px = np.array([[cos_h, -sin_h],
                     [sin_h,  cos_h]])
    rotated_px = cumsum_px @ R_px.T

    # 3) Translate the car so it lines up with its actual position on the map
    #    in pixel coordinates:  car_px = ((car_x - origin_x)/resolution, (car_y - origin_y)/resolution)
    car_px = np.array([
        (car_x - origin_x) / resolution,
        (car_y - origin_y) / resolution
    ])
    local_waypoints_px = rotated_px + car_px

    # ----------------------------------------------------------------
    # C) Define subplots with "origin='lower'" for both
    # ----------------------------------------------------------------
    # Left subplot: "Pixel coordinates"
    # We'll define the extent in raw pixel size: [0, W, 0, H].
    # Right subplot: "Global coordinates"
    # We'll define the extent in meters: [origin_x, origin_x+W*res, origin_y, origin_y+H*res].
    extent_pixels = [0, W, 0, H]  # left, right, bottom, top
    extent_global = [
        origin_x,
        origin_x + W * resolution,
        origin_y,
        origin_y + H * resolution
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --------------------
    # Subplot 1: Pixel Frame
    # --------------------
    axes[0].imshow(img, extent=extent_pixels, origin='lower')
    axes[0].scatter(car_px[0], car_px[1], c='magenta', s=60, marker='s', label='Car (pixel coords)')
    axes[0].scatter(local_waypoints_px[:,0], local_waypoints_px[:,1], c='red', label='Local Waypoints (px)')

    for i, pt in enumerate(local_waypoints_px):
        axes[0].annotate(str(i), (pt[0]+2, pt[1]+2))

    axes[0].set_title("Local Frame (Pixel) w/ Same Orientation")
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True)

    # --------------------
    # Subplot 2: Global Frame
    # --------------------
    axes[1].imshow(img, extent=extent_global, origin='lower')
    axes[1].scatter(car_x, car_y, c='magenta', s=60, marker='s', label='Car (global coords)')
    axes[1].scatter(global_waypoints[:,0], global_waypoints[:,1], c='red', label='Global Waypoints (m)')

    for i, pt in enumerate(global_waypoints):
        axes[1].annotate(str(i), (pt[0]+0.05, pt[1]+0.05))

    axes[1].set_title("Global Frame (Meters) w/ Same Orientation")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].axis('equal')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
