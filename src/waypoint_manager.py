import os
import numpy as np

# Define ROOT_DIR so that file paths are relative to the project root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class WaypointManager:
    def __init__(self, conf):
        # The conf object holds parameters like the file path, delimiter, etc.
        self.conf = conf
        self.load_waypoints()   #! SAL Change: Delete this call initialization of waypoints with SAL is done in compute_callback()

    def load_waypoints(self):   #! SAL Change: Delete this.
        """Loads waypoints from the CSV file and initializes the current window."""
        waypoint_file = os.path.join(ROOT_DIR, "assets", os.path.basename(self.conf.wpt_path))
        if not os.path.exists(waypoint_file):
            raise FileNotFoundError(f"Waypoint file not found: {waypoint_file}")
        raw_waypoints = np.loadtxt(
            waypoint_file,
            delimiter=self.conf.wpt_delim,
            skiprows=self.conf.wpt_rowskip,
        )[::2]
        if raw_waypoints.shape[1] < max(self.conf.wpt_xind, self.conf.wpt_yind):
            raise ValueError(f"Waypoint file has {raw_waypoints.shape[1]} columns, but expected at least {max(self.conf.wpt_xind, self.conf.wpt_yind)}.")
        # Extract x, y coordinates.
        self.original_waypoints = raw_waypoints
        self.waypoints = raw_waypoints[:16]
        self.waypoint_index = 0

    #! SAL Change: New version of load_next_waypoints() that uses the SAL instead.
    # def load_next_waypoints(self, current_car_x, current_car_y, current_car_heading):
    #     # Save the current car pose as attributes for later reference.
    #     self.current_car_x = current_car_x
    #     self.current_car_y = current_car_y
    #     self.current_car_heading = current_car_heading

    #     self.sal_wpts = SAL.generate_waypoints()  # Example call to the SAL interface.

    #     # Convert the SAL-generated waypoints to global coordinates using the set_path method.
    #     self.set_path(self.sal_wpts, current_car_x, current_car_y, current_car_heading, 0.0625)

    #     print(f"Loaded new SAL-generated waypoints at pose: ({current_car_x}, {current_car_y}, {current_car_heading})")

    def load_next_waypoints(self):
        """
        Shifts the waypoint window by 8 to maintain a window of 16 waypoints.
        If the window exceeds the total, it wraps around.
        """
        self.waypoint_index += 8  # Shift by 8.
        start_idx = self.waypoint_index
        end_idx = start_idx + 16
        if end_idx > len(self.original_waypoints):
            extra = end_idx - len(self.original_waypoints)
            part1 = self.original_waypoints[start_idx:]
            part2 = self.original_waypoints[:extra]
            self.waypoints = np.concatenate((part1, part2), axis=0)
        else:
            self.waypoints = self.original_waypoints[start_idx:end_idx, :2]
        print(f"Loaded waypoints {start_idx} to {end_idx} (wrapped if necessary).")

    def is_near_last_waypoint(self, position, threshold=1.0):
        """
        Checks if the vehicle is near the 8th waypoint in the current window.
        This is used to trigger a window shift.
        """
        if self.waypoints.shape[0] < 8:
            return False
        midpoint = self.waypoints[7, :]  # 8th waypoint (0-indexed).
        return np.linalg.norm(position - midpoint) < threshold
    
    def set_path(self, wpts_vector, car_x, car_y, rotation, scale):
        """
        Accepts a flat numpy array of relative x, y coordinates (in pixel units, tip-to-tail),
        converts them to global coordinates in meters, and updates self.waypoints.
        
        Parameters:
          - wpts_vector: Flat array of relative waypoints in pixel coordinates.
          - car_x, car_y: The current global position of the car (in meters).
          - scale: Conversion factor from pixels to meters (e.g., 0.0625 m/pixel).
          - rotation: The car's heading (in radians) used to rotate the relative waypoints.
        """
        # Reshape the flat array to (N, 2)
        rel_waypoints = wpts_vector.reshape((-1, 2))
        # Convert to global waypoints using tip-to-tail transformation:
        # 1. Scale pixel values to meters.
        # 2. Cumulatively sum the relative displacements.
        # 3. Rotate by the car's heading.
        # 4. Translate by the car's global position.
        global_waypoints = self.convert_to_global_waypoints(rel_waypoints, car_x, car_y, rotation, scale)
        # Use only the first 16 waypoints.
        self.waypoints = global_waypoints[:16]

    @staticmethod
    def convert_to_global_waypoints(rel_waypoints, car_x, car_y, rotation, scale):
        """
        Converts relative waypoints into global coordinates.
        
        Parameters:
            - rel_waypoints: np.ndarray of shape (N,2), relative waypoints.
            - car_x, car_y: Current global coordinates of the car.
            - scale: Scaling factor (conversion from pixels to meters).
            - rotation: Rotation angle (in radians) of the car.
        
        Returns:
            - global_waypoints: np.ndarray of shape (N,2).
        """
        if rel_waypoints.ndim != 2 or rel_waypoints.shape[1] != 2:
            raise ValueError("rel_waypoints must have shape (N,2)")
        # Convert from pixels to meters.
        scaled_waypoints = rel_waypoints * scale
        # Compute the cumulative sum (tip-to-tail) to get local coordinates.
        cumsum_waypoints = np.cumsum(scaled_waypoints, axis=0)
        # Create a rotation matrix to rotate the local waypoints into the global frame.
        cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_waypoints = cumsum_waypoints @ rotation_matrix.T
        # Translate by the car's global position.
        return rotated_waypoints + np.array([car_x, car_y])
