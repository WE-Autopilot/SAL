import os
import numpy as np
import torch as pt
class WaypointManager:
    def __init__(self, sal, conf_dict):
        # The conf object holds parameters like the file path, delimiter, etc.
        self.conf_dict = conf_dict
        self.sal = sal
        self.waypoints = []

        self.distance_to_wp = 0

    def load_next_waypoints(self, current_car_x, current_car_y, current_car_heading, lidar, current_vel):
        # Save the current car pose as attributes for later reference.
        self.current_car_x = current_car_x
        self.current_car_y = current_car_y
        self.current_car_heading = current_car_heading

        # print(lidar[None, None, ...].shape, current_vel[None, ...].shape)

        dist, value = self.sal(lidar, current_vel)

        path = dist.sample()
        # print(path)
        log_probs = dist.log_prob(path)
        # Convert the SAL-generated waypoints to global coordinates using the set_path method.
        self.set_path(path.numpy()[0], current_car_x, current_car_y, current_car_heading)

        # Calculate distance to the next waypoint by taking the global coordinate of the next wp and subtracting the current car pos from it.
        self.distance_to_wp = np.linalg.norm(self.waypoints[0, :] - np.array([current_car_x, current_car_y]))

        # print(f"Loaded new SAL-generated waypoints at pose: ({current_car_x}, {current_car_y}, {current_car_heading})")
        return path, log_probs, value, self.distance_to_wp
    
    def is_near_last_waypoint(self, position, threshold=1.0):
        """
        Checks if the vehicle is near the 8th waypoint in the current window.
        This is used to trigger a window shift.
        """
        if self.waypoints.shape[0] < 8:
            return False
        midpoint = self.waypoints[7, :]  # 8th waypoint (0-indexed).
        return np.linalg.norm(position - midpoint) < threshold
    
    def set_path(self, wpts_vector, car_x, car_y, rotation):
        """
        Accepts a flat numpy array of relative x, y coordinates (in pixel units, tip-to-tail),
        converts them to global coordinates in meters, and updates self.waypoints.
        
        Parameters:
          - wpts_vector: Flat array of relative waypoints in pixel coordinates.
          - car_x, car_y: The current global position of the car (in meters).
          - rotation: The car's heading (in radians) used to rotate the relative waypoints.
        """
        # Reshape the flat array to (N, 2)
        rel_waypoints = wpts_vector.reshape((-1, 2))
        # Convert to global waypoints using tip-to-tail transformation:
        # 1. Scale pixel values to meters.
        # 2. Cumulatively sum the relative displacements.
        # 3. Rotate by the car's heading.
        # 4. Translate by the car's global position.
        global_waypoints = self.convert_to_global_waypoints(rel_waypoints, car_x, car_y, rotation, self.conf_dict["resolution"])
        # print("global coordinates", global_waypoints)
        # Use only the first 16 waypoints.
        self.waypoints = global_waypoints[:16]

    @staticmethod
    def convert_to_global_waypoints(rel_waypoints, car_x, car_y, rotation, resolution):
        """
        Converts relative waypoints into global coordinates.
        
        Parameters:
            - rel_waypoints: np.ndarray of shape (N,2), relative waypoints.
            - car_x, car_y: Current global coordinates of the car.
            - resolution: Scaling factor (conversion from pixels to meters).
            - rotation: Rotation angle (in radians) of the car.
        
        Returns:
            - global_waypoints: np.ndarray of shape (N,2).
        """
        if rel_waypoints.ndim != 2 or rel_waypoints.shape[1] != 2:
            raise ValueError("rel_waypoints must have shape (N,2)")
        # Convert from pixels to meters.
        scaled_waypoints = rel_waypoints * resolution
        # Compute the cumulative sum (tip-to-tail) to get local coordinates.
        cumsum_waypoints = np.cumsum(scaled_waypoints, axis=0)
        # Create a rotation matrix to rotate the local waypoints into the global frame.
        cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_waypoints = cumsum_waypoints @ rotation_matrix.T
        # Translate by the car's global position.
        return rotated_waypoints + np.array([car_x, car_y])
