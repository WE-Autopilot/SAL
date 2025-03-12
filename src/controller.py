import os
import yaml
from argparse import Namespace
from weap_util.abstract_controller import AbstractController

import numpy as np

from pilot import PurePursuitPlanner
from waypoint_manager import WaypointManager

class Controller(AbstractController):
    def startup(self):
        
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(ROOT_DIR, "assets", "config.yaml")
        with open(config_path) as file:
            conf_dict = yaml.safe_load(file)
        conf = Namespace(**conf_dict)

        self.planner = PurePursuitPlanner(wheelbase=(0.17145 + 0.15875))
        self.waypoint_manager = WaypointManager(conf)

    def compute(self, obs):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """
        # Extract the current car pose from the observation
        current_x = obs['poses_x'][0]
        current_y = obs['poses_y'][0]
        current_theta = obs['poses_theta'][0]
        current_position = np.array([current_x, current_y])
        
        #! SAL Change:
        #! If no waypoints have been loaded yet (e.g. initial call) or if near the end of the current batch,
        #! call load_next_waypoints with the current pose (position and heading) so that SAL can generate new waypoints.
        #! This ensures that even the very first time compute_callback is invoked, the waypoint manager gets updated.
        if (self.waypoint_manager.waypoints is None or len(self.waypoint_manager.waypoints) == 0) \
        or self.waypoint_manager.is_near_last_waypoint(current_position):
            self.waypoint_manager.load_next_waypoints(current_x, current_y, current_theta)


        # Check if the vehicle is near the end of the current waypoint batch.
        # if waypoint_manager.is_near_last_waypoint(current_position):
        #     waypoint_manager.load_next_waypoints()
        
        # Compute control commands using the current waypoints.
        speed, steer = self.planner.plan(current_x, current_y, current_theta, self.waypoint_manager.waypoints)

        
        # Return both the computed commands and the current waypoints for rendering.
        return speed, steer, self.waypoint_manager.waypoints

    def shutdown():
        pass