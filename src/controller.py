
from weap_util.abstract_controller import AbstractController

import numpy as np

from torch.optim import AdamW
from sal import SAL
from pilot import PurePursuitPlanner
from waypoint_manager import WaypointManager
from weap_util.lidar import lidar_to_bitmap

class Controller(AbstractController):
    def setConf(self, conf_dict):
        self.conf_dict = conf_dict
        
    def startup(self):
        self.planner = PurePursuitPlanner(wheelbase=(0.17145 + 0.15875))
        self.sal = SAL()
        self.waypoint_manager = WaypointManager(self.sal, self.conf_dict)
        self.optimizer = AdamW(self.sal.parameters(), lr=1e-4)

        # Apped when we call SAL
        self.lidar_scans = []
        self.velocities = []
        self.log_probs = []
        self.paths = []
        self.value = 0
        self.costs = []
        self.advantages = []

        # Apped when we step in the environment
        self.deviations = []
        self.speeds = []

        self.distance_to_wp = 0

        self.a, self.b, self.c, self.d = 1, 1, 1, 1

    def compute(self, obs):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """
        # Extract the current car pose from the observation
        current_x = obs['poses_x'][0]
        current_y = obs['poses_y'][0]
        current_theta = obs['poses_theta'][0]

        current_velx = obs['linear_vels_x'][0]
        current_vely = obs['linear_vels_y'][0]
        current_ang_vel = obs['ang_vels_z'][0]

        current_vel = np.array([np.linalg.norm([current_velx, current_vely]), current_ang_vel])

        current_position = np.array([current_x, current_y])

        crashed = False

        if (self.waypoint_manager.waypoints is None or len(self.waypoint_manager.waypoints) == 0) \
        or self.waypoint_manager.is_near_last_waypoint(current_position) or crashed:
            
            if self.waypoint_manager.waypoints is not None:
                
                if len(self.deviations) == 0:
                    self.deviations.append(0)
                if len(self.speeds) == 0:
                    self.speeds.append(0)

                self.costs.append(self.a*np.array(self.deviations).mean() - self.b*np.array(self.speeds).mean() - self.c*self.distance_to_wp + self.d*crashed)
                print("Costs updated", self.costs[-1])
                self.advantages.append(self.costs[-1] - self.value)
                # print(f"Loaded new waypoints at pose: ({current_x}, {current_y}, {current_theta}, {current_vel})")
            
            self.deviations = []
            self.speeds = []

            path, log_probs, value, self.distance_to_wp = self.waypoint_manager.load_next_waypoints(current_x, current_y, current_theta, lidar_to_bitmap(obs['scans'][0]), current_vel)

            self.lidar_scans.append(lidar_to_bitmap(obs['scans'][0]))
            self.velocities.append(np.linalg.norm([current_velx, current_vely]))
            self.paths.append(path)
            self.log_probs.append(log_probs)
            self.value = value
            
        # Compute control commands using the current waypoints.
        speed, steer = self.planner.plan(current_x, current_y, current_theta, self.waypoint_manager.waypoints)

        self.deviations.append(self.planner._current_deviation)
        self.speeds.append(np.linalg.norm([current_velx, current_vely]))

        # Return both the computed commands and the current waypoints for rendering.
        # print("self-waypoints", self.waypoint_manager.waypoints)
        return speed, steer, self.waypoint_manager.waypoints

    def shutdown():
        pass
