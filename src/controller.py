from weap_util.abstract_controller import AbstractController

import numpy as np

import torch as pt
from torch.optim import AdamW
from sal import SAL
from pilot import PurePursuitPlanner
from waypoint_manager import WaypointManager
from weap_util.lidar import lidar_to_bitmap
from ppo_utils import angle_accel

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

        self.a, self.b, self.c, self.d = 1, 1, 5, 10

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

        current_vel = pt.tensor([[np.linalg.norm([current_velx, current_vely]), current_ang_vel]], dtype=pt.float32)

        current_position = np.array([current_x, current_y])

        crashed = obs["collisions"][0]

        #print(not (self.waypoint_manager.waypoints is None or len(self.waypoint_manager.waypoints) == 0), not (self.waypoint_manager.waypoints is None or len(self.waypoint_manager.waypoints) == 0) and (self.waypoint_manager.is_near_last_waypoint(current_position) or crashed))
        if not (self.waypoint_manager.waypoints is None or len(self.waypoint_manager.waypoints) == 0) and self.waypoint_manager.is_near_last_waypoint(current_position) or crashed:

            if len(self.deviations) == 0:
                self.deviations.append(0)
            if len(self.speeds) == 0:
                self.speeds.append(0)

            length = pt.linalg.norm(self.paths[-1].view(1, -1, 2), dim=-1).sum()
            accel = self.paths[-1][:, ::2].diff().mean()
            #print(length, accel)
            self.costs.append(pt.tensor([self.a*np.array(self.deviations).mean() - self.b*length - self.c*accel + self.d*crashed], dtype=pt.float32))
            # print("Costs updated", self.costs[-1])
            self.advantages.append(self.costs[-1] - self.value)
            # print(f"Loaded new waypoints at pose: ({current_x}, {current_y}, {current_theta}, {current_vel})")

        if ((self.waypoint_manager.waypoints is None or len(self.waypoint_manager.waypoints) == 0) or self.waypoint_manager.is_near_last_waypoint(current_position)) and not crashed:
            
            # print("lidar added")
            
            self.deviations = []
            self.speeds = []

            lidar_scan = pt.tensor(lidar_to_bitmap(obs['scans'][0])[None, None, ...], dtype=pt.float32)
            path, log_probs, value, self.distance_to_wp = self.waypoint_manager.load_next_waypoints(current_x, current_y, current_theta, lidar_scan, current_vel)

            self.lidar_scans.append(lidar_scan)
            self.velocities.append(current_vel)
            self.paths.append(path)
            self.log_probs.append(log_probs)
            self.value = value
            
        # Compute control commands using the current waypoints.
        speed, steer = self.planner.plan(current_x, current_y, current_theta, self.waypoint_manager.waypoints)

        self.deviations.append(self.planner._current_deviation)
        self.speeds.append(np.linalg.norm([current_velx, current_vely]))

        # Return both the computed commands and the current waypoints for rendering.
        # print("self-waypoints", self.waypoint_manager.waypoints)
        # print(len(self.lidar_scans), len(self.costs))
        return speed, steer, self.waypoint_manager.waypoints

    def shutdown():
        pass
