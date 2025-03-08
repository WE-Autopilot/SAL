import time
import yaml
import gym
import numpy as np
import os
from argparse import Namespace
from pyglet.gl import GL_POINTS, glPointSize

from f110_gym.envs.base_classes import Integrator
from pilot import PurePursuitPlanner
from waypoint_manager import WaypointManager

# Get the absolute path to the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(ROOT_DIR, "assets", "config.yaml")

# Global variable to store the current set of waypoints for rendering
current_waypoints_global = None
# Global list to store drawn waypoint objects for later clearing
rendered_waypoints = []

def render_callback(env_renderer):
    """
    Custom render callback that updates the camera and renders waypoints.
    Uses the global current_waypoints_global variable for drawing.
    """
    global rendered_waypoints
    e = env_renderer

    # Update camera to follow the car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 300

    e.left = left - 400
    e.right = right + 400
    e.top = top + 400
    e.bottom = bottom - 400

    # Clear previously drawn waypoints
    for obj in rendered_waypoints:
        obj.delete()
    rendered_waypoints = []

    # Render new waypoints using the global current_waypoints_global
    if current_waypoints_global is not None and current_waypoints_global.shape[0] > 0:
        points = current_waypoints_global[:, :2]
        scaled_points = 50 * points  # Scale factor for visualization
        glPointSize(5)  # Increase point size for clarity
        for i in range(len(points)):
            obj = e.batch.add(
                1,
                GL_POINTS,
                None,
                ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                ('c3B/stream', [255, 0, 0])
            )
            rendered_waypoints.append(obj)
    # print("Render callback: waypoints drawn.")

def compute_callback(planner: PurePursuitPlanner, waypoint_manager: WaypointManager, obs):
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
    # if (waypoint_manager.waypoints is None or len(waypoint_manager.waypoints) == 0) \
    #    or waypoint_manager.is_near_last_waypoint(current_position):
    #     waypoint_manager.load_next_waypoints(current_x, current_y, current_theta)
    

    # Check if the vehicle is near the end of the current waypoint batch.
    if waypoint_manager.is_near_last_waypoint(current_position):
        waypoint_manager.load_next_waypoints()
    
    # Compute control commands using the current waypoints.
    speed, steer = planner.plan(current_x, current_y, current_theta, waypoint_manager.waypoints)

    
    # Return both the computed commands and the current waypoints for rendering.
    return speed, steer, waypoint_manager.waypoints

def main(render_on=True):
    global current_waypoints_global
    # Load configuration from YAML.
    with open(config_path) as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    global planner, waypoint_manager
    # Instantiate the PurePursuitPlanner and WaypointManager.
    planner = PurePursuitPlanner(wheelbase=(0.17145 + 0.15875))
    waypoint_manager = WaypointManager(conf)

    # Create the environment.
    env = gym.make('f110_gym:f110-v0',
                   map=conf.map_path,
                   map_ext=conf.map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4)

    # Reset environment and get initial observation.
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    
    if render_on:
        print("Registering render callback...")
        env.add_render_callback(render_callback)
        env.render(mode='human')

    laptime = 0.0
    start = time.time()

    # Main simulation loop.
    while not done:
        speed, steer, current_waypoints = compute_callback(planner, waypoint_manager, obs)
        # Update the global variable for rendering.
        current_waypoints_global = current_waypoints
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        if render_on:
            env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

if __name__ == '__main__':
    main(render_on=True)
