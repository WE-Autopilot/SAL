import time
import yaml
import gym
import numpy as np
import os
from argparse import Namespace

from f110_gym.envs.base_classes import Integrator
from pilot import PurePursuitPlanner

# Get the absolute path to the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load config from YAML using the absolute path
config_path = os.path.join(ROOT_DIR, "assets", "config.yaml")

def render_callback(env_renderer):
    """
    Custom rendering function to ensure waypoints are drawn correctly.
    Prevents unnecessary redrawing of waypoints.
    """

    e = env_renderer

    # Update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 300

    # Control how much is visible on the screen
    e.left = left - 400
    e.right = right + 400
    e.top = top + 400
    e.bottom = bottom - 400

    # Avoid redundant waypoint rendering
    if hasattr(planner, "_last_rendered_waypoints") and np.array_equal(planner.waypoints, planner._last_rendered_waypoints):
        return  # Skip rendering if waypoints haven't changed

    planner.render_waypoints(env_renderer)  # Draw waypoints only if they changed

    planner._last_rendered_waypoints = planner.waypoints.copy()  # Track last drawn waypoints

    print("Waypoints rendering called.")  # Debugging log


def main(render_on=True, use_csv=False):
    """
    Main entry point for running the F110 environment with a chosen planner.
    Set render_on=False to disable rendering and save resources.
    """
    # Load config from YAML
    with open(config_path) as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    global planner
    planner = PurePursuitPlanner(conf, wheelbase=(0.17145 + 0.15875))

    print("Planner Waypoints:\n", planner.waypoints)
    rel_wpts = planner.waypoints[:, :2].flatten()
    print("Relative waypoints:\n", rel_wpts)
    
    scale_factor = 1.0  # No scaling
    rotation_angle = 0.0  # No rotation

    # Set the new path using relative waypoints
    planner.set_path(rel_wpts, scale=scale_factor, rotation=rotation_angle)

    print("Converted and set waypoints in global coordinates:\n", planner.waypoints)

    # Create environment
    env = gym.make('f110_gym:f110-v0',
                   map=conf.map_path,
                   map_ext=conf.map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4)

    # Reset environment before rendering
    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )

    if render_on:
        print("Registering render callback...")  # Debugging log
        env.add_render_callback(render_callback)
        env.render(mode='human')

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            planner.tlad,
            planner.vgain
        )
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        if render_on:
            env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

if __name__ == '__main__':
    main(render_on=True)
