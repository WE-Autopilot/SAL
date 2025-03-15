import time
import yaml
import gym
import numpy as np
import torch as pt

from time import sleep
from argparse import Namespace
from pyglet.gl import GL_POINTS, glPointSize

from f110_gym.envs.base_classes import Integrator
from ppo_utils import ppo_update

# Global variable to store the current set of waypoints for rendering
current_waypoints_global = None
# Global list to store drawn waypoint objects for later clearing
rendered_waypoints = []

def _render_callback(env_renderer):
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

def train_run(model, config_path, sx, sy, stheta, render_on=True):
    model.startup()

    global current_waypoints_global
    # Load configuration from YAML.
    with open(config_path) as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    # Create the environment.
    env = gym.make('f110_gym:f110-v0',
                   map=conf.map_path,
                   map_ext=conf.map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4)

    # Reset environment and get initial observation.
    # obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))
    obs, step_reward, done, info = env.reset(np.array([[sx, sy, stheta]]))
    # Retrieve lap count for the ego agent.
    # The environment's lap_counts is assumed to be a list or array (one entry per agent).
    lap_count = env.lap_counts[env.ego_idx] if hasattr(env, "lap_counts") else 0
    
    # Check for termination: either a crash or when 1 lap is completed.
    if done or lap_count >= 1:
        if lap_count >= 1:
            done = True

    if render_on:
        # print("Registering render callback...")
        env.add_render_callback(_render_callback)
        env.render(mode='human')

    laptime = 0.0
    start = time.time()

    speed, steer = 0, 0

    frozen_timer = 0.0

    # Main simulation loop.
    while not done:
        # Update the global variable for rendering.
        
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        #sleep(0.1)
        laptime += step_reward
        if render_on:
            env.render(mode='human')

        if speed == 0 and frozen_timer > 100:
            done = True
            obs["collisions"][0] = 1
            print("\n\n\nafk kicked ", end="")
        if speed == 0:
            frozen_timer += 1
        else:
            frozen_timer = 0
            
        speed, steer, current_waypoints = model.compute(obs)
        current_waypoints_global = current_waypoints
    print("crashed\n\n\n" if obs["collisions"] else "done\n\n\n")

    lidar_scans = pt.cat(model.lidar_scans, dim=0)
    velocities = pt.cat(model.velocities, dim=0)
    paths = pt.cat(model.paths, dim=0)
    log_probs = pt.cat(model.log_probs, dim=0)
    costs = pt.cat(model.costs, dim=0)
    advantages = pt.cat(model.advantages, dim=0)
    
    print(lidar_scans.shape, velocities.shape, paths.shape, log_probs.shape, costs.shape, advantages.shape)
    print(lidar_scans.isnan().any(), velocities.isnan().any(), paths.isnan().any(), log_probs.isnan().any(), costs.isnan().any(), advantages.isnan().any())
    ppo_update(model.sal, model.optimizer, lidar_scans, velocities, paths, log_probs.detach(), costs.detach(), advantages.detach(), mini_batch_size=16, epochs=8)
    #print(len(model.lidar_scans), len(model.velocities), len(model.paths), len(model.log_probs), len(model.costs), len(model.advantages))
    
    # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
