from argparse import Namespace
from controller import Controller
from weap_util.weap_container import run
import os
import yaml
import numpy as np
import gym
from waypoint_manager import WaypointManager
from f110_gym.envs.base_classes import Integrator

# Enable training mode or normal execution
TRAIN_MODE = True  # Toggle this flag to switch between training and normal execution
RANDOM_MOVEMENT = False  # Toggle this flag to enable random movement vs use of pure-pursuit

def get_track_names(maps_folder):
    """
    Returns a list of available track names based on files in the maps folder.
    Expects each track to have a _map.png, _map.yaml, and _raceline.csv.
    """
    global tracks
    tracks = [] 
    for file in os.listdir(maps_folder):
        if file.endswith(".png"):
            track_name = file.replace(".png", "")
            yaml_path = os.path.join(maps_folder, f"{track_name}.yaml")
            csv_path = os.path.join(maps_folder, f"{track_name}.csv")
            if os.path.exists(yaml_path):
                tracks.append((track_name, yaml_path, csv_path))
    return tracks

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

    if render_on:
        print("Registering render callback...")
        env.add_render_callback(_render_callback)
        env.render(mode='human')

    laptime = 0.0
    start = time.time()

    # Main simulation loop.
    while not done:
        speed, steer, current_waypoints = model.compute(obs)
        # Update the global variable for rendering.
        current_waypoints_global = current_waypoints
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        if render_on:
            env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

#! Combined Implementation
def training_mode():
    """Runs the training mode."""
    controller = Controller()
    
    tracks = get_track_names("../assets/maps")

    for track_name, yaml_path, csv_path in tracks:
        print(f"\nStarting training on track: {track_name}")

        waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=1, usecols=[0,1,3])[::32]
        with open(yaml_path) as file:
            conf_dict = yaml.safe_load(file)
            conf = Namespace(**conf_dict)
        
        for sx, sy, stheta in waypoints:
            conf_dict["sx"] = sx
            conf_dict["sy"] = sy
            conf_dict["stheta"] = stheta
            train_run(controller, yaml_path, sx, sy, stheta, True)

def normal_mode():
    """Runs the normal driving mode."""
    controller = Controller()
    run(controller, "../assets/config.yaml", True)

if __name__ == "__main__":
    if TRAIN_MODE:
        training_mode()
    else:
        normal_mode()