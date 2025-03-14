from argparse import Namespace
from controller import Controller
from weap_util.weap_container import run, train_run
import os
import yaml
import numpy as np
import gym
from waypoint_manager import WaypointManager

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