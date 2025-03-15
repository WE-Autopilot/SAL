import os
import yaml

import numpy as np
from PIL import Image

from controller import Controller
from weap_util.weap_container import run
from train_container import train_run
from f110_gym.envs.base_classes import Integrator

# Monkey-patch PIL.Image.open so that it only returns the red channel (i.e. a single-channel image)
_orig_open = Image.open
Image.open = lambda *args, **kwargs: _orig_open(*args, **kwargs).convert("RGB").split()[0]

# Enable training mode or normal execution
TRAIN_MODE = True  # Toggle this flag to switch between training and normal execution

def get_track_names(maps_folder):
    """
    Returns a list of available track names based on files in the maps folder.
    Expects each track to have a _map.png, _map.yaml, and _raceline.csv.
    """
    tracks = []
    for file in os.listdir(maps_folder):
        if file.endswith(".png"):
            track_name = file.replace(".png", "")
            yaml_path = os.path.join(maps_folder, f"{track_name}.yaml")
            csv_path = os.path.join(maps_folder, f"{track_name}.csv")
            if os.path.exists(yaml_path):
                tracks.append((track_name, yaml_path, csv_path))
    return tracks

def training_mode():
    """Runs the training mode."""
    controller = Controller()
    
    tracks = get_track_names("../assets/maps")

    for track_name, yaml_path, csv_path in tracks:
        print(f"\nStarting training on track: {track_name}")

        waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=1, usecols=[0,1,3])[::32]
        with open(yaml_path) as file:
            conf_dict = yaml.safe_load(file)

        for sx, sy, stheta in waypoints:
            conf_dict["sx"] = sx
            conf_dict["sy"] = sy
            conf_dict["stheta"] = stheta
            controller.setConf(conf_dict)
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
