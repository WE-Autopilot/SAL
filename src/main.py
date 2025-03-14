from controller import Controller
from weap_util.weap_container import run
import os
import yaml
import numpy as np
import gym
from waypoint_manager import WaypointManager

# Enable training mode or normal execution
TRAIN_MODE = False  # Toggle this flag

def load_yaml(yaml_path):
    """Loads the YAML configuration file."""
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def save_yaml(yaml_path, config):
    """Saves updated YAML configurations."""
    with open(yaml_path, "w") as file:
        yaml.dump(config, file)

def get_track_names(maps_folder):
    """Returns a list of available track names based on files in the maps folder."""
    tracks = []
    for file in os.listdir(maps_folder):
        if file.endswith("_map.png"):
            track_name = file.replace("_map.png", "")
            yaml_path = os.path.join(maps_folder, f"{track_name}_map.yaml")
            csv_path = os.path.join(maps_folder, f"{track_name}_raceline.csv")
            if os.path.exists(yaml_path) and os.path.exists(csv_path):
                tracks.append((track_name, yaml_path, csv_path))
    return tracks

def training_mode():
    """Runs the training process."""
    maps_folder = "../assets/maps"
    track_names = get_track_names(maps_folder)

    if not track_names:
        print("No valid tracks found!")
        return

    for track_name, yaml_path, csv_path in track_names:
        print(f"Starting training on track: {track_name}")
        config = load_yaml(yaml_path)

        waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=0)
        num_waypoints = len(waypoints)

        for start_idx in range(0, num_waypoints, 32):
            print(f"Starting at waypoint {start_idx} on {track_name}")

            # Set initial car position and heading
            sx, sy = waypoints[start_idx, 1], waypoints[start_idx, 2]
            next_idx = min(start_idx + 1, num_waypoints - 1)
            nx, ny = waypoints[next_idx, 1], waypoints[next_idx, 2]
            stheta = np.arctan2(ny - sy, nx - sx)

            # Update config with new starting pose
            config["sx"], config["sy"], config["stheta"] = sx, sy, stheta
            save_yaml(yaml_path, config)

            # Initialize the F1Tenth environment
            env = gym.make('f110_gym:f110-v0', render_mode="human")
            obs, _ = env.reset(options={"map": track_name, "poses": np.array([[sx, sy, stheta]])})

            controller = Controller()
            waypoint_manager = WaypointManager(config)
            deviations, distances = [], []
            crashed = False
            done = False

            for step in range(500):
                # Fake LiDAR scan (Placeholder)
                lidar_image = "lidar.scan()"  # Someone else is implementing this I think

                #! Pretend to call SAL (Replace this when SAL is ready)
                # from sal import SAL
                # sal = SAL()
                # path = sal(lidar_image)

                # Use `WaypointManager.load_next_waypoints()` instead of SAL for now
                waypoint_manager.load_next_waypoints()

                # Get steering and speed from controller
                speed, steer, _ = controller.compute(obs)

                # Step the environment forward
                obs, step_reward, done, truncated, info = env.step(np.array([[steer, speed]]))

                # Compute deviation from waypoints
                deviation = controller.planner._current_deviation
                target_distance = np.linalg.norm([sx - nx, sy - ny])

                deviations.append(deviation)
                distances.append(target_distance)

                if done:
                    print("Crash detected, relocating car to nearest waypoint...")
                    crashed = True

                    # Find the closest waypoint and reset the car
                    min_dist = float("inf")
                    closest_idx = 0
                    for i in range(len(waypoints)):
                        dist = np.linalg.norm([obs["poses_x"][0] - waypoints[i, 1], obs["poses_y"][0] - waypoints[i, 2]])
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i

                    # Reset car at closest waypoint
                    sx, sy = waypoints[closest_idx, 1], waypoints[closest_idx, 2]
                    stheta = np.arctan2(ny - sy, nx - sx)
                    env.reset(options={"map": track_name, "poses": np.array([[sx, sy, stheta]])})

                    break  # Stop current iteration

            # Pretend to update SAL
            # sal.update(deviations, target_distance, crashed)
            print(f"Run completed: Crash={crashed}, Avg Deviation={np.mean(deviations):.2f}")

            # Move to next starting waypoint (or next map)
            env.close()

def normal_mode():
    """Runs the normal driving mode."""
    controller = Controller()
    run(controller, "../assets/config.yaml", False)

if __name__ == "__main__":
    if TRAIN_MODE:
        training_mode()
    else:
        normal_mode()
