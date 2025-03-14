from controller import Controller
from weap_util.weap_container import run
import os
import yaml
import numpy as np
import gym
from waypoint_manager import WaypointManager

# Enable training mode or normal execution
TRAIN_MODE = True  # Toggle this flag to switch between training and normal execution
RANDOM_MOVEMENT = False  # Toggle this flag to enable random movement vs use of pure-pursuit

def load_yaml(yaml_path):
    """Loads the YAML configuration file."""
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def save_yaml(yaml_path, config):
    """Saves updated YAML configurations."""
    with open(yaml_path, "w") as file:
        yaml.dump(config, file)

def get_track_names(maps_folder):
    """
    Returns a list of available track names based on files in the maps folder.
    Expects each track to have a _map.png, _map.yaml, and _raceline.csv.
    """
    tracks = [] 
    for file in os.listdir(maps_folder):
        if file.endswith("_map.png"):
            track_name = file.replace("_map.png", "")
            yaml_path = os.path.join(maps_folder, f"{track_name}_map.yaml")
            csv_path = os.path.join(maps_folder, f"{track_name}_raceline.csv")
            if os.path.exists(yaml_path) and os.path.exists(csv_path):
                tracks.append((track_name, yaml_path, csv_path))
    return tracks


# def training_mode():
#     """
#     Cycles through each track and its starting waypoints.
#     For each starting pose:
#       - Sets the car's initial position and heading (based on the waypoints in the CSV)
#       - Resets the F1Tenth environment with that pose
#       - Runs the simulation until a crash occurs or the target waypoint is reached
#       - Then closes the environment and proceeds to the next starting pose
#     """
#     maps_folder = "../assets/maps"  # Adjust this to the correct folder path
#     track_data = get_track_names(maps_folder)
    
#     if not track_data:
#         print("No valid tracks found!")
#         return

#     for track_name, csv_path in track_data:
#         print(f"\nStarting training on track: {track_name}")
#         # Load waypoints from CSV; assuming two columns: x, y.
#         waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=0)
#         num_waypoints = len(waypoints)
        
#         # Cycle through starting waypoints every 32 indices.
#         for start_idx in range(0, num_waypoints, 32):
#             print(f"\nStarting at waypoint index: {start_idx}")
#             # Get the starting pose (x, y) from the CSV.
#             sx, sy = waypoints[start_idx, 0], waypoints[start_idx, 1]
#             # Compute heading based on the next waypoint (or use current if at end).
#             next_idx = min(start_idx + 1, num_waypoints - 1)
#             nx, ny = waypoints[next_idx, 0], waypoints[next_idx, 1]
#             stheta = np.arctan2(ny - sy, nx - sx)
            
#             # Create the environment.
#             env = gym.make('f110_gym:f110-v0', render_mode="human")
#             # The environment expects a poses array. Here we assume one agent,
#             # so we create a 1x3 array with [x, y, theta].
#             poses = np.array([[sx, sy, stheta]])
#             obs = env.reset(poses)
            
#             # Run for up to 500 steps.
#             for step in range(500):
#                 # For testing, use a dummy action: zero steering, constant speed.
#                 action = np.array([[0.0, 1.0]])  # [steer, speed]
#                 obs, reward, done, truncated, info = env.step(action)
                
#                 # Check for termination (e.g. crash).
#                 if done:
#                     print("Crash or termination detected; moving to next starting waypoint.")
#                     break
                
#                 # Optionally, check if the car is near the target waypoint.
#                 # (Here we use a 1-meter threshold.)
#                 current_pose = np.array([obs["poses_x"][0], obs["poses_y"][0]])
#                 target_pose = np.array([nx, ny])
#                 if np.linalg.norm(current_pose - target_pose) < 1.0:
#                     print("Target waypoint reached; moving to next starting waypoint.")
#                     break
            
#            env.close()

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

        # Load waypoints assuming CSV has two columns: x and y
        waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=0)
        num_waypoints = len(waypoints)

        for start_idx in range(0, num_waypoints, 32):
            print(f"Starting at waypoint {start_idx} on {track_name}")

            # Set initial car position and heading using columns 0 (x) and 1 (y)
            sx, sy = waypoints[start_idx, 0], waypoints[start_idx, 1]
            next_idx = min(start_idx + 1, num_waypoints - 1)
            nx, ny = waypoints[next_idx, 0], waypoints[next_idx, 1]
            stheta = np.arctan2(ny - sy, nx - sx)

            # Convert numpy scalars to native Python floats to avoid YAML serialization issues
            config["sx"] = float(sx)
            config["sy"] = float(sy)
            config["stheta"] = float(stheta)
            save_yaml(yaml_path, config)

            # Initialize the F1Tenth environment.
            env = gym.make('f110_gym:f110-v0', render_mode="human")
            
            # Create poses array for the number of agents the environment expects.
            num_agents = getattr(env, "num_agents", 1)
            poses = np.array([[sx, sy, stheta]])
            if poses.shape[0] != num_agents:
                poses = np.repeat(poses, num_agents, axis=0)
            
            reset_ret = env.reset(poses)
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            
            controller = Controller()
            waypoint_manager = WaypointManager(config)
            deviations, distances = [], []
            crashed = False

            for step in range(500):
                # Fake LiDAR scan (Placeholder)
                lidar_image = "lidar.scan()"  # Placeholder

                # Use WaypointManager to update waypoints (instead of SAL for now)
                waypoint_manager.load_next_waypoints()

                # Get steering and speed from controller
                speed, steer, _ = controller.compute(obs)

                # Step the environment forward
                obs, step_reward, done, truncated, info = env.step(np.array([[steer, speed]]))

                # Compute deviation from waypoints
                deviation = controller.planner._current_deviation
                target_distance = np.linalg.norm([nx - sx, ny - sy])
                deviations.append(deviation)
                distances.append(target_distance)

                if done:
                    print("Crash detected, relocating car to nearest waypoint...")
                    crashed = True

                    # Find the closest waypoint to the car's current position.
                    min_dist = float("inf")
                    closest_idx = 0
                    # Use columns 0 and 1 for x and y respectively
                    for i in range(len(waypoints)):
                        dist = np.linalg.norm([
                            obs["poses_x"][0] - waypoints[i, 0], 
                            obs["poses_y"][0] - waypoints[i, 1]
                        ])
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i

                    # Reset car at closest waypoint.
                    sx, sy = waypoints[closest_idx, 0], waypoints[closest_idx, 1]
                    if closest_idx < num_waypoints - 1:
                        nx, ny = waypoints[closest_idx + 1, 0], waypoints[closest_idx + 1, 1]
                    else:
                        nx, ny = waypoints[closest_idx - 1, 0], waypoints[closest_idx - 1, 1]
                    stheta = np.arctan2(ny - sy, nx - sx)

                    # Create new poses array for reset
                    poses = np.array([[sx, sy, stheta]])
                    if poses.shape[0] != num_agents:
                        poses = np.repeat(poses, num_agents, axis=0)
                    
                    reset_ret = env.reset(poses)
                    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
                    break  # Stop current iteration after crash

            # Pretend to update SAL (for future integration)
            print(f"Run completed: Crash={crashed}, Avg Deviation={np.mean(deviations):.2f}")
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
