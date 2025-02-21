import os
import numpy as np

from numba import njit
from pyglet.gl import GL_POINTS, glPointSize


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

"""
Pure Pursuit Helper Functions
"""

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast.
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]  # differences between consecutive waypoints
    l2s   = diffs[:,0]**2 + diffs[:,1]**2         # squared lengths of each segment

    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)

    # nearest point, dist to nearest point, param t, segment index
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    Given a circle (point, radius) and a piecewise linear trajectory, find the
    first point on the trajectory that intersects with the circle.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None

    trajectory = np.ascontiguousarray(trajectory)

    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:] + 1e-6
        V = end - start

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius * radius
        )
        discriminant = b*b - 4*a*c

        if discriminant < 0:
            continue

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if 0.0 <= t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if 0.0 <= t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        else:
            if 0.0 <= t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif 0.0 <= t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    # Wrap around if no intersection found and wrap=True
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius * radius
            )
            discriminant = b*b - 4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if 0.0 <= t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif 0.0 <= t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Return desired speed and steering angle using pure pursuit geometry.
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner that uses a simple pure pursuit strategy.
    """
    def __init__(self, conf, wheelbase):
        self.wheelbase = wheelbase
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.0  # maximum reacquire distance
        self.drawn_waypoints = []
        
        # Default pure pursuit parameters
        self.tlad = 0.8246   # Lookahead distance (tlad)
        self.vgain = 0.9034  # Velocity gain (vgain)

        self._latest_speed = 0.0
        self._latest_steering_angle = 0.0
        self._current_segment_index = None
        self._current_deviation = 0.0
        
    def load_waypoints(self, conf):
        """
        Load waypoints from file.
        """
        waypoint_file = os.path.join(ROOT_DIR, "assets", os.path.basename(conf.wpt_path))

        if not os.path.exists(waypoint_file):
            raise FileNotFoundError(f"Waypoint file not found: {waypoint_file}")

        self.waypoints = np.loadtxt(waypoint_file,
                                    delimiter=conf.wpt_delim,
                                    skiprows=conf.wpt_rowskip)

    @staticmethod
    def convert_to_global_waypoints(rel_waypoints, start_x, start_y, scale=1.0, rotation=0.0):
        """
        Convert a relative path vector to global waypoints.

        Parameters:
        - rel_waypoints: np.ndarray, shape (N,2), relative x, y positions
        - start_x: float, global starting x position
        - start_y: float, global starting y position
        - scale: float, scaling factor applied to waypoints
        - rotation: float, rotation angle in radians (counterclockwise)

        Returns:
        - global_waypoints: np.ndarray, shape (N,2), global x, y positions
        """
        if rel_waypoints.ndim != 2 or rel_waypoints.shape[1] != 2:
            raise ValueError("rel_waypoints must have shape (N,2)")

        # Apply scaling
        scaled_waypoints = rel_waypoints * scale

        # Apply cumulative summation to get absolute positions
        cumsum_waypoints = np.cumsum(scaled_waypoints, axis=0)

        # Apply rotation using rotation matrix
        cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        rotated_waypoints = cumsum_waypoints @ rotation_matrix.T

        # Translate to global coordinates
        global_waypoints = rotated_waypoints + np.array([start_x, start_y])

        return global_waypoints

    def set_path(self, wpts_vector, scale=1.0, rotation=0.0):
        """
        Accepts a numpy array of relative x, y coordinates, reshapes them, 
        converts them to global waypoints, and overrides self.waypoints.
        """
        self.drawn_waypoints.clear()  # Clear old waypoints before setting new ones

        # Reshape the flat array to (N,2)
        rel_waypoints = wpts_vector.reshape((-1, 2))

        # Convert to global waypoints using tip-to-tail transformation
        global_waypoints = self.convert_to_global_waypoints(
            rel_waypoints, 
            self.conf.sx, self.conf.sy,  # Start position from config.yaml
            scale=scale, 
            rotation=rotation
        )

        # Keep only 16 waypoints at a time
        self.waypoints = global_waypoints[:16]

        print("New waypoints set:\n", self.waypoints)

    def render_waypoints(self, e):
        """
        Render waypoints dynamically, loading new sets when all 16 waypoints are used.
        """

        if self.waypoints is None or self.waypoints.shape[0] == 0:
            print("No waypoints available for rendering.")
            return

        print(f"Rendering {len(self.waypoints)} waypoints...")

        # Ensure waypoints array has at least (x, y)
        if self.waypoints.shape[1] < 2:
            raise ValueError(f"Waypoints array must have at least (x, y) columns, but got shape {self.waypoints.shape}")

        # Use only the first two columns (x, y)
        points = self.waypoints[:16, :2]

        # Scale for rendering
        scaled_points = 50.0 * points

        glPointSize(10)  # Increase point size

        # Draw or update waypoints
        for i in range(len(points)):
            if len(self.drawn_waypoints) < len(points):
                print(f"Adding waypoint {i}: {scaled_points[i]}")
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ('c3B/stream', [255, 0, 0])  # Red color
                )
                self.drawn_waypoints.append(b)
            else:
                print(f"Updating waypoint {i}: {scaled_points[i]}")
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0
                ]

        # Detect when all waypoints are used and load a new set
        if hasattr(self, "waypoint_index"):
            self.waypoint_index += 1
        else:
            self.waypoint_index = 1

        if self.waypoint_index % 16 == 0:  # Every 16 waypoints, generate a new set
            print("Generating new waypoints...")
            self.load_new_waypoints()

    def load_new_waypoints(self):
        """
        Refresh the waypoints dynamically, either from CSV or randomly generated.
        """
        if self.use_csv:
            # Load next 16 waypoints from the CSV file
            start_idx = (self.waypoint_index // 16) * 16
            end_idx = start_idx + 16
            if end_idx > self.waypoints.shape[0]:  # Loop back if needed
                start_idx = 0
                end_idx = 16
            self.waypoints = self.waypoints[start_idx:end_idx, :2]
            print(f"Loaded next 16 waypoints from CSV (indices {start_idx}-{end_idx})")
        else:
            # Generate new random waypoints to simulate SAL input
            num_waypoints = 16
            max_distance = 2.0  # Max distance each waypoint can be from the previous one
            random_offsets = np.random.uniform(-max_distance, max_distance, (num_waypoints, 2))
            self.waypoints = random_offsets
            print("Generated new set of random waypoints.")

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        Gets the current waypoint to follow.
        """
        if waypoints.shape[1] >= 2:
            wpts = waypoints[:, :2]  # Only take (x, y) columns
        else:
            raise ValueError(f"Waypoints array has invalid shape {waypoints.shape}, expected at least 2 columns (x, y).")

        nearest_point_, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)

        self._current_segment_index = i
        self._current_deviation = nearest_dist

        constant_velocity = 6.0

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]

            # speed if we use it from csv
            # current_waypoint[2] = waypoints[i, self.conf.wpt_vind]

            # speed without csv, constant
            current_waypoint[2] = 6.0
            
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            # If close, but not within lookahead distance, reacquire the nearest waypoint.
            return np.append(
                wpts[i, :],
                constant_velocity
                # waypoints[i, self.conf.wpt_vind]
            )
        else:
            # Too far from track
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        Computes the speed and steering angle command using pure pursuit.
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        # If no valid waypoint found, just slow down and go straight
        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position,
            lookahead_distance, self.wheelbase
        )
        # Scale speed
        speed = vgain * speed

        self._latest_speed = speed
        self._latest_steering_angle = steering_angle

        return speed, steering_angle
    
    def get_controls(self):
        """
        Return the most recently computed steering angle and speed.
        This is useful if you need these values outside the main plan loop.
        """
        return self._latest_steering_angle, self._latest_speed
    
    def get_current_segment_index(self):
        """
        Returns the index of the waypoint at the start of 
        the line segment we are projecting on.
        """
        return self._current_segment_index
    
    def get_path_deviation(self):
        """
        Returns the distance (in meters) from the car to the path.
        """
        return self._current_deviation