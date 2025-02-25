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
        Load waypoints as relative and convert them to global only once.
        """
        waypoint_file = os.path.join(ROOT_DIR, "assets", os.path.basename(conf.wpt_path))

        if not os.path.exists(waypoint_file):
            raise FileNotFoundError(f"Waypoint file not found: {waypoint_file}")

        # Read CSV
        raw_waypoints = np.loadtxt(waypoint_file,
                                delimiter=conf.wpt_delim,
                                skiprows=conf.wpt_rowskip)[::2]

        # Ensure correct column indexing
        if raw_waypoints.shape[1] < max(conf.wpt_xind, conf.wpt_yind) + 1:
            raise ValueError(f"Waypoint file has {raw_waypoints.shape[1]} columns, "
                            f"but expected at least {max(conf.wpt_xind, conf.wpt_yind) + 1}.")

        # Extract x and y coordinates
        rel_waypoints = raw_waypoints[:, [conf.wpt_xind, conf.wpt_yind]]

        # Store all waypoints
        self.original_waypoints = rel_waypoints

        # Initialize with the first 16 waypoints
        self.waypoints = rel_waypoints[:16]
        self.waypoint_index = 0  # Track the index of the current batch

        # # Convert to global coordinates
        # self.waypoints = self.convert_to_global_waypoints(
        #     rel_waypoints, conf.sx, conf.sy, scale=1.0
        # )

    @staticmethod
    def convert_to_global_waypoints(rel_waypoints, start_x, start_y, scale=1.0, rotation=1.37079632679):
        """
        Convert relative waypoints into global waypoints.
        """
        if rel_waypoints.ndim != 2 or rel_waypoints.shape[1] != 2:
            raise ValueError("rel_waypoints must have shape (N,2)")

        # Apply scaling
        scaled_waypoints = rel_waypoints * scale

        # Apply cumulative summation to get absolute positions
        cumsum_waypoints = np.cumsum(scaled_waypoints, axis=0)

        # Apply rotation
        cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        rotated_waypoints = cumsum_waypoints @ rotation_matrix.T

        # Translate to global coordinates
        return rotated_waypoints + np.array([start_x, start_y])


    def set_path(self, wpts_vector, scale=1.0, rotation=0.0):

        self.drawn_waypoints.clear()

        # Reshape to (N,2)
        global_waypoints = wpts_vector.reshape((-1, 2))

        # Store the new waypoints
        self.waypoints = global_waypoints[:16]

    def render_waypoints(self, e):
        """
        Render waypoints dynamically.
        """
        if self.waypoints is None or self.waypoints.shape[0] == 0:
            return

        # Ensure waypoints array has at least (x, y)
        if self.waypoints.shape[1] < 2:
            raise ValueError(f"Waypoints array must have at least (x, y) columns, but got shape {self.waypoints.shape}")

        # Clear previously drawn waypoints
        for b in self.drawn_waypoints:
            b.delete()
        self.drawn_waypoints.clear()

        # Use only the first two columns (x, y)
        points = self.waypoints[:16, :2]

        # Scale for rendering
        scaled_points = 50 * points

        glPointSize(5)  # Increase point size

        # Draw new waypoints
        for i in range(len(points)):
            b = e.batch.add(
                1,
                GL_POINTS,
                None,
                ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                ('c3B/stream', [255, 0, 0])  # Red color
            )
            self.drawn_waypoints.append(b)

    def _is_near_last_waypoints(self, position, radius=1.0):
        """
        Check if the vehicle is within a small radius of the second or third-to-last waypoint.
        """
        if self.waypoints.shape[0] < 3:
            return False  # Not enough waypoints to check

        last_waypoints = self.waypoints[-3:-1, :2]  # Get the last three waypoints

        for waypoint in last_waypoints:
            distance = np.linalg.norm(position - waypoint)
            if distance < radius:
                return True  # Vehicle is close to the second or third-to-last waypoint

        return False

    def load_next_waypoints(self):
        """
        Load the next 16 waypoints from self.original_waypoints.
        """
        self.waypoint_index += 16  # Move to the next set of waypoints

        start_idx = self.waypoint_index
        end_idx = start_idx + 16

        if start_idx >= len(self.original_waypoints):  # If we reach the end, loop back
            self.waypoint_index = 0
            start_idx = 0
            end_idx = 16

        self.waypoints = self.original_waypoints[start_idx:end_idx, :2]
        print(f"Loaded waypoints {start_idx} to {end_idx}")

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

        if self._is_near_last_waypoints(position):
            print("Approaching last waypoints, loading next 16...")
            self.load_next_waypoints()

        constant_velocity = 6.0

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.empty((3,))
            current_waypoint[0:2] = wpts[i2, :]
            current_waypoint[2] = constant_velocity
            return current_waypoint

        elif nearest_dist < self.max_reacquire:
            return np.append(
                wpts[i, :],
                constant_velocity
            )
        else:
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