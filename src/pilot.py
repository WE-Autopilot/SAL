import os
import numpy as np
from numba import njit

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

"""
Pure Pursuit Helper Functions (Stateless)
"""

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along a piecewise linear trajectory.
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2
    dots = np.empty(trajectory.shape[0] - 1)
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty(projections.shape[0])
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    Given a circle (point, radius) and a piecewise linear trajectory,
    find the first intersection with the circle.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = end - start
        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c
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
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start
            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c
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
    Computes the steering angle using pure pursuit geometry.
    (This function is stateless.)
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1 / (2.0 * waypoint_y / (lookahead_distance**2))
    steering_angle = np.arctan(wheelbase / radius)
    return steering_angle

"""
Pure Pursuit Planner Class
"""

class PurePursuitPlanner:
    """
    This planner computes control commands using a traditional pure pursuit approach,
    with a constant lookahead distance and a per-segment constant velocity.
    
    For each segment from Pᵢ to Pᵢ₊₁, the constant speed is computed as:
         vᵢ = ||Pᵢ₊₁ - Pᵢ|| / T
    where T is the segment_period (a constant time period).
    The target waypoint is determined using a fixed lookahead distance.
    """
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase
        self.tlad = 1                   # Constant lookahead distance. 
        #? Might have to implement dynamic lookahead distance to account for speed increase with segment length.
        self.vgain = 0.9034             # Velocity gain.
        self.segment_period = 0.05      # Constant time period T to traverse one segment.
        self._current_segment_index = None
        self._current_deviation = 0.0

    def get_current_waypoint(self, waypoints, position, theta, lookahead_distance):
        """
        Computes the target waypoint (x, y) using geometric projection.
        Updates internal state (current segment index and deviation).
        """
        if waypoints.shape[1] < 2:
            raise ValueError("Waypoints must have at least two columns (x, y).")
        wpts = waypoints[:, :2]
        _, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        self._current_segment_index = i
        self._current_deviation = nearest_dist
        # Use the fixed lookahead distance to choose the target waypoint.
        if nearest_dist < lookahead_distance:
            target, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 is None:
                return None
            return wpts[i2, :]
        elif nearest_dist < self.tlad * 2:  # If not within lookahead, still use the nearest waypoint.
            return wpts[i, :]
        else:
            return None

    def calculate_segment_speed(self, waypoints):
        """
        Computes the constant speed for the current segment.
        For the segment from Pᵢ to Pᵢ₊₁, the speed is:
              v = ||Pᵢ₊₁ - Pᵢ|| / segment_period
        If the current segment index is invalid, returns a default speed.
        """
        idx = self._current_segment_index
        if idx is None or idx >= waypoints.shape[0] - 1:
            return 0.0
        p_current = waypoints[idx, :2]
        p_next = waypoints[idx + 1, :2]
        segment_distance = np.linalg.norm(p_next - p_current)
        return segment_distance / self.segment_period

    def plan(self, pose_x, pose_y, pose_theta, waypoints):
        """
        Computes control commands:
          1. Retrieves the target waypoint using a fixed lookahead.
          2. Computes the constant speed for the current segment.
          3. Constructs a lookahead point (with the computed speed) and computes the steering angle.
        Returns final speed (scaled by vgain) and steering angle.
        """
        position = np.array([pose_x, pose_y])
        target_waypoint = self.get_current_waypoint(waypoints, position, pose_theta, self.tlad)
        if target_waypoint is None:
            # If no valid waypoint, command zero steering.
            return 0.0, 0.0
        speed = self.calculate_segment_speed(waypoints)
        # Construct lookahead point as (x, y, speed)
        lookahead_point = np.array([target_waypoint[0], target_waypoint[1], speed])
        steering_angle = get_actuation(pose_theta, lookahead_point, position, self.tlad, self.wheelbase)
        final_speed = self.vgain * speed
        print(f"Speed: {final_speed:.2f}, Steering: {steering_angle:.2f}")
        return final_speed, steering_angle
    
    #TODO Look into and implement curvature calculation later on to replace current segment period method.
    # def _compute_curvature(self, waypoints, index):
    #     """
    #     Computes curvature at the given index using three consecutive points.
    #     Uses: curvature = (4 * area) / (d12 * d23 * d13).
    #     Returns 0 if the triangle is degenerate.
    #     """
    #     if index <= 0 or index >= len(waypoints) - 1:
    #         return 0.0
    #     p1 = waypoints[index - 1, :2]
    #     p2 = waypoints[index, :2]
    #     p3 = waypoints[index + 1, :2]
    #     d12 = np.linalg.norm(p2 - p1)
    #     d23 = np.linalg.norm(p3 - p2)
    #     d13 = np.linalg.norm(p3 - p1)
    #     area = 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))
    #     if d12 * d23 * d13 == 0:
    #         return 0.0
    #     curvature = (4 * area) / (d12 * d23 * d13)
    #     return curvature