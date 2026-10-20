"""navfollow - offline ROS2/Nav2-style navigation and person-following simulation.

This package is the pure-Python core of the project.  It re-implements the parts of
the ROS2 navigation stack that we need in order to run the full
``SLAM-style map -> A* global plan -> DWA local control -> YOLO-style perception ->
TF2 transform -> PID/Kalman follower`` chain on a laptop without ROS2/Gazebo.

The ROS2 (rclpy) nodes live in ``ros2_ws/src/follow_robot`` and reuse the same
algorithms; they are documented as "not executed on this machine".
"""

__version__ = "1.0.0"
