import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Real-robot bringup (Raspberry Pi 5 + RPLIDAR A1 + depth camera).

    Assumes the robot's micro-ROS / serial bridge already publishes ``/odom``,
    ``/scan`` and the camera topics, and that the map was recorded with
    ``slam_toolbox`` (see docs/real_robot.md).  Start with::

        ros2 launch follow_robot follow.launch.py map:=/home/pi/maps/warehouse_a.yaml

    For a first test without localisation, pass ``use_amcl:=false`` and start
    the follow node against the default odom frame.
    """
    pkg = get_package_share_directory("follow_robot")
    nav2_bringup = get_package_share_directory("nav2_bringup")

    map_yaml = LaunchConfiguration("map")
    use_amcl = LaunchConfiguration("use_amcl")
    detector = LaunchConfiguration("detector")
    use_mqtt = LaunchConfiguration("use_mqtt")
    use_ws = LaunchConfiguration("use_ws")
    broker = LaunchConfiguration("mqtt_host")

    declare = [
        DeclareLaunchArgument("map", default_value="/home/pi/maps/warehouse_a.yaml"),
        DeclareLaunchArgument("use_amcl", default_value="true"),
        DeclareLaunchArgument("detector", default_value="yolo"),
        DeclareLaunchArgument("use_mqtt", default_value="true"),
        DeclareLaunchArgument("use_ws", default_value="true"),
        DeclareLaunchArgument("mqtt_host", default_value="127.0.0.1"),
    ]

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, "launch", "localization_launch.py")),
        launch_arguments={
            "map": map_yaml,
            "params_file": os.path.join(pkg, "config", "nav2_params.yaml"),
            "use_sim_time": "false",
        }.items(),
        condition=IfCondition(use_amcl),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, "launch", "navigation_launch.py")),
        launch_arguments={
            "params_file": os.path.join(pkg, "config", "nav2_params.yaml"),
            "use_sim_time": "false",
        }.items(),
    )

    tf_broadcaster = Node(
        package="follow_robot",
        executable="tf_broadcaster",
        name="tf_broadcaster",
        parameters=[{"publish_odom": True, "use_sim_time": False}],
        output="screen",
    )

    perception = Node(
        package="follow_robot",
        executable="perception_node",
        name="perception_node",
        parameters=[
            {
                "detector": detector,
                "use_sim_time": False,
                "depth_units": 0.001,
                "sync_slop": 0.05,
            }
        ],
        remappings=[
            ("/camera/color/image_raw", "/camera/color/image_raw"),
            ("/cmd_vel", "/cmd_vel"),
        ],
        output="screen",
    )

    controller = Node(
        package="follow_robot",
        executable="follow_controller",
        name="follow_controller",
        parameters=[{"use_sim_time": False, "max_v": 0.45, "max_w": 1.0}],
        output="screen",
    )

    mqtt = Node(
        package="follow_robot",
        executable="mqtt_bridge",
        name="mqtt_bridge",
        parameters=[{"host": broker, "prefix": "follow_robot"}],
        condition=IfCondition(use_mqtt),
        output="screen",
    )

    ws = Node(
        package="follow_robot",
        executable="ws_bridge",
        name="ws_bridge",
        parameters=[{"host": "0.0.0.0", "port": 8765}],
        condition=IfCondition(use_ws),
        output="screen",
    )

    return LaunchDescription(declare + [localization, navigation, tf_broadcaster, perception, controller, mqtt, ws])
