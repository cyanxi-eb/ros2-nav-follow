import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Gazebo Harmonic + Nav2 + SLAM + perception/follow bringup.

    Usage::

        ros2 launch follow_robot sim.launch.py use_slam:=true
        ros2 launch follow_robot sim.launch.py use_slam:=false map:=warehouse_a.yaml

    ``use_slam:=true`` starts SLAM Toolbox for mapping; with ``false`` the
    map_server loads the exported occupancy grid from ``maps/``.
    """
    pkg = get_package_share_directory("follow_robot")
    nav2_bringup = get_package_share_directory("nav2_bringup")

    use_slam = LaunchConfiguration("use_slam")
    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    map_yaml = LaunchConfiguration("map")
    use_mqtt = LaunchConfiguration("use_mqtt")
    use_ws = LaunchConfiguration("use_ws")
    detector = LaunchConfiguration("detector")

    declare = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_slam", default_value="true"),
        DeclareLaunchArgument("world", default_value=os.path.join(pkg, "worlds", "warehouse.sdf")),
        DeclareLaunchArgument("map", default_value=os.path.join(pkg, "maps", "warehouse_a.yaml")),
        DeclareLaunchArgument("use_mqtt", default_value="false"),
        DeclareLaunchArgument("use_ws", default_value="false"),
        DeclareLaunchArgument("detector", default_value="yolo"),
        DeclareLaunchArgument("x", default_value="2.0"),
        DeclareLaunchArgument("y", default_value="6.5"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
    ]

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": ["-r ", world]}.items(),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "robot_state.launch.py")),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "follow_robot", "-x", LaunchConfiguration("x"),
                   "-y", LaunchConfiguration("y"), "-z", "0.05", "-Y", LaunchConfiguration("yaw")],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, "launch", "navigation_launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": os.path.join(pkg, "config", "nav2_params.yaml"),
            "slam": use_slam,
        }.items(),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(use_slam),
    )

    perception = Node(
        package="follow_robot",
        executable="perception_node",
        name="perception_node",
        parameters=[{"use_sim_time": use_sim_time, "detector": detector}],
        output="screen",
    )

    controller = Node(
        package="follow_robot",
        executable="follow_controller",
        name="follow_controller",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    tf_broadcaster = Node(
        package="follow_robot",
        executable="tf_broadcaster",
        name="tf_broadcaster",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg, "rviz", "follow.rviz")],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    mqtt = Node(
        package="follow_robot",
        executable="mqtt_bridge",
        name="mqtt_bridge",
        parameters=[{"host": "127.0.0.1"}],
        condition=IfCondition(use_mqtt),
        output="screen",
    )

    ws = Node(
        package="follow_robot",
        executable="ws_bridge",
        name="ws_bridge",
        parameters=[{"host": "0.0.0.0"}],
        condition=IfCondition(use_ws),
        output="screen",
    )

    return LaunchDescription(
        declare
        + [
            gz_sim,
            robot_state_publisher,
            TimerAction(period=3.0, actions=[spawn_robot]),
            nav2,
            slam,
            TimerAction(period=5.0, actions=[tf_broadcaster, perception, controller, rviz, mqtt, ws]),
        ]
    )
