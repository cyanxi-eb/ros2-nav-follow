import os
from glob import glob

from setuptools import setup

package_name = "follow_robot"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cyanxi-eb",
    maintainer_email="cyanxi-eb@users.noreply.github.com",
    description="Person following + Nav2 navigation for ROS2 Jazzy",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = follow_robot.perception_node:main",
            "follow_controller = follow_robot.follow_controller:main",
            "tf_broadcaster = follow_robot.tf_broadcaster:main",
            "mqtt_bridge = follow_robot.mqtt_bridge:main",
            "ws_bridge = follow_robot.ws_bridge:main",
        ],
    },
)
