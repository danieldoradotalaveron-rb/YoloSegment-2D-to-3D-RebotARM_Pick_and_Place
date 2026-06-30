from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("rebotarm_moveit_demos")
    common = PathJoinSubstitution([share, "config", "common.yaml"])
    config_file = PathJoinSubstitution([share, "config", "pick_pompoms_cork.yaml"])

    return LaunchDescription(
        [
            Node(
                package="rebotarm_moveit_demos",
                executable="pick_pompoms_cork",
                name="pick_pompoms_cork",
                output="screen",
                parameters=[common, config_file],
            )
        ]
    )
