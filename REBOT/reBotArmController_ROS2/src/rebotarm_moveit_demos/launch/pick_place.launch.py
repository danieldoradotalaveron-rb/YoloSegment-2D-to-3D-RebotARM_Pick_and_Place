from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [
            FindPackageShare("rebotarm_moveit_demos"),
            "config",
            "pick_place.yaml",
        ]
    )

    return LaunchDescription(
        [
            Node(
                package="rebotarm_moveit_demos",
                executable="pick_place",
                name="pick_place",
                output="screen",
                parameters=[config_file],
            )
        ]
    )
