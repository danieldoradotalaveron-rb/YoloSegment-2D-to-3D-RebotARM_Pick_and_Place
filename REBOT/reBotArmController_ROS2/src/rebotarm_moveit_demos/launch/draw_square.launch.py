from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [
            FindPackageShare("rebotarm_moveit_demos"),
            "config",
            "draw_square.yaml",
        ]
    )

    return LaunchDescription(
        [
            Node(
                package="rebotarm_moveit_demos",
                executable="draw_square",
                name="draw_square",
                output="screen",
                parameters=[config_file],
            )
        ]
    )
