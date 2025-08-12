#!/usr/bin/env python3
"""
ROS2 Launch file for Biped Policy Inference Node

This launch file provides a flexible way to start the biped inference node
with configurable parameters.

Usage:
    ros2 launch biped_inference.launch.py experiment_name:=my-experiment checkpoint:=200
    
    ros2 launch biped_inference.launch.py device:=cpu control_frequency:=25.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    # Declare launch arguments
    experiment_name_arg = DeclareLaunchArgument(
        'experiment_name',
        default_value='biped-walking',
        description='Name of the trained experiment to load'
    )
    
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint',
        default_value='100',
        description='Checkpoint number to load'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for inference (cuda/cpu)'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='50.0',
        description='Control loop frequency in Hz'
    )
    
    timeout_duration_arg = DeclareLaunchArgument(
        'timeout_duration',
        default_value='1.0',
        description='Maximum age for observations before timeout (seconds)'
    )
    
    logs_root_arg = DeclareLaunchArgument(
        'logs_root',
        default_value='./logs',
        description='Root directory for experiment logs'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug/info/warn/error)'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    )
    
    # Biped Policy Inference Node
    biped_inference_node = Node(
        package='your_package_name',  # Replace with your actual package name
        executable='biped_ros_inference.py',
        name='biped_policy_inference',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[{
            'experiment_name': LaunchConfiguration('experiment_name'),
            'checkpoint': LaunchConfiguration('checkpoint'),
            'device': LaunchConfiguration('device'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'timeout_duration': LaunchConfiguration('timeout_duration'),
            'logs_root': LaunchConfiguration('logs_root'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        # Optional topic remappings
        remappings=[
            # Add remappings here if your topics have different names
            # ('/imu/data', LaunchConfiguration('namespace') + '/imu/data'),
            # ('/joint_states', LaunchConfiguration('namespace') + '/joint_states'),
            # ('/cmd_vel', LaunchConfiguration('namespace') + '/cmd_vel'),
            # ('/motor_commands', LaunchConfiguration('namespace') + '/motor_commands'),
        ]
    )
    
    # Optional: Launch info message
    launch_info = LogInfo(
        msg=[
            'Starting Biped Policy Inference Node with:',
            '\n  Experiment: ', LaunchConfiguration('experiment_name'),
            '\n  Checkpoint: ', LaunchConfiguration('checkpoint'),
            '\n  Device: ', LaunchConfiguration('device'),
            '\n  Frequency: ', LaunchConfiguration('control_frequency'), ' Hz',
            '\n  Namespace: ', LaunchConfiguration('namespace'),
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        experiment_name_arg,
        checkpoint_arg,
        device_arg,
        control_frequency_arg,
        timeout_duration_arg,
        logs_root_arg,
        use_sim_time_arg,
        log_level_arg,
        namespace_arg,
        
        # Launch info
        launch_info,
        
        # Nodes
        biped_inference_node,
    ])


if __name__ == '__main__':
    generate_launch_description()
