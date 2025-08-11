# Biped ROS2 Inference Package

This package provides ROS2 integration for running trained biped robot policies in real-time using observations from various ROS2 topics.

## Features

- **Real-time Inference**: Runs trained PPO policies at configurable frequencies (default 50Hz)
- **Multiple Input Sources**: Subscribes to standard ROS2 message types for observations
- **Flexible Topic Mapping**: Easy topic remapping for different robot configurations
- **Safety Features**: Timeout detection, data validation, and emergency stops
- **Diagnostics**: Built-in monitoring and debugging capabilities
- **Device Support**: Both CPU and GPU inference

## Architecture

### Input Topics (Subscribed)

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/imu/data` | `sensor_msgs/Imu` | IMU data for angular velocity and orientation |
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions and velocities |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands from higher-level planner |
| `/robot_state/base_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Base position and orientation |
| `/robot_state/base_velocity` | `geometry_msgs/TwistStamped` | Base linear and angular velocity |
| `/contact_sensors/left_foot` | `std_msgs/Bool` | Left foot contact state |
| `/contact_sensors/right_foot` | `std_msgs/Bool` | Right foot contact state |
| `/gravity_vector` | `geometry_msgs/Vector3Stamped` | Projected gravity vector |

### Output Topics (Published)

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/motor_commands` | `sensor_msgs/JointState` | Motor position commands |
| `/policy_diagnostics` | `std_msgs/String` | Policy status and diagnostics (JSON) |
| `/policy_observations` | `std_msgs/Float64MultiArray` | Raw observation vector (debugging) |

### Observation Vector Structure (38 dimensions)

The node constructs a 38-dimensional observation vector from the input topics:

1. **Base Linear Velocity (3)**: [vx, vy, vz] from base_velocity topic
2. **Base Angular Velocity (3)**: [wx, wy, wz] from IMU data
3. **Projected Gravity (3)**: [gx, gy, gz] from gravity_vector or calculated from IMU
4. **Commands (3)**: [cmd_vx, cmd_vy, cmd_wz] from cmd_vel topic
5. **Joint Positions (9)**: Current joint positions from joint_states
6. **Joint Velocities (9)**: Current joint velocities from joint_states  
7. **Last Actions (9)**: Previous motor commands for temporal consistency
8. **Foot Contacts (2)**: [left_contact, right_contact] from contact sensors

## Installation

### Prerequisites

- ROS2 Humble
- Python 3.8+
- PyTorch
- rsl-rl-lib==2.2.4

### Install Dependencies

```bash
# Install ROS2 dependencies
sudo apt update
sudo apt install ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-tf2-ros

# Install Python dependencies
pip install torch numpy rsl-rl-lib==2.2.4
```

### Build the Package

```bash
# Navigate to your ROS2 workspace
cd ~/your_ros2_workspace/src

# Clone or copy this package
cp -r /path/to/biped_inference .

# Build the workspace
cd ~/your_ros2_workspace
colcon build --packages-select biped_inference

# Source the workspace
source install/setup.bash
```

## Usage

### 1. Basic Usage

```bash
# Launch with default parameters
ros2 launch biped_inference biped_inference.launch.py

# Launch with custom experiment and checkpoint
ros2 launch biped_inference biped_inference.launch.py \
    experiment_name:=my-experiment \
    checkpoint:=200

# Launch with CPU inference
ros2 launch biped_inference biped_inference.launch.py device:=cpu
```

### 2. Using Configuration Files

```bash
# Launch with parameter file
ros2 launch biped_inference biped_inference.launch.py \
    params_file:=./config/biped_inference_params.yaml
```

### 3. Topic Remapping

If your robot publishes topics with different names, you can remap them:

```bash
ros2 launch biped_inference biped_inference.launch.py \
    imu_topic:=/robot/sensors/imu \
    joint_states_topic:=/robot/joint_states \
    cmd_vel_topic:=/navigation/cmd_vel
```

### 4. Running the Node Directly

```bash
# Run the node directly (useful for debugging)
ros2 run biped_inference biped_ros_inference \
    --ros-args \
    -p experiment_name:=biped-walking \
    -p checkpoint:=100 \
    -p device:=cuda \
    -p control_frequency:=50.0
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | string | "biped-walking" | Name of trained experiment |
| `checkpoint` | int | 100 | Model checkpoint to load |
| `device` | string | "cuda" | Inference device (cuda/cpu) |
| `control_frequency` | float | 50.0 | Control loop frequency (Hz) |
| `timeout_duration` | float | 1.0 | Max age for observations (seconds) |
| `logs_root` | string | "./logs" | Path to experiment logs |
| `use_sim_time` | bool | false | Use simulation time |

## Monitoring and Debugging

### View Diagnostics

```bash
# Monitor policy status
ros2 topic echo /policy_diagnostics

# View raw observation vector
ros2 topic echo /policy_observations

# Monitor motor commands
ros2 topic echo /motor_commands
```

### Check Node Status

```bash
# List active topics
ros2 topic list

# Check node info
ros2 node info /biped_policy_inference

# Monitor computational performance
ros2 topic hz /motor_commands
```

### Logging

The node provides detailed logging at different levels:

```bash
# Run with debug logging
ros2 run biped_inference biped_ros_inference --ros-args --log-level DEBUG

# View logs
ros2 log view biped_policy_inference
```

## Integration Examples

### With Navigation Stack

```bash
# Terminal 1: Navigation
ros2 launch nav2_bringup navigation_launch.py

# Terminal 2: Biped Inference
ros2 launch biped_inference biped_inference.launch.py \
    cmd_vel_topic:=/cmd_vel

# Terminal 3: Send navigation goals
ros2 run nav2_simple_commander example_nav_to_pose.py
```

### With MoveIt2

```bash
# Terminal 1: MoveIt2
ros2 launch your_robot_moveit_config demo.launch.py

# Terminal 2: Biped Inference  
ros2 launch biped_inference biped_inference.launch.py \
    joint_states_topic:=/joint_states
```

## Troubleshooting

### Common Issues

1. **"Policy failed to load"**
   - Check experiment name and checkpoint exist in logs directory
   - Verify logs_root parameter points to correct directory

2. **"Insufficient observation data"**
   - Check all required topics are being published
   - Verify topic names match your robot configuration
   - Use `ros2 topic list` to see available topics

3. **"High inference latency"**
   - Switch to CPU if GPU memory is insufficient
   - Reduce control frequency
   - Check system resources

4. **"Joint commands not working"**
   - Verify joint names match your robot configuration
   - Check motor controller is subscribed to `/motor_commands`
   - Ensure proper units (positions in radians)

### Debug Commands

```bash
# Check topic frequencies
ros2 topic hz /imu/data
ros2 topic hz /joint_states

# Monitor observation freshness
ros2 topic echo /policy_diagnostics | grep timestamp

# Test inference standalone
python3 biped_policy_inference.py
```

## Safety Considerations

- The node includes timeout detection for stale sensor data
- Motor commands are only published when fresh observations are available
- Emergency stop functionality can be triggered via ROS2 services
- Joint velocity and acceleration limits can be configured

## Advanced Usage

### Custom Observation Processing

You can modify the `build_observation_vector()` method to customize how observations are constructed from ROS topics.

### Multi-Robot Support

Use namespaces for multiple robots:

```bash
ros2 launch biped_inference biped_inference.launch.py namespace:=robot1
ros2 launch biped_inference biped_inference.launch.py namespace:=robot2
```

### Integration with Simulators

The package works with Gazebo, Isaac Sim, and other simulators:

```bash
# For Gazebo
ros2 launch biped_inference biped_inference.launch.py use_sim_time:=true

# For Isaac Sim
ros2 launch biped_inference biped_inference.launch.py \
    use_sim_time:=true \
    device:=cuda
```

## Contributing

Feel free to submit issues, feature requests, and pull requests to improve this package.

## License

MIT License - see LICENSE file for details.
