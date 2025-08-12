#!/usr/bin/env python3
"""
ROS2 Biped Policy Inference Node

This node integrates the trained biped policy with ROS2 Humble by subscribing to
various standard ROS2 message topics for observations and publishing motor commands.

Subscribed Topics:
    /imu/data (sensor_msgs/Imu): IMU data for angular velocity and orientation
    /joint_states (sensor_msgs/JointState): Current joint positions and velocities
    /cmd_vel (geometry_msgs/Twist): Velocity commands
    /robot_state/base_pose (geometry_msgs/PoseWithCovarianceStamped): Base position/orientation
    /robot_state/base_velocity (geometry_msgs/TwistStamped): Base linear/angular velocity
    /contact_sensors/left_foot (std_msgs/Bool): Left foot contact state
    /contact_sensors/right_foot (std_msgs/Bool): Right foot contact state
    /gravity_vector (geometry_msgs/Vector3Stamped): Projected gravity vector

Published Topics:
    /motor_commands (sensor_msgs/JointState): Motor position commands
    /policy_diagnostics (std_msgs/String): Policy status and diagnostics

Parameters:
    experiment_name: Name of the trained experiment
    checkpoint: Model checkpoint to load
    device: Inference device ("cuda" or "cpu")
    control_frequency: Control loop frequency in Hz
    timeout_duration: Maximum time to wait for observations before timeout
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import torch
import json
import threading
from typing import Dict, Any, Optional, List
import time

# ROS2 message types
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, TwistStamped, Vector3Stamped
from std_msgs.msg import Bool, String, Float64MultiArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Import our custom inference class
from biped_policy_inference import BipedPolicyInference


class BipedROS2InferenceNode(Node):
    """
    ROS2 node that runs biped policy inference using observations from various topics.
    """
    
    def __init__(self):
        super().__init__('biped_policy_inference')
        
        # Declare parameters
        self.declare_parameter('experiment_name', 'biped-walking')
        self.declare_parameter('checkpoint', 100)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('timeout_duration', 1.0)
        self.declare_parameter('logs_root', './logs')
        self.declare_parameter('use_sim_time', False)
        
        # Get parameters
        self.experiment_name = self.get_parameter('experiment_name').value
        self.checkpoint = self.get_parameter('checkpoint').value
        self.device = self.get_parameter('device').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.timeout_duration = self.get_parameter('timeout_duration').value
        self.logs_root = self.get_parameter('logs_root').value
        
        # Joint names (must match your URDF and training configuration)
        self.joint_names = [
            "right_hip1", "right_hip2", "right_knee", "right_ankle",
            "left_hip1", "left_hip2", "left_knee", "left_ankle",
            "revolute_torso"
        ]
        
        # Initialize observation buffer
        self.obs_buffer = {
            'imu_data': None,
            'joint_states': None,
            'cmd_vel': None,
            'base_pose': None,
            'base_velocity': None,
            'left_foot_contact': None,
            'right_foot_contact': None,
            'gravity_vector': None,
            'last_actions': np.zeros(9)  # Previous actions
        }
        
        # Timestamps for timeout detection
        self.obs_timestamps = {key: None for key in self.obs_buffer.keys() if key != 'last_actions'}
        
        # Thread safety
        self.obs_lock = threading.Lock()
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.subscribers = {}
        self.subscribers['imu'] = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, sensor_qos)
        self.subscribers['joint_states'] = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, sensor_qos)
        self.subscribers['cmd_vel'] = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, reliable_qos)
        self.subscribers['base_pose'] = self.create_subscription(
            PoseWithCovarianceStamped, '/robot_state/base_pose', self.base_pose_callback, sensor_qos)
        self.subscribers['base_velocity'] = self.create_subscription(
            TwistStamped, '/robot_state/base_velocity', self.base_velocity_callback, sensor_qos)
        self.subscribers['left_foot_contact'] = self.create_subscription(
            Bool, '/contact_sensors/left_foot', self.left_foot_contact_callback, sensor_qos)
        self.subscribers['right_foot_contact'] = self.create_subscription(
            Bool, '/contact_sensors/right_foot', self.right_foot_contact_callback, sensor_qos)
        self.subscribers['gravity_vector'] = self.create_subscription(
            Vector3Stamped, '/gravity_vector', self.gravity_vector_callback, sensor_qos)
        
        # Publishers
        self.motor_cmd_pub = self.create_publisher(JointState, '/motor_commands', reliable_qos)
        self.diagnostics_pub = self.create_publisher(String, '/policy_diagnostics', reliable_qos)
        self.observation_pub = self.create_publisher(Float64MultiArray, '/policy_observations', reliable_qos)
        
        # Initialize policy
        try:
            self.policy_inference = BipedPolicyInference(
                experiment_name=self.experiment_name,
                checkpoint=self.checkpoint,
                device=self.device,
                logs_root=self.logs_root
            )
            self.get_logger().info(f"Policy loaded successfully!")
            self.publish_diagnostics("READY", "Policy loaded and ready for inference")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            self.publish_diagnostics("ERROR", f"Failed to load policy: {e}")
            return
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop)
        
        # Statistics
        self.inference_count = 0
        self.last_inference_time = time.time()
        self.inference_times = []
        
        self.get_logger().info(f"Biped ROS2 Inference Node initialized")
        self.get_logger().info(f"Control frequency: {self.control_frequency} Hz")
        self.get_logger().info(f"Experiment: {self.experiment_name}, Checkpoint: {self.checkpoint}")
    
    def imu_callback(self, msg: Imu):
        """Handle IMU data (angular velocity and orientation)."""
        with self.obs_lock:
            self.obs_buffer['imu_data'] = msg
            self.obs_timestamps['imu_data'] = self.get_clock().now()
    
    def joint_states_callback(self, msg: JointState):
        """Handle joint state data (positions and velocities)."""
        with self.obs_lock:
            self.obs_buffer['joint_states'] = msg
            self.obs_timestamps['joint_states'] = self.get_clock().now()
    
    def cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands."""
        with self.obs_lock:
            self.obs_buffer['cmd_vel'] = msg
            self.obs_timestamps['cmd_vel'] = self.get_clock().now()
    
    def base_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Handle base pose data."""
        with self.obs_lock:
            self.obs_buffer['base_pose'] = msg
            self.obs_timestamps['base_pose'] = self.get_clock().now()
    
    def base_velocity_callback(self, msg: TwistStamped):
        """Handle base velocity data."""
        with self.obs_lock:
            self.obs_buffer['base_velocity'] = msg
            self.obs_timestamps['base_velocity'] = self.get_clock().now()
    
    def left_foot_contact_callback(self, msg: Bool):
        """Handle left foot contact data."""
        with self.obs_lock:
            self.obs_buffer['left_foot_contact'] = msg
            self.obs_timestamps['left_foot_contact'] = self.get_clock().now()
    
    def right_foot_contact_callback(self, msg: Bool):
        """Handle right foot contact data."""
        with self.obs_lock:
            self.obs_buffer['right_foot_contact'] = msg
            self.obs_timestamps['right_foot_contact'] = self.get_clock().now()
    
    def gravity_vector_callback(self, msg: Vector3Stamped):
        """Handle gravity vector data."""
        with self.obs_lock:
            self.obs_buffer['gravity_vector'] = msg
            self.obs_timestamps['gravity_vector'] = self.get_clock().now()
    
    def check_data_freshness(self) -> bool:
        """Check if all required data is fresh enough."""
        current_time = self.get_clock().now()
        timeout_ns = int(self.timeout_duration * 1e9)
        
        required_topics = ['imu_data', 'joint_states', 'cmd_vel']
        
        for topic in required_topics:
            if self.obs_timestamps[topic] is None:
                return False
            
            age = (current_time - self.obs_timestamps[topic]).nanoseconds
            if age > timeout_ns:
                return False
        
        return True
    
    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def build_observation_vector(self) -> Optional[np.ndarray]:
        """
        Build the observation vector from ROS messages.
        
        Observation structure (38 dimensions total):
        - Base linear velocity (3): [vx, vy, vz]
        - Base angular velocity (3): [wx, wy, wz] 
        - Projected gravity (3): [gx, gy, gz]
        - Commands (3): [cmd_vx, cmd_vy, cmd_wz]
        - Joint positions (9): [joint_pos_0, ..., joint_pos_8]
        - Joint velocities (9): [joint_vel_0, ..., joint_vel_8]
        - Last actions (9): [last_action_0, ..., last_action_8]
        - Foot contacts (2): [left_contact, right_contact]
        Total: 3+3+3+3+9+9+9+2 = 41 -> but we use 38 based on your config
        """
        
        with self.obs_lock:
            obs_data = self.obs_buffer.copy()
        
        # Check if we have the minimum required data
        if not self.check_data_freshness():
            return None
        
        try:
            observation = []
            
            # Base linear velocity (3) - from base_velocity or default
            if obs_data['base_velocity'] is not None:
                twist = obs_data['base_velocity'].twist
                observation.extend([twist.linear.x, twist.linear.y, twist.linear.z])
            else:
                observation.extend([0.0, 0.0, 0.0])
            
            # Base angular velocity (3) - from IMU
            if obs_data['imu_data'] is not None:
                angular_vel = obs_data['imu_data'].angular_velocity
                observation.extend([angular_vel.x, angular_vel.y, angular_vel.z])
            else:
                observation.extend([0.0, 0.0, 0.0])
            
            # Projected gravity (3) - from gravity_vector or IMU orientation
            if obs_data['gravity_vector'] is not None:
                gravity = obs_data['gravity_vector'].vector
                observation.extend([gravity.x, gravity.y, gravity.z])
            elif obs_data['imu_data'] is not None:
                # Calculate projected gravity from orientation
                quat = obs_data['imu_data'].orientation
                roll, pitch, yaw = self.quaternion_to_euler(quat)
                # Project global gravity [0, 0, -1] into body frame
                gravity_body = [
                    -np.sin(pitch),
                    np.sin(roll) * np.cos(pitch), 
                    -np.cos(roll) * np.cos(pitch)
                ]
                observation.extend(gravity_body)
            else:
                observation.extend([0.0, 0.0, -1.0])  # Default gravity
            
            # Commands (3) - from cmd_vel
            if obs_data['cmd_vel'] is not None:
                cmd = obs_data['cmd_vel']
                observation.extend([cmd.linear.x, cmd.linear.y, cmd.angular.z])
            else:
                observation.extend([0.0, 0.0, 0.0])
            
            # Joint positions (9) and velocities (9)
            if obs_data['joint_states'] is not None:
                joint_msg = obs_data['joint_states']
                
                # Initialize arrays
                joint_positions = np.zeros(9)
                joint_velocities = np.zeros(9)
                
                # Map joint states to our expected order
                for i, joint_name in enumerate(self.joint_names):
                    try:
                        idx = joint_msg.name.index(joint_name)
                        if idx < len(joint_msg.position):
                            joint_positions[i] = joint_msg.position[idx]
                        if idx < len(joint_msg.velocity):
                            joint_velocities[i] = joint_msg.velocity[idx]
                    except ValueError:
                        # Joint not found, use default value
                        pass
                
                observation.extend(joint_positions.tolist())
                observation.extend(joint_velocities.tolist())
            else:
                # Default joint states
                observation.extend([0.0] * 18)  # 9 positions + 9 velocities
            
            # Last actions (9)
            observation.extend(obs_data['last_actions'].tolist())
            
            # Foot contacts (2)
            left_contact = 1.0 if (obs_data['left_foot_contact'] and obs_data['left_foot_contact'].data) else 0.0
            right_contact = 1.0 if (obs_data['right_foot_contact'] and obs_data['right_foot_contact'].data) else 0.0
            observation.extend([left_contact, right_contact])
            
            # Convert to numpy array and check size
            obs_array = np.array(observation, dtype=np.float32)
            
            if len(obs_array) != 38:
                self.get_logger().warning(f"Observation size mismatch: expected 38, got {len(obs_array)}")
                # Pad or trim to expected size
                if len(obs_array) < 38:
                    obs_array = np.pad(obs_array, (0, 38 - len(obs_array)))
                else:
                    obs_array = obs_array[:38]
            
            return obs_array
            
        except Exception as e:
            self.get_logger().error(f"Error building observation vector: {e}")
            return None
    
    def control_loop(self):
        """Main control loop called at specified frequency."""
        start_time = time.time()
        
        # Build observation vector
        observation = self.build_observation_vector()
        
        if observation is None:
            self.publish_diagnostics("WARNING", "Insufficient or stale observation data")
            return
        
        try:
            # Run inference
            actions = self.policy_inference.get_actions_numpy(observation)
            
            # Update last actions for next iteration
            with self.obs_lock:
                self.obs_buffer['last_actions'] = actions.copy()
            
            # Publish motor commands
            self.publish_motor_commands(actions)
            
            # Publish observation vector for debugging
            self.publish_observations(observation)
            
            # Update statistics
            self.inference_count += 1
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last 100 inference times for averaging
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            # Publish periodic diagnostics
            if self.inference_count % 100 == 0:
                avg_time = np.mean(self.inference_times) * 1000  # Convert to ms
                self.publish_diagnostics(
                    "RUNNING", 
                    f"Inference #{self.inference_count}, Avg time: {avg_time:.2f}ms"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")
            self.publish_diagnostics("ERROR", f"Inference error: {e}")
    
    def publish_motor_commands(self, actions: np.ndarray):
        """Publish motor commands as JointState message."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = actions.tolist()
        
        self.motor_cmd_pub.publish(msg)
    
    def publish_observations(self, observations: np.ndarray):
        """Publish observation vector for debugging."""
        msg = Float64MultiArray()
        msg.data = observations.tolist()
        self.observation_pub.publish(msg)
    
    def publish_diagnostics(self, status: str, message: str):
        """Publish diagnostics information."""
        diag_msg = String()
        diag_data = {
            "timestamp": self.get_clock().now().to_msg(),
            "status": status,
            "message": message,
            "inference_count": self.inference_count,
            "experiment": self.experiment_name,
            "checkpoint": self.checkpoint
        }
        diag_msg.data = json.dumps(diag_data)
        self.diagnostics_pub.publish(diag_msg)
        
        # Also log to console
        if status == "ERROR":
            self.get_logger().error(message)
        elif status == "WARNING":
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    """Main function to run the ROS2 node."""
    rclpy.init(args=args)
    
    try:
        node = BipedROS2InferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()