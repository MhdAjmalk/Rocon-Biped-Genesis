# imu.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import numpy as np

class IMUSensorPublisher:
    """
    A class to represent the IMU sensor, handle its data, and publish it.
    """
    def __init__(self, parent_node: Node, robot_entity):
        """
        Initializes the IMU sensor publisher.

        Args:
            parent_node (Node): The parent ROS 2 node that will do the publishing.
            robot_entity: The Genesis robot entity from which to read data.
        """
        self.node = parent_node
        self.robot = robot_entity
        self.imu_publisher = self.node.create_publisher(Imu, 'imu', 10)
        self.node.get_logger().info("IMU sensor publisher initialized.")

    def publish_data(self, current_time):
        """
        Gathers IMU data from the simulation and publishes it.
        
        Args:
            current_time: The current simulation time from the node's clock.
        """
        quat_data = self.robot.get_quat().cpu().numpy()
        vel_data = self.robot.get_vel().cpu().numpy()
        ang_data = self.robot.get_ang().cpu().numpy()

        # --- Process Quaternion ---
        if quat_data.ndim == 1 and len(quat_data) == 4:
            orientation_quat = quat_data
        elif quat_data.ndim == 2 and quat_data.shape[0] > 0:
            orientation_quat = quat_data[0]
        else:
            self.node.get_logger().warn(f"Unexpected quaternion shape: {quat_data.shape}")
            orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # --- Process Velocities ---
        linear_velocity = vel_data[0] if vel_data.ndim == 2 and vel_data.shape[0] > 0 else vel_data
        angular_velocity = ang_data[0] if ang_data.ndim == 2 and ang_data.shape[0] > 0 else ang_data

        # --- Create and Publish IMU Message ---
        imu_msg = Imu()
        imu_msg.header = Header(stamp=current_time, frame_id='base_link')
        
        imu_msg.orientation.w = float(orientation_quat[0])
        imu_msg.orientation.x = float(orientation_quat[1])
        imu_msg.orientation.y = float(orientation_quat[2])
        imu_msg.orientation.z = float(orientation_quat[3])
        
        imu_msg.angular_velocity.x = float(angular_velocity[0])
        imu_msg.angular_velocity.y = float(angular_velocity[1])
        imu_msg.angular_velocity.z = float(angular_velocity[2])
        
        # Note: Using linear velocity as linear acceleration for this example
        imu_msg.linear_acceleration.x = float(linear_velocity[0])
        imu_msg.linear_acceleration.y = float(linear_velocity[1])
        imu_msg.linear_acceleration.z = float(linear_velocity[2])
        
        self.imu_publisher.publish(imu_msg)
