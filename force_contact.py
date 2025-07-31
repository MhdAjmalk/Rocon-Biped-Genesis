# force_contact.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import numpy as np
from genesis.sensors import RigidContactForceGridSensor

class ForceContactSensorPublisher:
    """
    A class to represent a single force contact grid sensor and its publisher.
    """
    def __init__(self, parent_node: Node, robot_entity, link_name: str):
        """
        Initializes the Force Contact sensor.

        Args:
            parent_node (Node): The parent ROS 2 node.
            robot_entity: The Genesis robot entity.
            link_name (str): The name of the link to attach the sensor to.
        """
        self.node = parent_node
        self.sensor = None
        self.publisher = None
        self.link_name = link_name

        # Find the link and attach the sensor
        for link in robot_entity.links:
            if link.name == self.link_name:
                self.sensor = RigidContactForceGridSensor(
                    entity=robot_entity, link_idx=link.idx, grid_size=(2, 2, 2)
                )
                self.publisher = self.node.create_publisher(Float32, f'contact/{self.link_name}', 10)
                self.node.get_logger().info(f"Attached grid contact sensor to '{self.link_name}'")
                return # Exit after finding the link
        
        # If the loop completes without finding the link
        self.node.get_logger().error(f"Could not find link named '{self.link_name}' to attach contact sensor.")

    def publish_data(self):
        """
        Reads the contact force data and publishes the maximum force.
        """
        if self.sensor and self.publisher:
            try:
                grid_forces = self.sensor.read()
                # Calculate the norm for each grid point force vector and find the max
                max_force = np.max(np.linalg.norm(grid_forces, axis=-1))
                msg = Float32(data=float(max_force))
                self.publisher.publish(msg)
            except Exception as e:
                self.node.get_logger().warn(f"Error reading contact sensor for '{self.link_name}': {e}")
