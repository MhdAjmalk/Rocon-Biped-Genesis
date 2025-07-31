# genesislaunch.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np

# Import the modular components
from genesis_world import GenesisSimulation
from imu import IMUSensorPublisher
from force_contact import ForceContactSensorPublisher

class GenesisPublisherNode(Node):
    def __init__(self):
        super().__init__('genesis_publisher_node')

        # --- 1. Initialize the Simulation Environment ---
        self.sim = GenesisSimulation(show_viewer=True)
        
        # --- 2. Instantiate Sensor Publishers ---
        # Pass the current node (self) and the robot object to the sensor classes
        self.imu_sensor = IMUSensorPublisher(self, self.sim.biped_robot)
        self.right_foot_sensor = ForceContactSensorPublisher(self, self.sim.biped_robot, self.sim.right_foot_link_name)
        self.left_foot_sensor = ForceContactSensorPublisher(self, self.sim.biped_robot, self.sim.left_foot_link_name)

        # --- 3. Build the Genesis Scene ---
        # This must be done after all sensors are attached.
        self.sim.build_scene()
        self.get_logger().info('Genesis simulation with modular sensors is fully initialized.')

        # --- 4. Create other ROS 2 Publishers ---
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # --- 5. Set up the main timer ---
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # --- Get Joint Names for JointState messages ---
        num_dofs = len(self.sim.biped_robot.get_dofs_position().cpu().numpy())
        self.joint_names = [f'joint_{i}' for i in range(num_dofs)]

    def timer_callback(self):
        """
        Main loop for the simulation and data publishing.
        """
        try:
            # Step the simulation forward
            self.sim.step()
            current_time = self.get_clock().now().to_msg()
            
            # Publish data using the dedicated sensor objects
            self.imu_sensor.publish_data(current_time)
            self.right_foot_sensor.publish_data()
            self.left_foot_sensor.publish_data()

            # Publish JointState Message directly from this node
            self.publish_joint_states(current_time)
                    
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")

    def publish_joint_states(self, current_time):
        """Gets and publishes the robot's joint states."""
        joint_positions = self.sim.biped_robot.get_dofs_position().cpu().numpy()
        joint_velocities = self.sim.biped_robot.get_dofs_velocity().cpu().numpy()

        joint_state_msg = JointState()
        joint_state_msg.header = Header(stamp=current_time)
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = joint_positions.tolist()
        joint_state_msg.velocity = joint_velocities.tolist()
        self.joint_state_publisher.publish(joint_state_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        genesis_publisher_node = GenesisPublisherNode()
        print("Genesis ROS 2 node with modular sensors running. Spinning...")
        rclpy.spin(genesis_publisher_node)
    except Exception as e:
        print(f"An error occurred during node execution: {e}")
    finally:
        # Cleanup
        if 'genesis_publisher_node' in locals() and rclpy.ok():
            genesis_publisher_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node shutdown complete.")

if __name__ == '__main__':
    main()
