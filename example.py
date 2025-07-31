import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Header, Float32
import genesis as gs
from genesis.sensors import RigidContactForceGridSensor
import pathlib
import numpy as np

class GenesisPublisher(Node):
    def __init__(self):
        super().__init__('genesis_publisher_node')

        # --- Initialize Genesis Simulator ---
        script_dir = pathlib.Path(__file__).parent.resolve()
        biped_urdf_path = script_dir / 'urdf/biped_v4.urdf'

        gs.init(backend=gs.cuda)
        self.scene = gs.Scene(show_viewer=True)
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Define the two link names separately
        self.right_foot_link_name = "revolute_rightfoot"
        self.left_foot_link_name = "revolute_leftfoot"

        # --- Load Robot ---
        links_to_keep_list = [self.right_foot_link_name, self.left_foot_link_name]

        self.biped_robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(biped_urdf_path),
                fixed=False,
                links_to_keep=links_to_keep_list
            )
        )

        # --- Create Two Separate Sensors ---
        self.right_foot_sensor = None
        self.left_foot_sensor = None

        for link in self.biped_robot.links:
            if link.name == self.right_foot_link_name:
                self.right_foot_sensor = RigidContactForceGridSensor(
                    entity=self.biped_robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )
                self.get_logger().info(f"Attached grid contact sensor to '{self.right_foot_link_name}'")
            elif link.name == self.left_foot_link_name:
                self.left_foot_sensor = RigidContactForceGridSensor(
                    entity=self.biped_robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )
                self.get_logger().info(f"Attached grid contact sensor to '{self.left_foot_link_name}'")

        self.scene.build()
        self.get_logger().info('Genesis simulation with grid sensors initialized.')

        # --- Create ROS 2 Publishers ---
        self.imu_publisher = self.create_publisher(Imu, 'imu', 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        self.right_contact_publisher = self.create_publisher(Float32, f'contact/{self.right_foot_link_name}', 10)
        self.left_contact_publisher = self.create_publisher(Float32, f'contact/{self.left_foot_link_name}', 10)
        
        # --- Set up a timer ---
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # --- Get Joint Names ---
        num_dofs = len(self.biped_robot.get_dofs_position().cpu().numpy())
        self.joint_names = [f'joint_{i}' for i in range(num_dofs)]

    def timer_callback(self):
        try:
            self.scene.step()
            current_time = self.get_clock().now().to_msg()
            
            # --- Get robot data with proper error handling ---
            # Get quaternion - might need different indexing
            quat_data = self.biped_robot.get_quat().cpu().numpy()
            self.get_logger().debug(f"Quat shape: {quat_data.shape}, data: {quat_data}")
            
            # Handle different possible shapes
            if quat_data.ndim == 1 and len(quat_data) == 4:
                orientation_quat = quat_data
            elif quat_data.ndim == 2 and quat_data.shape[0] > 0:
                orientation_quat = quat_data[0]
            else:
                self.get_logger().warn(f"Unexpected quaternion shape: {quat_data.shape}")
                orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Default identity quaternion
            
            # Get velocities
            vel_data = self.biped_robot.get_vel().cpu().numpy()
            if vel_data.ndim == 2 and vel_data.shape[0] > 0:
                linear_velocity = vel_data[0]
            elif vel_data.ndim == 1:
                linear_velocity = vel_data
            else:
                linear_velocity = np.array([0.0, 0.0, 0.0])
                
            ang_data = self.biped_robot.get_ang().cpu().numpy()
            if ang_data.ndim == 2 and ang_data.shape[0] > 0:
                angular_velocity = ang_data[0]
            elif ang_data.ndim == 1:
                angular_velocity = ang_data
            else:
                angular_velocity = np.array([0.0, 0.0, 0.0])

            # Get joint data
            joint_positions = self.biped_robot.get_dofs_position().cpu().numpy()
            joint_velocities = self.biped_robot.get_dofs_velocity().cpu().numpy()

            # Create and Publish IMU Message
            imu_msg = Imu()
            imu_msg.header = Header(stamp=current_time, frame_id='base_link')
            
            # Ensure we have 4 elements for quaternion (w, x, y, z)
            if len(orientation_quat) == 4:
                imu_msg.orientation.w = float(orientation_quat[0])
                imu_msg.orientation.x = float(orientation_quat[1])
                imu_msg.orientation.y = float(orientation_quat[2])
                imu_msg.orientation.z = float(orientation_quat[3])
            else:
                # Default identity quaternion
                imu_msg.orientation.w = 1.0
                imu_msg.orientation.x = 0.0
                imu_msg.orientation.y = 0.0
                imu_msg.orientation.z = 0.0
            
            # Set angular velocity
            if len(angular_velocity) >= 3:
                imu_msg.angular_velocity.x = float(angular_velocity[0])
                imu_msg.angular_velocity.y = float(angular_velocity[1])
                imu_msg.angular_velocity.z = float(angular_velocity[2])
            
            # Note: Using linear velocity as acceleration for now
            # You might want to compute actual acceleration from velocity differences
            if len(linear_velocity) >= 3:
                imu_msg.linear_acceleration.x = float(linear_velocity[0])
                imu_msg.linear_acceleration.y = float(linear_velocity[1])
                imu_msg.linear_acceleration.z = float(linear_velocity[2])
            
            self.imu_publisher.publish(imu_msg)

            # Create and Publish JointState Message
            joint_state_msg = JointState()
            joint_state_msg.header = Header(stamp=current_time)
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = joint_positions.tolist()
            joint_state_msg.velocity = joint_velocities.tolist()
            self.joint_state_publisher.publish(joint_state_msg)
            
            # --- Read from each sensor and publish to its topic ---
            if self.right_foot_sensor:
                try:
                    grid_forces = self.right_foot_sensor.read()
                    max_force = np.max(np.linalg.norm(grid_forces, axis=-1))
                    msg = Float32(data=float(max_force))
                    self.right_contact_publisher.publish(msg)
                except Exception as e:
                    self.get_logger().warn(f"Error reading right foot sensor: {e}")

            if self.left_foot_sensor:
                try:
                    grid_forces = self.left_foot_sensor.read()
                    max_force = np.max(np.linalg.norm(grid_forces, axis=-1))
                    msg = Float32(data=float(max_force))
                    self.left_contact_publisher.publish(msg)
                except Exception as e:
                    self.get_logger().warn(f"Error reading left foot sensor: {e}")
                    
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    genesis_publisher = GenesisPublisher()
    print("Genesis ROS 2 node with all sensors running. Spinning...")
    rclpy.spin(genesis_publisher)
    genesis_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()