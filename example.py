import genesis as gs
import pathlib
import numpy as np

class GenesisSimulation:
    """
    Handles the setup and state of the Genesis physics simulation.
    This class is not a ROS node. It purely manages simulation objects.
    """
    def __init__(self, show_viewer=False):
        """
        Initializes the Genesis scene, plane, and robot.
        """
        # --- Get URDF Path ---
        script_dir = pathlib.Path(__file__).parent.resolve()
        biped_urdf_path = script_dir / 'urdf/biped_v4.urdf'
        if not biped_urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found at: {biped_urdf_path}")
        
        # --- Initialize Genesis Simulator ---
        gs.init(backend=gs.cuda)
        self.scene = gs.Scene(show_viewer=show_viewer)
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        
        # --- Define Link Names ---
        # These are needed to correctly load the robot and attach sensors later.
        self.right_foot_link_name = "revolute_rightfoot"
        self.left_foot_link_name = "revolute_leftfoot"
        links_to_keep_list = [self.right_foot_link_name, self.left_foot_link_name]
        
        # --- Load Robot ---
        self.biped_robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(biped_urdf_path),
                fixed=False,
                links_to_keep=links_to_keep_list
            )
        )
        
        print("Genesis simulation environment initialized.")
    
    def build_scene(self):
        """
        Builds the simulation scene after all entities are added.
        """
        self.scene.build()
        print("Genesis scene built.")
        
        # Print link information to understand the structure
        print(f"Robot has {self.biped_robot.n_links} links:")
        
        # Access links directly from the links list
        self.right_foot_index = None
        self.left_foot_index = None
        
        for i, link in enumerate(self.biped_robot.links):
            print(f"  Link {i}: {link.name}")
            if link.name == self.right_foot_link_name:
                self.right_foot_index = i
                print(f"Found right foot '{self.right_foot_link_name}' at index {i}")
            elif link.name == self.left_foot_link_name:
                self.left_foot_index = i
                print(f"Found left foot '{self.left_foot_link_name}' at index {i}")
                
        if self.right_foot_index is None:
            print(f"Warning: Could not find right foot link '{self.right_foot_link_name}'")
        if self.left_foot_index is None:
            print(f"Warning: Could not find left foot link '{self.left_foot_link_name}'")
    
    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix.
        q: quaternion in [w, x, y, z] format (Genesis format)
        Returns: 3x3 rotation matrix
        """
        # Move tensor to CPU and convert to numpy if needed
        if hasattr(q, 'cpu'):
            q = q.cpu().numpy()
        
        w, x, y, z = q
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def get_axis_orientation_wrt_world_z(self, rotation_matrix, axis_index):
        """
        Get the orientation of a local axis relative to the world Z-axis.
        
        Args:
            rotation_matrix: 3x3 rotation matrix of the link
            axis_index: 0 for X-axis, 1 for Y-axis, 2 for Z-axis
            
        Returns:
            angle_degrees: Angle between the specified axis and world Z-axis in degrees
            dot_product: Dot product value (cosine of the angle)
        """
        # World Z-axis vector
        world_z = np.array([0, 0, 1])
        
        # Extract the specified axis from rotation matrix
        # Column vectors of rotation matrix represent the local axes in world coordinates
        local_axis = rotation_matrix[:, axis_index]  # X=0, Y=1, Z=2
        
        # Calculate dot product (cosine of angle between vectors)
        dot_product = np.dot(local_axis, world_z)
        
        # Clamp dot product to valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees, dot_product
    
    def step(self):
        """
        Advances the simulation by one step.
        """
        self.scene.step()
        
        # Check if we found the foot links
        if self.right_foot_index is None or self.left_foot_index is None:
            print("Cannot get foot orientations - foot links not found")
            return
            
        # Get all link positions and orientations using Genesis methods
        positions = self.biped_robot.get_links_pos()  # Returns numpy array
        quaternions = self.biped_robot.get_links_quat()  # Returns numpy array
        
        # Extract data for our target links
        right_pos = positions[self.right_foot_index]
        left_pos = positions[self.left_foot_index]
        right_quat = quaternions[self.right_foot_index]  # [w, x, y, z] format in Genesis
        left_quat = quaternions[self.left_foot_index]
        
        # Convert quaternions to rotation matrices
        right_rot_matrix = self.quaternion_to_rotation_matrix(right_quat)
        left_rot_matrix = self.quaternion_to_rotation_matrix(left_quat)
        
        # Calculate orientations relative to world Z-axis
        # Right foot X-axis vs World Z-axis
        right_x_angle, right_x_dot = self.get_axis_orientation_wrt_world_z(right_rot_matrix, 0)
        
        # Left foot Y-axis vs World Z-axis  
        left_y_angle, left_y_dot = self.get_axis_orientation_wrt_world_z(left_rot_matrix, 1)
        
        # Print the results
        print(f"\n=== Axis Orientations Relative to World Z-Axis ===")
        print(f"Right foot ({self.right_foot_link_name}) X-axis:")
        print(f"  Angle with World Z-axis: {right_x_angle:.2f} degrees")
        print(f"  Dot product (cosine): {right_x_dot:.4f}")
        # print(f"  X-axis vector in world frame: {right_rot_matrix[:, 0]}")
        
        print(f"Left foot ({self.left_foot_link_name}) Y-axis:")
        print(f"  Angle with World Z-axis: {left_y_angle:.2f} degrees")
        print(f"  Dot product (cosine): {left_y_dot:.4f}")
        # print(f"  Y-axis vector in world frame: {left_rot_matrix[:, 1]}")
        
        # Additional useful information
        # print(f"\n=== Additional Foot Information ===")
        # print(f"Right foot position: {right_pos}")
        # print(f"Left foot position: {left_pos}")
        # print(f"Right foot quaternion (w,x,y,z): {right_quat}")
        # print(f"Left foot quaternion (w,x,y,z): {left_quat}")
        
        # Example reward calculations based on axis orientations
        # For foot parallelism to ground, you might want the foot's normal (Z-axis) to align with world Z
        right_z_angle, right_z_dot = self.get_axis_orientation_wrt_world_z(right_rot_matrix, 2)
        left_z_angle, left_z_dot = self.get_axis_orientation_wrt_world_z(left_rot_matrix, 2)
        
        # print(f"\n=== Foot Parallelism to Ground (Z-axis alignment) ===")
        # print(f"Right foot Z-axis angle with World Z: {right_z_angle:.2f} degrees")
        # print(f"Left foot Z-axis angle with World Z: {left_z_angle:.2f} degrees")
        
        # Reward calculation examples
        # Higher reward when foot Z-axis is aligned with world Z (parallel to ground)
        right_parallelism_reward = abs(right_z_dot)  # Close to 1 when parallel
        left_parallelism_reward = abs(left_z_dot)
        total_parallelism_reward = (right_parallelism_reward + left_parallelism_reward) / 2.0
        # print(f"Foot parallelism reward: {total_parallelism_reward:.4f}")


def main():
    """
    Main function to run the simulation and print link orientations.
    """
    # Create simulation instance
    sim = GenesisSimulation(show_viewer=False)
    
    # Build the scene
    sim.build_scene()
    
    # Run simulation for a few steps to see the orientations
    print("Starting simulation and printing link orientations...")
    for step_num in range(100000):  # Run for 10 steps (reduced from 1000000)
        print(f"\n--- Step {step_num + 1} ---")
        sim.step()
        
        # Add a small delay to make output readable
        import time
        time.sleep(0.5)

if __name__ == "__main__":
    main()