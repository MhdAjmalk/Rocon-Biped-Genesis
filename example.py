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
        Advances the simulation by one step and demonstrates solver access methods.
        """
        self.scene.step()
        
        print("\n=== Direct Solver Access - Complete Working Example ===")
        
        try:
            # Access the solver's state fields directly
            solver = self.scene.rigid_solver
            
            # Get number of DOFs and links
            n_dofs = solver.n_dofs
            n_links = solver.n_links
            
            print(f"Number of DOFs: {n_dofs}")
            print(f"Number of links: {n_links}")
            
            # ===== METHOD 1: Using Solver Getter Methods (RECOMMENDED) =====
            print("\n--- Method 1: Using Solver Getter Methods (Recommended) ---")
            
            # Get DOF states using solver methods
            dof_positions = solver.get_dofs_position()
            dof_velocities = solver.get_dofs_velocity()
            dof_forces = solver.get_dofs_force()
            
            print("DOF States:")
            for i in range(min(5, n_dofs)):
                print(f"  DOF {i}: pos={dof_positions[i]:.6f}, vel={dof_velocities[i]:.6f}, force={dof_forces[i]:.6f}")
            
            # Get link states using solver methods
            link_positions = solver.get_links_pos()
            link_quaternions = solver.get_links_quat()
            link_masses = solver.get_links_inertial_mass()
            
            print("Link States:")
            for i in range(min(5, n_links)):
                pos = link_positions[i]
                quat = link_quaternions[i]
                mass = link_masses[i]
                print(f"  Link {i}: pos=[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}], mass={mass:.6f}")
            
            # ===== METHOD 2: Direct Taichi Field Access =====
            print("\n--- Method 2: Direct Taichi Field Access ---")
            
            # Access state arrays (these are Taichi fields)
            # Note: Be careful with indexing - these are internal solver arrays
            
            # Access DOF states directly from Taichi fields
            dofs_state = solver.dofs_state
            print("DOF States (Direct Taichi Access):")
            
            # For Taichi fields, we need to access individual elements
            for dof_idx in range(min(n_dofs, 5)):
                # Access Taichi field values - note the [None] indexing for scalar fields
                dof_pos = dofs_state.pos[dof_idx, 0]  # Shape is (n_dofs, 1)
                dof_vel = dofs_state.vel[dof_idx, 0]
                dof_force = dofs_state.force[dof_idx, 0]
                print(f"  DOF {dof_idx}: pos={dof_pos:.6f}, vel={dof_vel:.6f}, force={dof_force:.6f}")
                # print("Difference between 0 & 1")
                # dof_pos1 = dofs_state.pos[dof_idx, 0]  # Shape is (n_dofs, 1)
                # dof_vel1 = dofs_state.vel[dof_idx, 0]
                # dof_force1 = dofs_state.force[dof_idx, 0]
                # print(f"  DOF {dof_idx} (repeat): pos={dof_pos1:.6f}, vel={dof_vel1:.6f}, force={dof_force1:.6f}")
            
            # Access link states directly from Taichi fields
            links_state = solver.links_state
            print("Link States (Direct Taichi Access):")
            
            # for link_idx in range(min(n_links, 5)):
            #     # Access Taichi matrix field values - note the [None] indexing
            #     link_pos = links_state.pos[link_idx, 0]  # Returns a 3D vector
            #     link_quat = links_state.quat[link_idx, 0]  # Returns a 4D quaternion
            #     link_mass = links_state.cinr_mass[link_idx, 0]  # Total mass
                
            #     print(f"  Link {link_idx}: pos=[{link_pos[0]:.6f}, {link_pos[1]:.6f}, {link_pos[2]:.6f}], mass={link_mass:.6f}")
            #     print(f"             quat=[{link_quat[0]:.6f}, {link_quat[1]:.6f}, {link_quat[2]:.6f}, {link_quat[3]:.6f}]")
            
            # ===== METHOD 3: Entity-level Access (Original) =====
            print("\n--- Method 3: Entity-level Access (Robot-specific) ---")
            
            # This accesses only the robot entity's DOFs/links (excludes ground plane)
            robot_positions = self.biped_robot.get_links_pos()
            robot_quaternions = self.biped_robot.get_links_quat()
            robot_dof_positions = self.biped_robot.get_dofs_position()
            
            print(f"Robot Entity - Links: {robot_positions.shape[0]}, DOFs: {robot_dof_positions.shape[0]}")
            print(f"Robot Entity - First link position: {robot_positions[0]}")
            print(f"Robot Entity - First DOF position: {robot_dof_positions[0]:.6f}")
            
            # ===== COMPARISON AND VERIFICATION =====
            print("\n--- Verification: Comparing Methods ---")
            
            # Compare solver method vs entity method for DOF 0
            solver_dof_0 = dof_positions[0].item()
            entity_dof_0 = robot_dof_positions[0].item() 
            print(f"DOF 0 - Solver method: {solver_dof_0:.6f}")
            print(f"DOF 0 - Entity method: {entity_dof_0:.6f}")
            print(f"DOF 0 - Difference: {abs(solver_dof_0 - entity_dof_0):.8f}")
            
            # Note: Solver has 10 links (includes ground), Entity has 9 links (robot only)
            # Compare link 1 (first robot link)
            solver_link_1 = link_positions[1]  # Link 1 in solver = Link 0 in robot entity
            entity_link_0 = robot_positions[0]
            print(f"Link position - Solver[1]: {solver_link_1}")
            print(f"Link position - Entity[0]: {entity_link_0}")
            
        except Exception as e:
            print(f"Error accessing solver: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to run the simulation and test solver access.
    """
    # Create simulation instance
    sim = GenesisSimulation(show_viewer=False)
    
    # Build the scene
    sim.build_scene()
    
    # Run simulation for a few steps to test the solver access
    print("Starting simulation and testing direct solver access...")
    for step_num in range(3):  # Run for just 3 steps to test
        print(f"\n{'='*50}")
        print(f"--- Step {step_num + 1} ---")
        print(f"{'='*50}")
        sim.step()
        
        # Add a small delay to make output readable
        import time
        time.sleep(1.0)

if __name__ == "__main__":
    main()