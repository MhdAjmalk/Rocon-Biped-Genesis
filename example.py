import genesis as gs
import pathlib
import numpy as np

class GenesisSimulation:
    """
    Handles the setup and state of the Genesis physics simulation.
    This class demonstrates how to query DOF and link names from the solver.
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
        
        # --- Load Robot ---
        self.biped_robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(biped_urdf_path),
                fixed=False
            )
        )
        
        print("Genesis simulation environment initialized.")
    
    def build_scene(self):
        """
        Builds the simulation scene after all entities are added.
        """
        self.scene.build()
        print("Genesis scene built.")
    def explore_solver_methods(self):
        """
        Explore and print available methods and attributes of the solver.
        """
        print("\n=== Exploring Solver Object ===")
        solver = self.scene.rigid_solver
        
        # Get all attributes and methods
        all_methods = dir(solver)
        
        # Filter for actual methods (vs attributes)
        methods = [m for m in all_methods if callable(getattr(solver, m)) and not m.startswith('__')]
        attributes = [a for a in all_methods if not callable(getattr(solver, a)) and not a.startswith('__')]
        
        # Print methods with their docstrings
        print(f"\n=== Available Solver Methods ({len(methods)}) ===")
        for method in sorted(methods):
            doc = getattr(solver, method).__doc__
            doc_summary = doc.split('\n')[0] if doc else "No documentation"
            print(f"  • {method}() - {doc_summary}")
        
        # Print attributes
        print(f"\n=== Available Solver Attributes ({len(attributes)}) ===")
        for attr in sorted(attributes):
            value = getattr(solver, attr)
            type_info = type(value).__name__
            print(f"  • {attr}: {type_info}")
        
        # Explore specific getter methods more deeply
        getter_methods = [m for m in methods if m.startswith('get_')]
        if getter_methods:
            print(f"\n=== Detailed Getter Methods ({len(getter_methods)}) ===")
            for method in sorted(getter_methods):
                doc = getattr(solver, method).__doc__
                print(f"  • {method}():")
                print(f"    {doc if doc else 'No documentation'}")
        
    def query_dof_and_link_names(self):
        """
        Query and display DOF and link names with their indices.
        This helps understand the mapping between solver indices and names.
        """
        try:
            # Access the solver
            solver = self.scene.rigid_solver
            
            print(f"\n=== Solver Information ===")
            print(f"Total DOFs in solver: {solver.n_dofs}")
            print(f"Total links in solver: {solver.n_links}")
            
            print(f"\n=== Robot Entity Information ===")
            print(f"Robot DOFs: {self.biped_robot.n_dofs}")
            print(f"Robot links: {self.biped_robot.n_links}")
            
            # Get DOF names from robot joints
            print(f"\n=== DOF Index to Name Mapping ===")
            joints = self.biped_robot.joints
            print(f"Number of joints: {len(joints)}")
            for i, joint in enumerate(joints):
                print(f"DOF {i:2d}: {joint.name}")
                
            # Get link names from robot links
            print(f"\n=== Link Index to Name Mapping ===")
            links = self.biped_robot.links
            print(f"Number of links: {len(links)}")
            for i, link in enumerate(links):
                print(f"Link {i:2d}: {link.name}")
            
            # Create lookup dictionaries for easy access
            self.dof_idx_to_name = {i: joint.name for i, joint in enumerate(joints)}
            self.dof_name_to_idx = {joint.name: i for i, joint in enumerate(joints)}
            self.link_idx_to_name = {i: link.name for i, link in enumerate(links)}
            self.link_name_to_idx = {link.name: i for i, link in enumerate(links)}

            # Demonstrate lookup usage
            print(f"\n=== Lookup Dictionary Examples ===")
            if len(self.dof_idx_to_name) > 3:
                print(f"DOF index 3 is: '{self.dof_idx_to_name[3]}'")
            
            # Look for specific joints if they exist
            target_joints = ['right_knee', 'left_knee', 'right_hip1', 'left_hip1']
            for joint_name in target_joints:
                if joint_name in self.dof_name_to_idx:
                    idx = self.dof_name_to_idx[joint_name]
                    print(f"Joint '{joint_name}' is at index: {idx}")
            
            # Look for foot links
            target_links = ['revolute_leftfoot', 'revolute_rightfoot']
            for link_name in target_links:
                if link_name in self.link_name_to_idx:
                    idx = self.link_name_to_idx[link_name]
                    print(f"Link '{link_name}' is at index: {idx}")
                    
            # Show important robot DOF indices for control
            print(f"\n=== Important Robot DOF Indices for Control ===")
            control_joints = ['left_hip1', 'right_hip1', 'left_hip2', 'right_hip2', 
                            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            for joint_name in control_joints:
                if joint_name in self.dof_name_to_idx:
                    idx = self.dof_name_to_idx[joint_name]
                    print(f"  {joint_name:12s} -> DOF index {idx}")
                    
        except Exception as e:
            print(f"Error querying names: {e}")
            import traceback
            traceback.print_exc()
    
    def test_solver_attributes_access(self):
        """
        Test accessing solver attributes: entities, joints, and links directly.
        """
        try:
            solver = self.scene.rigid_solver
            
            print(f"\n=== Testing Solver Attributes Access ===")
            
            # Access solver entities
            print(f"\n--- Solver Entities ---")
            print(f"Number of entities: {solver._n_entities}")
            print(f"Entities list length: {len(solver.entities)}")
            for i, entity in enumerate(solver.entities):
                print(f"  Entity {i}: {type(entity).__name__}")
                if hasattr(entity, 'morph'):
                    print(f"    Morph: {type(entity.morph).__name__}")
            
            # Access solver joints
            print(f"\n--- Solver Joints ---")
            print(f"Number of joints: {len(solver.joints)}")
            for i, joint in enumerate(solver.joints):
                joint_name = joint.name if hasattr(joint, 'name') else f"joint_{i}"
                joint_type = type(joint).__name__
                print(f"  Joint {i:2d}: {joint_name} ({joint_type})")
                
                # Try to get joint properties if available
                if hasattr(joint, 'dof_start') and hasattr(joint, 'dof_end'):
                    print(f"    DOF range: {joint.dof_start} to {joint.dof_end}")
                if hasattr(joint, 'joint_type'):
                    print(f"    Type: {joint.joint_type}")
            
            # Access solver links
            print(f"\n--- Solver Links ---")
            print(f"Number of links: {len(solver.links)}")
            for i, link in enumerate(solver.links):
                link_name = link.name if hasattr(link, 'name') else f"link_{i}"
                link_type = type(link).__name__
                print(f"  Link {i:2d}: {link_name} ({link_type})")
                
                # Try to get link properties if available
                if hasattr(link, 'mass'):
                    print(f"    Mass: {link.mass}")
                if hasattr(link, 'entity_idx'):
                    print(f"    Entity index: {link.entity_idx}")
            
            # Compare with robot entity access
            print(f"\n--- Comparison: Robot Entity vs Solver ---")
            print(f"Robot entity joints: {len(self.biped_robot.joints)}")
            print(f"Robot entity links: {len(self.biped_robot.links)}")
            
            print(f"Solver joints: {len(solver.joints)}")
            print(f"Solver links: {len(solver.links)}")
            print("solver.links_state")
            print(solver.links_state.pos[2:])
            # Show the difference (solver includes all entities, robot is just one entity)
            if len(solver.joints) > len(self.biped_robot.joints):
                extra_joints = len(solver.joints) - len(self.biped_robot.joints)
                print(f"Solver has {extra_joints} additional joints (likely from other entities)")
            
        except Exception as e:
            print(f"Error accessing solver attributes: {e}")
            import traceback
            traceback.print_exc()

    def test_direct_access_with_names(self):
        """
        Test direct Taichi field access using the name mappings.
        """
        try:
            solver = self.scene.rigid_solver
            dofs_state = solver.dofs_state
            links_state = solver.links_state

            
            
            print(f"\n=== Testing Direct Taichi Field Access ===")
            
            # Show first few DOF states with names (if we have them)
            print(f"\nDOF States (with names if available):")
            for dof_idx in range(solver.n_dofs):
                dof_name = "unknown"
                if hasattr(self, 'dof_idx_to_name') and dof_idx in self.dof_idx_to_name:
                    dof_name = self.dof_idx_to_name[dof_idx]
                
                dof_pos = dofs_state.pos[dof_idx, 0]
                dof_vel = dofs_state.vel[dof_idx, 0]
                dof_force = dofs_state.force[dof_idx, 0]
                print(f"  {dof_idx:2d} ({dof_name:15s}): pos={dof_pos:8.4f}, vel={dof_vel:8.4f}, force={dof_force:8.4f}")
            
            # Show first few link states with names (if we have them)
            print(f"\nLink States (with names if available):")
            for link_idx in range(solver.n_links):
                link_name = "unknown"
                if hasattr(self, 'link_idx_to_name') and link_idx in self.link_idx_to_name:
                    link_name = self.link_idx_to_name[link_idx]
                
                link_pos = links_state.pos[link_idx, 0]
                print(f"  {link_idx:2d} ({link_name:20s}): pos=[{link_pos[0]:7.4f}, {link_pos[1]:7.4f}, {link_pos[2]:7.4f}]")
                
        except Exception as e:
            print(f"Error in direct access test: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to demonstrate DOF and link name querying.
    """
    # Create simulation instance
    sim = GenesisSimulation(show_viewer=False)
    
    # Build the scene
    sim.build_scene()
    
    # Query and display all names
    sim.query_dof_and_link_names()
    
    # Test solver attributes access
    sim.test_solver_attributes_access()
    
    # Step simulation once to get meaningful state values
    sim.scene.step()
    
    # Test direct access with names
    sim.test_direct_access_with_names()

if __name__ == "__main__":
    main()