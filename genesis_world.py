# genesis.py

import genesis as gs
import pathlib

class GenesisSimulation:
    """
    Handles the setup and state of the Genesis physics simulation.
    This class is not a ROS node. It purely manages simulation objects.
    """
    def __init__(self, show_viewer=True):
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

    def step(self):
        """
        Advances the simulation by one step.
        """
        self.scene.step()

