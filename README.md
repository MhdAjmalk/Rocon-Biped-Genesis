# Genesis Biped Simulation with ROS 2

This project uses the Genesis World Physics Simulator to run a simulation of a bipedal robot. It publishes the robot's state, including IMU data, joint states, and foot contact forces, to ROS 2 topics.

## Prerequisites

* Ubuntu 22.04
* A working NVIDIA driver
* Miniconda or Anaconda

## Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and Activate the Conda Environment**
    This command will automatically create the `genesis` environment and install all necessary dependencies, including PyTorch, CUDA, and the latest version of Genesis from its GitHub repository.
    ```bash
    conda env create -f environment.yml
    conda activate genesis
    ```

## Usage

1.  **Source ROS 2**
    Make sure your ROS 2 environment is sourced.
    ```bash
    source /opt/ros/humble/setup.bash
    ```
2.  **Run the Node**
    Execute the main script to start the simulation and begin publishing data.
    ```bash
    python3 example.py
    ```

## Published Topics

* **/imu** (`sensor_msgs/Imu`)
    * Publishes the orientation, angular velocity, and linear velocity of the robot's base link.
* **/joint_states** (`sensor_msgs/JointState`)
    * Publishes the position and velocity of all the robot's joints.
* **/contact/revolute_rightfoot** (`std_msgs/Float32`)
    * Publishes the maximum contact force magnitude on the right foot sensor grid.
* **/contact/revolute_leftfoot** (`std_msgs/Float32`)
    * Publishes the maximum contact force magnitude on the left foot sensor grid.
