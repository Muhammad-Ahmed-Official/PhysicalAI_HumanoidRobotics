---
sidebar_label: 'Chapter 2.4: Robot Model Import and Control'
---

# Chapter 2.4: Robot Model Import and Control

## Introduction

Robot model import and control form the foundation of any simulation environment, defining how virtual robots behave and respond to commands. Importing robot models into simulation platforms requires proper representation of the robot's physical properties, kinematic structure, and control interfaces. Proper control systems enable the simulation to respond realistically to commands while maintaining the physics constraints and limitations that would affect the real robot.

In AI and humanoid robotics, accurate robot models are critical for developing and testing control algorithms, path planning, and AI decision-making systems. The import process requires precise definitions of joint configurations, physical properties, and sensor placements, while the control system must replicate the response characteristics of the actual robot hardware.

This chapter explores the process of importing robot models into simulation environments and establishing effective control mechanisms for realistic simulation.

## Learning Objectives

By the end of this chapter, you will be able to:

- Import robot models using standard formats (URDF, SDF, MJCF)
- Configure joint properties, limits, and dynamics for accurate simulation
- Implement control interfaces that mirror real robot systems
- Validate robot model behavior against expected physical properties
- Set up sensor integration within the robot model

## Explanation

### Robot Description Formats

Robot simulation environments typically support one or more standard robot description formats:

1. **URDF (Unified Robot Description Format)**: An XML-based format widely used in ROS/ROS2 ecosystems. It defines the robot's kinematic and dynamic properties, including links, joints, and inertial properties.

2. **SDF (Simulation Description Format)**: Used primarily in Gazebo, it extends URDF capabilities with simulation-specific features like plugins and physics parameters.

3. **MJCF (MuJoCo XML)**: A format used in the MuJoCo physics engine, known for its detailed physics modeling and efficient simulation capabilities.

Each format allows for detailed specification of the robot's physical and kinematic properties.

### Model Import Process

The robot model import process involves several key components:

- **Kinematic Structure**: Definition of joints, links, and their relationships
- **Physical Properties**: Mass, center of mass, and inertial tensor for each link
- **Visual and Collision Models**: Meshes for visualization and simplified geometries for collision detection
- **Sensor Configuration**: Placement and properties of simulated sensors
- **Material Properties**: Surface characteristics affecting physics simulation

### Control Systems in Simulation

Simulation control systems mirror those in real robots, typically including:

- **Joint Controllers**: Position, velocity, or effort control for individual joints
- **High-level Controllers**: Trajectory planning and execution
- **Force/Torque Control**: For manipulation tasks requiring precise force application
- **Whole-Body Controllers**: For complex humanoid behaviors like walking or balancing

## Example Walkthrough

Consider importing a humanoid robot model into a Gazebo simulation with detailed Gazebo integration:

**Step 1: Complete URDF Model with Gazebo Plugins**
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include material definitions -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.1" />
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.2 0.2" />
      </geometry>
      <material name="blue" />
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.2 0.2" />
      </geometry>
    </collision>
  </link>

  <!-- Torso link -->
  <link name="torso">
    <inertial>
      <mass value="5.0" />
      <origin xyz="0 0 0.2" />
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.15 0.4" />
      </geometry>
      <material name="red" />
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0" />
      <geometry>
        <box size="0.2 0.15 0.4" />
      </geometry>
    </collision>
  </link>

  <!-- Hip joint -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link" />
    <child link="torso" />
    <origin xyz="0 0 0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0" />
    <dynamics damping="1.0" friction="0.5" />
  </joint>

  <!-- Left leg -->
  <link name="left_thigh">
    <inertial>
      <mass value="3.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <joint name="left_hip" type="revolute">
    <parent link="torso" />
    <child link="left_thigh" />
    <origin xyz="0 0.1 -0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-2.0" upper="1.0" effort="100" velocity="3.0" />
    <dynamics damping="1.0" friction="0.5" />
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.04" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.04" />
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh" />
    <child link="left_shin" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="0" upper="2.5" effort="100" velocity="3.0" />
    <dynamics damping="1.0" friction="0.5" />
  </joint>

  <!-- Right leg -->
  <link name="right_thigh">
    <inertial>
      <mass value="3.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.05" />
      </geometry>
    </collision>
  </link>

  <joint name="right_hip" type="revolute">
    <parent link="torso" />
    <child link="right_thigh" />
    <origin xyz="0 -0.1 -0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-2.0" upper="1.0" effort="100" velocity="3.0" />
    <dynamics damping="1.0" friction="0.5" />
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.04" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.4" radius="0.04" />
      </geometry>
    </collision>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh" />
    <child link="right_shin" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="0" upper="2.5" effort="100" velocity="3.0" />
    <dynamics damping="1.0" friction="0.5" />
  </joint>

  <!-- Left arm -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.04" />
      </geometry>
      <material name="black" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.04" />
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder" type="revolute">
    <parent link="torso" />
    <child link="left_upper_arm" />
    <origin xyz="0 0.15 0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3.0" />
    <dynamics damping="0.5" friction="0.2" />
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0025" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.03" />
      </geometry>
      <material name="black" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.03" />
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm" />
    <child link="left_lower_arm" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="0" upper="2.0" effort="50" velocity="3.0" />
    <dynamics damping="0.5" friction="0.2" />
  </joint>

  <!-- Right arm -->
  <link name="right_upper_arm">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.04" />
      </geometry>
      <material name="black" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.04" />
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder" type="revolute">
    <parent link="torso" />
    <child link="right_upper_arm" />
    <origin xyz="0 -0.15 0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3.0" />
    <dynamics damping="0.5" friction="0.2" />
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0025" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.03" />
      </geometry>
      <material name="black" />
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder length="0.2" radius="0.03" />
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm" />
    <child link="right_lower_arm" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="1 0 0" />
    <limit lower="0" upper="2.0" effort="50" velocity="3.0" />
    <dynamics damping="0.5" friction="0.2" />
  </joint>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="1.5" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.1" />
      </geometry>
      <material name="white" />
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.1" />
      </geometry>
    </collision>
  </link>

  <joint name="neck" type="revolute">
    <parent link="torso" />
    <child link="head" />
    <origin xyz="0 0 0.4" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-0.5" upper="0.5" effort="20" velocity="2.0" />
    <dynamics damping="0.5" friction="0.2" />
  </joint>

  <!-- Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="left_thigh">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="right_thigh">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_shin">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="right_shin">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_upper_arm">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="left_lower_arm">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_lower_arm">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <!-- Gazebo plugins for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- Transmissions for ROS Control -->
  <transmission name="tran_hip_joint">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_left_hip">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_right_hip">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_left_knee">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_right_knee">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_knee">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_knee_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_left_shoulder">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_right_shoulder">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_shoulder">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_shoulder_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_left_elbow">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_elbow">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_elbow_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_right_elbow">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_elbow">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_elbow_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="tran_neck">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="neck">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="neck_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

**Step 2: Gazebo-Specific Control Configuration**
```yaml
# controllers.yaml for ROS control with Gazebo
humanoid_robot:
  # Joint state controller
  joint_state_controller:
    type: joint_state_controller/JointStateController
    publish_rate: 50

  # Position controllers for each joint
  hip_joint_position_controller:
    type: position_controllers/JointPositionController
    joint: hip_joint
    pid: {p: 100.0, i: 0.01, d: 10.0}

  left_hip_position_controller:
    type: position_controllers/JointPositionController
    joint: left_hip
    pid: {p: 100.0, i: 0.01, d: 10.0}

  right_hip_position_controller:
    type: position_controllers/JointPositionController
    joint: right_hip
    pid: {p: 100.0, i: 0.01, d: 10.0}

  left_knee_position_controller:
    type: position_controllers/JointPositionController
    joint: left_knee
    pid: {p: 100.0, i: 0.01, d: 10.0}

  right_knee_position_controller:
    type: position_controllers/JointPositionController
    joint: right_knee
    pid: {p: 100.0, i: 0.01, d: 10.0}

  left_shoulder_position_controller:
    type: position_controllers/JointPositionController
    joint: left_shoulder
    pid: {p: 50.0, i: 0.01, d: 5.0}

  right_shoulder_position_controller:
    type: position_controllers/JointPositionController
    joint: right_shoulder
    pid: {p: 50.0, i: 0.01, d: 5.0}

  left_elbow_position_controller:
    type: position_controllers/JointPositionController
    joint: left_elbow
    pid: {p: 50.0, i: 0.01, d: 5.0}

  right_elbow_position_controller:
    type: position_controllers/JointPositionController
    joint: right_elbow
    pid: {p: 50.0, i: 0.01, d: 5.0}

  neck_position_controller:
    type: position_controllers/JointPositionController
    joint: neck
    pid: {p: 20.0, i: 0.01, d: 2.0}
```

**Step 3: Launch Configuration with Gazebo**
```xml
<!-- Launch file to load robot model and controllers -->
<launch>
  <!-- Set the path to the URDF file -->
  <param name="robot_description" command="$(find xacro)/xacro $(find humanoid_description)/urdf/humanoid.xacro" />

  <!-- Publish TFs for links without joints -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" respawn="false" output="screen">
    <param name="publish_frequency" type="double" value="50.0" />
    <param name="use_tf_sending" type="bool" value="true" />
  </node>

  <!-- Spawn the robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-param robot_description -urdf -model humanoid_robot -x 0 -y 0 -z 1"
        respawn="false" output="screen" />

  <!-- Start Gazebo with the world file -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find humanoid_gazebo)/worlds/humanoid_lab.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Load controller configurations -->
  <rosparam file="$(find humanoid_control)/config/controllers.yaml" command="load"/>

  <!-- Start controller manager -->
  <node name="controller_spawner" pkg="controller_manager" type="spawner" respawn="false"
        output="screen" args="joint_state_controller
                              hip_joint_position_controller
                              left_hip_position_controller
                              right_hip_position_controller
                              left_knee_position_controller
                              right_knee_position_controller
                              left_shoulder_position_controller
                              right_shoulder_position_controller
                              left_elbow_position_controller
                              right_elbow_position_controller
                              neck_position_controller"/>
</launch>
```

**Step 4: Testing the Configuration**
```bash
# Launch the simulation
roslaunch humanoid_gazebo humanoid_world.launch

# Send a command to move a joint
rostopic pub /humanoid_robot/left_shoulder_position_controller/command std_msgs/Float64 "data: 0.5"

# Monitor joint states
rostopic echo /humanoid_robot/joint_states
```

### Unity Robot Model Import and Control

**Step 1: Robot Model in Unity**
In Unity, robot models are typically imported as 3D models with properly configured colliders, rigidbodies, and joint components:

```csharp
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class UnityRobotJoint : MonoBehaviour
{
    [Header("Joint Configuration")]
    public ArticulationBody joint;
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float maxForce = 100f;

    [Header("Control Interface")]
    public float targetPosition = 0f;
    public float stiffness = 100f;
    public float damping = 10f;

    void Start()
    {
        if (joint == null)
            joint = GetComponent<ArticulationBody>();

        ConfigureJoint();
    }

    void ConfigureJoint()
    {
        if (joint != null)
        {
            // Set joint limits
            ArticulationDrive drive = joint.xDrive;
            drive.lowerLimit = minAngle;
            drive.upperLimit = maxAngle;
            drive.forceLimit = maxForce;
            drive.stiffness = stiffness;
            drive.damping = damping;

            joint.xDrive = drive;
        }
    }

    void Update()
    {
        SetJointTarget(targetPosition);
    }

    public void SetJointTarget(float position)
    {
        if (joint != null)
        {
            ArticulationDrive drive = joint.xDrive;
            drive.target = Mathf.Clamp(position, minAngle, maxAngle);
            joint.xDrive = drive;
        }
    }
}
```

**Step 2: Robot Controller for Unity**
Create a central controller to manage the entire robot:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public UnityRobotJoint[] joints;
    public string jointStatesTopic = "joint_states";

    [Header("Control Settings")]
    public float controlFrequency = 50f; // Hz
    private float controlInterval;
    private float lastControlTime;

    private ROSConnection ros;
    private string[] jointNames;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(jointStatesTopic);

        // Initialize joint names based on joint objects
        jointNames = new string[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            jointNames[i] = joints[i].name;
        }

        controlInterval = 1.0f / controlFrequency;
        lastControlTime = Time.time;
    }

    void Update()
    {
        // Publish joint states at the specified frequency
        if (Time.time - lastControlTime >= controlInterval)
        {
            PublishJointStates();
            lastControlTime = Time.time;
        }
    }

    void PublishJointStates()
    {
        JointStateMsg jointState = new JointStateMsg
        {
            header = new HeaderMsg
            {
                stamp = new TimeMsg { sec = (int)Time.time, nanosec = (int)((Time.time % 1) * 1e9) },
                frame_id = "base_link"
            },
            name = jointNames,
            position = new double[joints.Length],
            velocity = new double[joints.Length],
            effort = new double[joints.Length]
        };

        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i].joint != null)
            {
                jointState.position[i] = joints[i].joint.jointPosition.x;
                jointState.velocity[i] = joints[i].joint.jointVelocity.x;
                jointState.effort[i] = joints[i].joint.jointForce.x;
            }
        }

        ros.Publish(jointStatesTopic, jointState);
    }

    public void SetJointPositions(float[] positions)
    {
        if (positions.Length != joints.Length)
        {
            Debug.LogError("Number of positions doesn't match number of joints");
            return;
        }

        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].SetJointTarget(positions[i]);
        }
    }
}
```

**Step 3: Robot Model Import Process in Unity**
1. Import 3D model files (FBX, OBJ, etc.) into Unity
2. Create appropriate colliders for each part of the robot
3. Add Rigidbody components to parts that need physics simulation
4. Use ArticulationBody components to create joints between robot parts
5. Configure joint limits, friction, and motor properties
6. Add sensor components (cameras, raycast sensors) as needed
7. Implement control scripts to respond to commands from ROS or AI systems

**Step 4: Testing Unity Robot Configuration**
- Verify that all joints move correctly within their limits
- Check that physics behave realistically
- Test sensor data output
- Validate ROS communication if using ROS integration

This comprehensive approach creates a complete humanoid robot model with proper physics, visualization, and control interfaces in Unity simulation, complementing the capabilities provided by Gazebo with its advantages in rendering and AI training.

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Robot Model Import & Control                      │
│                                                                     │
│  ┌─────────────────────────┐        ┌─────────────────────────────┐│
│  │    Robot Description    │   →    │    Simulation Environment   ││
│  │         (URDF/SDF)      │        │        (Gazebo/Unity)      ││
│  │                         │        │                             ││
│  │ ┌─────────────────────┐ │        │ ┌─────────────────────────┐ ││
│  │ │ Kinematic Structure │ │        │ │ Physics Simulation      │ ││
│  │ │ • Joints            │ │        │ │ • Forces & Collisions   │ ││
│  │ │ • Links             │ │        │ │ • Mass & Inertia        │ ││
│  │ │ • Transforms        │ │        │ │ • Friction & Damping    │ ││
│  │ └─────────────────────┘ │        │ └─────────────────────────┘ ││
│  │                         │        │                             ││
│  │ ┌─────────────────────┐ │        │ ┌─────────────────────────┐ ││
│  │ │ Physical Properties │ │        │ │ Control Interface       │ ││
│  │ │ • Mass              │ │        │ │ • Joint Controllers     │ ││
│  │ │ • Inertia           │ │        │ │ • Trajectory Planning   │ ││
│  │ │ • Geometry          │ │        │ │ • Sensor Feedback       │ ││
│  │ └─────────────────────┘ │        │ └─────────────────────────┘ ││
│  │                         │        │                             ││
│  │ ┌─────────────────────┐ │        │ ┌─────────────────────────┐ ││
│  │ │ Visual/Collision    │ │        │ │ Real-time Visualization │ ││
│  │ │ • Meshes            │ │        │ │ • Rendering (Unity)     │ ││
│  │ │ • Materials         │ │        │ │ • Sensors Display       │ ││
│  │ │ • Sensors           │ │        │ │ • Robot State Overlay   │ ││
│  │ └─────────────────────┘ │        │ └─────────────────────────┘ ││
│  └─────────────────────────┘        └─────────────────────────────┘│
│                                                                     │
│  ←───────────────────── ROS/Unity Control ────────────────────────→ │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Import robot models using standard formats (URDF, SDF, MJCF)
- [ ] Configure joint properties, limits, and dynamics
- [ ] Set up control interfaces that mirror real robot systems
- [ ] Detail the kinematic structure definition process
- [ ] Explain physical properties configuration
- [ ] Provide example of complete robot model definition
- [ ] Include visual representation of model import and control system