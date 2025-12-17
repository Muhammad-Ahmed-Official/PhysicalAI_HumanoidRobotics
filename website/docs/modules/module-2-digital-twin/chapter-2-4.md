---
id: chapter-2-4
title: "Chapter 2.4: Robot Model Import and Control"
description: "Importing robot models and implementing control systems in simulation"
tags: [robot-model, control, urdf, sdf, simulation]
---

# Chapter 2.4: Robot Model Import and Control

## Introduction

Importing accurate robot models and implementing effective control systems in simulation is essential for the simulation-first approach to humanoid robotics. This chapter covers techniques for creating, importing, and controlling robot models in simulation environments, ensuring that the virtual robot accurately represents the physical counterpart.

## Learning Outcomes

- Students will understand the process of creating robot models for simulation
- Learners will be able to import robot models in Gazebo and Unity
- Readers will be familiar with different control approaches for simulated robots
- Students will be able to implement basic control systems in simulation

## Core Concepts

Effective robot model import and control in simulation requires understanding of several key areas:

1. **Model Format**: Understanding URDF (Unified Robot Description Format) for ROS/Gazebo or other formats for Unity
2. **Joint Configuration**: Properly defining joint types, limits, and dynamics
3. **Transmission Systems**: Configuring how actuators control joints
4. **Control Framework**: Implementing controllers for robot joints and actuators
5. **Hardware Interfaces**: Simulating hardware interfaces that will exist on the physical robot

Proper model import ensures that the virtual robot behaves similarly to the physical robot, which is essential for successful simulation-to-real transfer.

## Simulation Walkthrough

Importing and controlling robot models in both Gazebo and Unity:

<Tabs>
  <TabItem value="gazebo" label="Gazebo Model/Control">
    ```xml
    <!-- Complete humanoid robot model with control -->
    <robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
      <!-- Include position and velocity transmissions for control -->
      <transmission name="left_hip_trans">
        <type>transmission_interface/SimpleTransmission</type>
        <joint name="left_hip_joint">
          <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
        </joint>
        <actuator name="left_hip_motor">
          <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
          <mechanicalReduction>1</mechanicalReduction>
        </actuator>
      </transmission>
      
      <transmission name="right_hip_trans">
        <type>transmission_interface/SimpleTransmission</type>
        <joint name="right_hip_joint">
          <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
        </joint>
        <actuator name="right_hip_motor">
          <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
          <mechanicalReduction>1</mechanicalReduction>
        </actuator>
      </transmission>
      
      <!-- Controller configuration -->
      <gazebo>
        <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
          <robotNamespace>/humanoid</robotNamespace>
        </plugin>
      </gazebo>
    </robot>
    ```
    
    ```yaml
    # Controller configuration file (controllers.yaml)
    humanoid_controller:
      # Joint trajectory controller for position control
      joint_trajectory_controller:
        type: position_controllers/JointTrajectoryController
        joints:
          - left_hip_joint
          - right_hip_joint
          - left_knee_joint
          - right_knee_joint
          - left_ankle_joint
          - right_ankle_joint
        state_publish_rate: 50
        action_monitor_rate: 20
        stop_trajectory_duration: 0
        state_interface: PositionCommandInterface
        joint_state_publish_rate: 50
      
      # Individual joint controllers
      joint_state_controller:
        type: joint_state_controller/JointStateController
        publish_rate: 50
    ```
  </TabItem>
  <TabItem value="unity" label="Unity Model/Control">
    ```
    // Unity Robot Model and Control Implementation
    
    1. Import Robot Model:
       - Convert URDF to Unity format using URDF-Importer
       - Or manually create model with correct joint hierarchy
       - Ensure center of mass is correctly positioned
       - Add appropriate colliders and rigidbodies
    
    2. Joint Configuration:
       - Use Unity's ConfigurableJoint component
       - Set appropriate joint limits and constraints
       - Configure spring and damper properties
       - Set joint drive for actuator simulation
    
    3. Control Implementation:
       a) Position Control:
          - Set targetPosition property on ConfigurableJoint
          - Use joint drive with appropriate force limits
          
       b) Velocity Control:
          - Set targetVelocity property on ConfigurableJoint
          - Apply torque based on velocity error
          
       c) Torque Control:
          - Apply forces/torques directly to rigidbody
          - More realistic but requires careful force management
    
    4. Control Script Example:
       public class HumanoidJointController : MonoBehaviour
       {
           public ConfigurableJoint joint;
           public float targetPosition = 0f;
           
           void Update()
           {
               joint.targetPosition = targetPosition;
           }
       }
    ```
  </TabItem>
</Tabs>

## Visual Explanation

```
[Robot Model Import and Control Diagram]

Physical Robot              Simulated Robot
┌─────────────────┐       ┌─────────────────┐
│  Real Hardware  │       │  Virtual Model  │
│  (Motors,      │  ←→   │  (URDF/SDF,    │
│   Sensors, etc.)│       │   Unity Model)  │
└─────────────────┘       └─────────────────┘
         │                         │
         │ Joint States            │ Joint States
         │ (Position, Velocity,    │ (Simulated)
         │  Effort)                │ Position, Velocity,
         └─────────────────────────┘ Effort)
                   │
            ┌─────────────┐
            │ Controllers │←─┐ Control Commands
            │ (Position,  │  │ (Target Position,
            │  Velocity,  │  │  Velocity, Torque)
            │  Torque)    │──┘
            └─────────────┘
                   │
        ┌─────────────────────────┐
        │ Control Communication   │
        │ (ROS topics, Unity RPC) │
        └─────────────────────────┘

The simulated robot model accurately represents the physical robot
with proper joints and control interfaces for effective simulation.
```

## Checklist

- [x] Understand the importance of accurate robot models
- [x] Know how to configure transmissions for control
- [x] Implement basic controllers in simulation
- [ ] Successfully imported robot model in simulation
- [ ] Configured control interfaces properly
- [ ] Self-assessment: How would you validate that your simulated robot behaves similarly to the physical robot?