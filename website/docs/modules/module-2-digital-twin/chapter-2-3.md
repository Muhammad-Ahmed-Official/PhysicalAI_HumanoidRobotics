---
id: chapter-2-3
title: "Chapter 2.3: Physics and Sensor Simulation"
description: "Implementing accurate physics and sensor models in robotics simulation"
tags: [physics, sensors, simulation, robotics]
---

# Chapter 2.3: Physics and Sensor Simulation

## Introduction

Accurate physics and sensor simulation is critical for effective simulation-to-real transfer in humanoid robotics. This chapter explores how to create realistic physics models and sensor simulations that closely match their real-world counterparts, ensuring that behaviors learned in simulation translate effectively to physical robots.

## Learning Outcomes

- Students will understand the importance of accurate physics simulation
- Learners will be able to configure realistic sensor models in simulation
- Readers will be familiar with techniques for minimizing the reality gap
- Students will be able to tune physics parameters for accurate simulation

## Core Concepts

Successful physics and sensor simulation requires attention to several key areas:

1. **Mass Distribution**: Accurately modeling the mass and inertia properties of the robot
2. **Friction Models**: Implementing realistic friction coefficients for different materials
3. **Sensor Noise**: Adding appropriate noise models to simulated sensors
4. **Dynamics Accuracy**: Ensuring simulated joint dynamics match physical robots
5. **Contact Mechanics**: Modeling contact forces and collisions accurately

To minimize the reality gap (the difference between simulation and reality), engineers often employ domain randomization techniques, where simulation parameters are randomly varied within realistic bounds.

## Simulation Walkthrough

Configuring physics and sensors in both Gazebo and Unity:

<Tabs>
  <TabItem value="gazebo" label="Gazebo Physics/Sensors">
    ```xml
    <!-- Example URDF with accurate physics properties -->
    <robot name="humanoid_with_physics">
      <link name="torso">
        <inertial>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <mass value="5.0"/>
          <inertia ixx="0.1" ixy="0.0" ixz="0.0"
                   iyy="0.15" iyz="0.0" izz="0.1"/>
        </inertial>
        <collision>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <geometry>
            <box size="0.2 0.2 0.4"/>
          </geometry>
        </collision>
        <visual>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <geometry>
            <box size="0.2 0.2 0.4"/>
          </geometry>
        </visual>
      </link>
      
      <!-- Example sensor configuration -->
      <gazebo reference="head_camera">
        <sensor type="camera" name="head_camera_sensor">
          <update_rate>30.0</update_rate>
          <camera name="head_camera">
            <horizontal_fov>1.3962634</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
              <format>R8G8B8</format>
            </image>
            <clip>
              <near>0.1</near>
              <far>100</far>
            </clip>
          </camera>
          <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
            <alwaysOn>true</alwaysOn>
            <updateRate>30.0</updateRate>
            <cameraName>head_camera</cameraName>
            <imageTopicName>image_raw</imageTopicName>
            <cameraInfoTopicName>camera_info</cameraInfoTopicName>
            <frameName>head_camera_frame</frameName>
          </plugin>
        </sensor>
      </gazebo>
    </robot>
    ```
  </TabItem>
  <TabItem value="unity" label="Unity Physics/Sensors">
    ```
    // Unity Physics and Sensor Configuration
    
    1. Physics Configuration:
       - Use PhysX engine with appropriate gravity settings
       - Configure Material with:
         * Static friction: 0.5-0.8 (depending on surface)
         * Dynamic friction: 0.3-0.6
         * Bounce combine: Multiply
         * Friction combine: Average
    
    2. Robot Link Configuration:
       - Add Rigidbody component with:
         * Mass matching physical robot
         * Appropriate drag and angular drag
         * Interpolate: "Interpolate" for smooth motion
       - Add appropriate Collider shapes (Box, Sphere, Capsule, Mesh)
    
    3. Sensor Simulation:
       - Camera sensors: Configure with realistic FOV, resolution, and noise
       - IMU: Add custom script to simulate accelerometer/gyroscope data
       - Force/Torque sensors: Use PhysX contact callbacks
       - Joint position/velocity: Read from joint components directly
    
    4. Noise Implementation:
       - Apply random noise to sensor readings
       - Add realistic time delays
       - Simulate sensor saturation limits
    ```
  </TabItem>
</Tabs>

## Visual Explanation

```
[Physics and Sensor Simulation Diagram]

Physical Robot Sensor         Simulated Sensor
┌─────────────────────┐    ┌─────────────────────┐
│ Raw Data:           │    │ Raw Data:           │
│ Position, Velocity  │    │ Position, Velocity  │
│ Acceleration, etc.  │ -> │ Acceleration, etc.  │
└─────────────────────┘    └─────────────────────┘
           ↓                        ↓
    ┌─────────────┐          ┌─────────────┐
    │ Noise &     │    ≠     │ Noise &     │
    │ Distortions │ Actual   │ Distortions │ Simulated
    │ (Real)      │ Values   │ (Modeled)   │ Values
    └─────────────┘          └─────────────┘
           ↓                        ↓
   ┌───────────────┐        ┌───────────────┐
   │ Filtered Data │   ≈    │ Filtered Data │
   │ (Processed)   │        │ (Processed)   │
   └───────────────┘        └───────────────┘
           ↓                        ↓
    ┌─────────────────────────────────────┐
    │ Reality Gap Minimization Techniques │
    │ - Domain randomization              │
    │ - System identification             │
    │ - High-fidelity modeling            │
    │ - Sensor calibration                │
    └─────────────────────────────────────┘

Accurate physics and sensor simulation aims to minimize the
difference between real and simulated sensor readings and
robot dynamics.
```

## Checklist

- [x] Understand the importance of accurate physics simulation
- [x] Know how to configure sensors in simulation
- [ ] Implemented realistic friction models
- [ ] Added appropriate noise to sensor simulations
- [ ] Tuned physics parameters for accuracy
- [ ] Self-assessment: How would you validate that your sensor noise models are realistic?