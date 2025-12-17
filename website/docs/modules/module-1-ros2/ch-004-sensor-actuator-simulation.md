---
id: ch-004
title: "Chapter 1.4: Sensor and Actuator Simulation"
description: "Simulating sensors and actuators in ROS 2 for humanoid robotics development"
tags: [ros2, simulation, sensors, actuators, gazebo]
---

# Chapter 1.4: Sensor and Actuator Simulation

## Introduction

In humanoid robotics development, simulating sensors and actuators is crucial for testing and validating robot behaviors before deploying to real hardware. ROS 2 provides powerful tools and patterns for creating realistic simulations that can be used for development, testing, and training. This chapter explores how to simulate various sensors and actuators in ROS 2, enabling safe and cost-effective development of humanoid robot behaviors.

## Learning Outcomes

- Students will understand the importance of sensor and actuator simulation in robotics development
- Learners will be able to implement simulated sensors and actuators in ROS 2
- Readers will be familiar with different types of sensors commonly used in humanoid robots
- Students will be able to create simulation-ready sensor and actuator models

## Core Concepts

### Types of Sensors in Humanoid Robots
Humanoid robots use various sensors to perceive their environment:
- **IMU (Inertial Measurement Unit)**: Measures orientation, velocity, and gravitational forces
- **LIDAR**: Provides 2D or 3D distance measurements for mapping and navigation
- **Cameras**: Provide visual information for perception and recognition
- **Force/Torque sensors**: Measure forces applied to robot joints or end-effectors
- **Joint position/velocity sensors**: Track the position and speed of each joint
- **Tactile sensors**: Detect physical contact and pressure

### Types of Actuators in Humanoid Robots
Actuators enable humanoid robots to interact with the environment:
- **Servo motors**: Precise positioning control for joints
- **Linear actuators**: Provide linear motion for specific applications
- **Pneumatic/hydraulic actuators**: High-force actuators for specific applications
- **Muscle-based actuators**: Emerging technology mimicking biological muscles

### Simulation Considerations
Simulation must balance accuracy with computational efficiency:
- **Realism**: How closely the simulation matches real-world behavior
- **Computational cost**: How much processing power is required
- **Development speed**: How quickly scenarios can be tested
- **Transferability**: How well behaviors learned in simulation transfer to reality

### Sensor Simulation Patterns
Common patterns for implementing sensor simulation in ROS 2:
- **Publisher-based**: Sensors publish data to topics at regular intervals
- **Service-based**: Sensors provide on-demand data when requested
- **Action-based**: Complex sensor operations that take time to complete

### Actuator Simulation Patterns
Patterns for simulating actuator behavior:
- **Position control**: Commanding specific positions
- **Velocity control**: Controlling the speed of movement
- **Effort control**: Controlling the force/torque applied
- **Hybrid control**: Combining different control types

## Simulation Walkthrough

Let's implement a simulated IMU sensor and joint actuator for a humanoid robot:

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Float64MultiArray
    from builtin_interfaces.msg import Time
    import numpy as np
    import math
    import random


    class HumanoidSensorSimulator(Node):
        """
        Simulates IMU and joint sensors for a humanoid robot
        """
        
        def __init__(self):
            super().__init__('humanoid_sensor_simulator')
            
            # IMU sensor publisher
            self.imu_publisher = self.create_publisher(Imu, '/imu/data', 10)
            
            # Joint state publisher
            self.joint_publisher = self.create_publisher(JointState, '/joint_states', 10)
            
            # Joint command subscriber (for actuator control)
            self.joint_command_subscriber = self.create_subscription(
                Float64MultiArray,
                '/joint_commands',
                self.joint_command_callback,
                10
            )
            
            # Timer for publishing sensor data
            self.timer = self.create_timer(0.02, self.publish_sensor_data)  # 50 Hz
            
            # Initialize joint states
            self.joint_names = [
                'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
                'left_shoulder_joint', 'left_elbow_joint',
                'right_shoulder_joint', 'right_elbow_joint'
            ]
            self.joint_positions = [0.0] * len(self.joint_names)
            self.joint_velocities = [0.0] * len(self.joint_names)
            self.joint_efforts = [0.0] * len(self.joint_names)
            
            # IMU initial state
            self.orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion (x, y, z, w)
            self.angular_velocity = [0.0, 0.0, 0.0]
            self.linear_acceleration = [0.0, 0.0, 9.81]  # Gravity only initially
            
            self.get_logger().info('Humanoid sensor simulator initialized')
            
        def joint_command_callback(self, msg):
            """Handle joint command messages"""
            # In simulation, we update joint positions based on commands
            if len(msg.data) == len(self.joint_names):
                for i in range(len(msg.data)):
                    # Apply some simple dynamics (velocity based on position difference)
                    target_pos = msg.data[i]
                    current_pos = self.joint_positions[i]
                    velocity = (target_pos - current_pos) * 2.0  # Simple PD-like control
                    self.joint_positions[i] = target_pos
                    self.joint_velocities[i] = velocity
                self.get_logger().info(f'Updated {len(msg.data)} joint positions')
            else:
                self.get_logger().warn(
                    f'Joint command length mismatch: expected {len(self.joint_names)}, got {len(msg.data)}'
                )
                
        def publish_sensor_data(self):
            """Publish simulated IMU and joint sensor data"""
            # Publish joint states
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = self.joint_names
            joint_msg.position = self.joint_positions[:]
            joint_msg.velocity = self.joint_velocities[:]
            joint_msg.effort = self.joint_efforts[:]
            self.joint_publisher.publish(joint_msg)
            
            # Simulate IMU data with some noise
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'imu_link'
            
            # Add some realistic noise to IMU data
            noise_factor = 0.01
            imu_msg.orientation.x = self.orientation[0] + random.uniform(-noise_factor, noise_factor)
            imu_msg.orientation.y = self.orientation[1] + random.uniform(-noise_factor, noise_factor)
            imu_msg.orientation.z = self.orientation[2] + random.uniform(-noise_factor, noise_factor)
            imu_msg.orientation.w = self.orientation[3] + random.uniform(-noise_factor, noise_factor)
            
            # Normalize quaternion
            norm = math.sqrt(
                imu_msg.orientation.x**2 + 
                imu_msg.orientation.y**2 + 
                imu_msg.orientation.z**2 + 
                imu_msg.orientation.w**2
            )
            imu_msg.orientation.x /= norm
            imu_msg.orientation.y /= norm
            imu_msg.orientation.z /= norm
            imu_msg.orientation.w /= norm
            
            # Angular velocity with noise
            angular_noise = 0.001
            imu_msg.angular_velocity.x = self.angular_velocity[0] + random.uniform(-angular_noise, angular_noise)
            imu_msg.angular_velocity.y = self.angular_velocity[1] + random.uniform(-angular_noise, angular_noise)
            imu_msg.angular_velocity.z = self.angular_velocity[2] + random.uniform(-angular_noise, angular_noise)
            
            # Linear acceleration with gravity and movement
            accel_noise = 0.1
            imu_msg.linear_acceleration.x = self.linear_acceleration[0] + random.uniform(-accel_noise, accel_noise)
            imu_msg.linear_acceleration.y = self.linear_acceleration[1] + random.uniform(-accel_noise, accel_noise)
            imu_msg.linear_acceleration.z = self.linear_acceleration[2] + random.uniform(-accel_noise, accel_noise)
            
            # Add covariance matrices (diagonal values only for simplicity)
            imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            
            self.imu_publisher.publish(imu_msg)

            # Log periodically
            if self.get_clock().now().nanoseconds % 1000000000 < 20000000:  # Every second
                self.get_logger().info(
                    f'Published sensor data: joints={len(joint_msg.name)}, IMU updated'
                )


    def main(args=None):
        rclpy.init(args=args)
        sensor_sim = HumanoidSensorSimulator()
        
        # Example of sending joint commands
        cmd_publisher = sensor_sim.create_publisher(Float64MultiArray, '/joint_commands', 10)
        
        # Schedule a sample joint command after 2 seconds
        def send_sample_command():
            cmd = Float64MultiArray()
            # Move all joints to a slightly different position
            cmd.data = [0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 0.2, 0.1]
            cmd_publisher.publish(cmd)
            sensor_sim.get_logger().info('Sent sample joint commands')
        
        # Send command after 2 seconds
        sensor_sim.create_timer(2.0, send_sample_command)
        
        rclpy.spin(sensor_sim)
        
        sensor_sim.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    // Initialize ROS 2 system for sensor simulation
    Initialize ROS
    Create Node "humanoid_sensor_simulator"
    
    // Create publishers for sensor data
    imu_publisher = CreatePublisher(Imu, "/imu/data", queue_size=10)
    joint_publisher = CreatePublisher(JointState, "/joint_states", queue_size=10)
    
    // Create subscriber for actuator commands
    joint_cmd_subscriber = CreateSubscriber(Float64MultiArray, "/joint_commands", command_handler)
    
    // Initialize joint state variables
    joint_names = [list of joint names for humanoid robot]
    joint_positions = [0.0 for each joint]
    joint_velocities = [0.0 for each joint]
    joint_efforts = [0.0 for each joint]
    
    // Initialize IMU state
    orientation = [0.0, 0.0, 0.0, 1.0]  // Quaternion
    angular_velocity = [0.0, 0.0, 0.0]
    linear_acceleration = [0.0, 0.0, 9.81]  // Gravity
    
    // Timer for publishing sensor data at 50 Hz
    CreateTimer(0.02 seconds, publish_sensor_data)
    
    Function command_handler(command_message):
        If command length matches joint count:
            For each joint:
                Update position based on command
                Calculate velocity from position change
        Else:
            Log error for mismatch
        End If
    End Function
    
    Function publish_sensor_data():
        // Publish joint states
        joint_msg = New JointState message
        joint_msg.header.stamp = Current time
        joint_msg.name = joint_names
        joint_msg.position = joint_positions
        joint_msg.velocity = joint_velocities
        joint_msg.effort = joint_efforts
        Publish joint_msg
        
        // Publish IMU data with simulated noise
        imu_msg = New Imu message
        imu_msg.header.stamp = Current time
        imu_msg.header.frame_id = "imu_link"
        
        // Add noise to orientation
        imu_msg.orientation = AddNoiseTo(orientation, 0.01)
        NormalizeQuaternion(imu_msg.orientation)
        
        // Add noise to angular velocity
        imu_msg.angular_velocity = AddNoiseTo(angular_velocity, 0.001)
        
        // Add noise to linear acceleration
        imu_msg.linear_acceleration = AddNoiseTo(linear_acceleration, 0.1)
        
        // Set covariance matrices
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        
        Publish imu_msg
    End Function
    ```
  </TabItem>
</Tabs>

This implementation simulates both IMU sensors and joint actuators for a humanoid robot. The code demonstrates how sensor data is published at regular intervals and how actuator commands are processed to update the simulated state of the robot.

## Visual Explanation

```
Humanoid Robot Sensor/Actuator Simulation Architecture

Real Robot                Simulation Environment
   |                            |
   |    Sensors:               |    Simulated Sensors:
   |    - IMU                 |    - IMU publisher
   |    - Cameras             |    - Camera publisher
   |    - LIDAR               |    - LIDAR publisher
   |    - Joint encoders      |    - JointState publisher
   |    - Force/Torque        |    - Force/Torque publisher
   |__________________________|__________________________
   |    Actuators:            |    Actuator Simulators:
   |    - Joint motors        |    - Joint command subscriber
   |    - Grippers            |    - Joint dynamics model
   |    - Displays           |    - Physics engine
   |__________________________|__________________________
   |    Controller           |    Controller
   |    (same code!)         |    (same code!)
   |__________________________|__________________________
   |    ROS 2 Middleware     |    ROS 2 Middleware
   |__________________________|__________________________

The same controller code can run with either real sensors/actuators
or simulated ones, enabling easy transfer from simulation to reality.
```

This architecture allows the same control algorithms to work with both simulated and real robots, making it easier to develop and test behaviors in simulation before deploying to real hardware.

## Checklist

- [x] Understand the types of sensors used in humanoid robots
- [x] Know how to simulate actuators in ROS 2
- [x] Can implement sensor publishers for simulated data
- [x] Understand the importance of simulation for humanoid robotics
- [ ] Self-assessment: Can create simulation-ready sensor models
- [ ] Self-assessment: Understand how to add realistic noise to sensor data