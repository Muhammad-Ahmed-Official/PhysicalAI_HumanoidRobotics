---
id: ch-003
title: "Chapter 1.3: Middleware Design Patterns"
description: "Understanding design patterns used in ROS 2 middleware for humanoid robotics applications"
tags: [ros2, middleware, design-patterns, architecture]
---

# Chapter 1.3: Middleware Design Patterns

## Introduction

Middleware design patterns in ROS 2 are essential for creating efficient and scalable robotic systems, particularly for humanoid robots which require complex coordination between many components. These patterns provide tested solutions to common communication and coordination challenges that arise in distributed robotic systems. Understanding these patterns helps developers build robust, maintainable robot applications.

## Learning Outcomes

- Students will understand common middleware design patterns in ROS 2
- Learners will be able to identify when to apply each pattern in robot development
- Readers will be familiar with patterns for distributed coordination in humanoid robotics
- Students will be able to implement communication patterns for robot subsystems

## Core Concepts

### Publisher-Subscriber Pattern
The most fundamental pattern in ROS 2, enabling decoupled communication where publishers send messages to topics without knowing the subscribers. This pattern is ideal for broadcasting sensor data, status updates, or other information that multiple nodes might need.

### Client-Service Pattern
Synchronous request-response communication where a client sends a request to a service and waits for a response. This pattern is suitable for operations that require a definitive response, such as configuration changes or computational tasks.

### Action Pattern
An extended service pattern that supports long-running tasks with feedback and goal preemption. This is essential for humanoid robotics operations like navigation, manipulation, or complex behaviors that take time to complete.

### Layered Architecture Pattern
Organizing the system into functional layers (e.g., sensing, perception, planning, execution) with well-defined interfaces between layers. This helps manage complexity in humanoid robots with many subsystems.

### Component-Based Pattern
Breaking complex systems into reusable, well-defined components that can be combined in various ways. This pattern promotes code reuse and modular design, essential for complex humanoid robot systems.

### Event-Driven Pattern
A pattern where system components react to events rather than continuously polling for information. This can improve performance and responsiveness in robotic systems with many concurrent activities.

## Simulation Walkthrough

Let's examine how to implement a layered architecture for a humanoid robot's walking controller:

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32
    from humanoid_msgs.srv import WalkCommand
    from humanoid_msgs.action import Walk
    import rclpy.action
    import math


    class WalkController(Node):
        """
        Implements layered architecture with:
        - Perception layer: processes sensor data
        - Planning layer: determines walking gait
        - Execution layer: controls actuators
        """
        
        def __init__(self):
            super().__init__('walk_controller')
            
            # Perception layer: subscribe to joint states
            self.joint_subscriber = self.create_subscription(
                JointState,
                '/joint_states',
                self.joint_state_callback,
                10
            )
            
            # Planning layer: service for walk commands
            self.service = self.create_service(
                WalkCommand,
                'start_walk',
                self.walk_command_callback
            )
            
            # Execution layer: publishers for actuator commands
            self.left_leg_publisher = self.create_publisher(
                Float32,
                '/left_leg/hip_position',
                10
            )
            self.right_leg_publisher = self.create_publisher(
                Float32,
                '/right_leg/hip_position',
                10
            )
            
            # Action server for long-running walk tasks
            self._action_server = rclpy.action.ActionServer(
                self,
                Walk,
                'perform_walk',
                self.execute_walk_callback
            )
            
            # State variables
            self.current_joint_states = None
            self.is_walking = False
            self.walk_params = {'step_length': 0.1, 'step_height': 0.05, 'step_duration': 1.0}
            
        def joint_state_callback(self, msg):
            """Perception layer: process joint state data"""
            self.current_joint_states = msg
            # Process sensor data to determine current state
            self.get_logger().info(f'Received joint states for {len(msg.name)} joints')
            
        def walk_command_callback(self, request, response):
            """Planning layer: process walk command"""
            self.get_logger().info(f'Received walk command: direction={request.direction}, distance={request.distance}')
            
            # Adjust walking parameters based on command
            if request.direction == 'forward':
                self.walk_params['step_length'] = 0.1
            elif request.direction == 'backward':
                self.walk_params['step_length'] = -0.1
            
            # Validate and execute walk command
            if request.distance > 0:
                response.success = True
                response.message = f'Starting walk: {request.direction} for {request.distance}m'
                self.start_walking()
            else:
                response.success = False
                response.message = 'Invalid distance'
                
            return response
            
        def execute_walk_callback(self, goal_handle):
            """Action server: long-running walk execution with feedback"""
            self.get_logger().info('Executing walk action...')
            
            # Calculate number of steps needed
            steps = int(goal_handle.request.distance / self.walk_params['step_length'])
            
            feedback_msg = Walk.Feedback()
            result = Walk.Result()
            
            for step in range(steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Walk canceled'
                    return result
                
                # Execute one step of walking
                self.execute_single_step(step)
                
                # Publish feedback
                feedback_msg.current_distance = step * self.walk_params['step_length']
                goal_handle.publish_feedback(feedback_msg)
                
                # Simulate step duration
                self.get_clock().sleep_for(rclpy.duration.Duration(seconds=self.walk_params['step_duration']))
            
            goal_handle.succeed()
            result.success = True
            result.message = f'Walk completed: {goal_handle.request.distance}m'
            return result
            
        def start_walking(self):
            """Execution layer: begin walking pattern"""
            self.is_walking = True
            self.get_logger().info('Starting walking pattern')
            
            # Implement walking gait pattern
            timer_period = 0.1  # seconds
            self.walk_timer = self.create_timer(timer_period, self.walk_step_callback)
            
        def execute_single_step(self, step_number):
            """Execute a single walking step"""
            # Calculate positions based on step number and gait pattern
            left_hip_angle = math.sin(step_number * 0.5) * 0.2
            right_hip_angle = math.sin(step_number * 0.5 + math.pi) * 0.2
            
            # Publish actuator commands
            left_msg = Float32()
            left_msg.data = left_hip_angle
            self.left_leg_publisher.publish(left_msg)
            
            right_msg = Float32()
            right_msg.data = right_hip_angle
            self.right_leg_publisher.publish(right_msg)
            
            self.get_logger().info(f'Executed step {step_number}: left={left_hip_angle:.2f}, right={right_hip_angle:.2f}')
            
        def walk_step_callback(self):
            """Timer callback for executing walking pattern"""
            if not self.is_walking:
                self.walk_timer.cancel()
                return
                
            # This could be replaced with more complex gait control
            self.get_logger().info('Walking step executed')


    def main(args=None):
        rclpy.init(args=args)
        walk_controller = WalkController()
        
        rclpy.spin(walk_controller)
        
        walk_controller.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    // Initialize ROS 2 system with layered architecture
    Initialize ROS
    Create Node "walk_controller"
    
    // Perception Layer: Subscribe to sensor data
    joint_subscriber = CreateSubscriber(JointState, "/joint_states", joint_callback)
    
    // Planning Layer: Service for walk commands
    walk_service = CreateService("start_walk", walk_command_handler)
    
    // Execution Layer: Publishers for actuator control
    left_leg_publisher = CreatePublisher(Float32, "/left_leg/hip_position", queue_size=10)
    right_leg_publisher = CreatePublisher(Float32, "/right_leg/hip_position", queue_size=10)
    
    // Action Server: Long-running walk operations
    walk_action = CreateActionServer("perform_walk", execute_walk_handler)
    
    Initialize walk parameters (step_length, step_height, step_duration)
    
    Function joint_callback(joint_data):
        Store current joint states
        Process sensor data
    End Function
    
    Function walk_command_handler(request, response):
        Process walk command
        Adjust walking parameters based on request
        Validate and start walk if valid
        Return response
    End Function
    
    Function execute_walk_handler(goal, feedback, result):
        Calculate steps needed for distance
        For each step:
            If goal canceled:
                Return canceled result
            Execute single walking step
            Publish feedback
            Wait for step duration
        End For
        Return success result
    End Function
    
    Function execute_single_step(step_number):
        Calculate actuator positions based on gait pattern
        Publish commands to left and right legs
        Log step execution
    End Function
    ```
  </TabItem>
</Tabs>

This example demonstrates a layered architecture for a humanoid robot walking controller, showing how different patterns and layers work together to create a complex behavior. The perception layer handles sensor data, the planning layer processes commands and determines the walking pattern, and the execution layer controls the actuators.

## Visual Explanation

```
Middleware Design Patterns in Humanoid Robotics

Layered Architecture:
+-------------------------+
|   Application Layer     |  <- High-level behaviors (walking, talking, etc.)
+-------------------------+
|    Planning Layer       |  <- Path planning, action planning
+-------------------------+
|  Perception/Control L.  |  <- Sensor fusion, low-level control
+-------------------------+
|     Middleware L.       |  <- ROS 2 communication patterns
+-------------------------+

Communication Patterns by Use Case:
- Topics: Sensor data broadcasting, status updates
- Services: Configuration, calibration, one-time commands
- Actions: Navigation, manipulation, complex behaviors
- Events: Error conditions, state changes

Component-Based Architecture:
[Head Component]  [Arm Component]  [Leg Component]
       |               |               |
       +--------[Central Controller]---+
               (coordinates actions)
```

These patterns help structure complex humanoid robot systems in a maintainable and scalable way, with clear separation of concerns between different system components.

## Checklist

- [x] Understand the layered architecture pattern
- [x] Know when to use different communication patterns
- [x] Can implement a component-based approach for robot systems
- [x] Understand how patterns improve system maintainability
- [ ] Self-assessment: Can identify appropriate patterns for different robot subsystems
- [ ] Self-assessment: Understand how patterns apply to humanoid-specific challenges