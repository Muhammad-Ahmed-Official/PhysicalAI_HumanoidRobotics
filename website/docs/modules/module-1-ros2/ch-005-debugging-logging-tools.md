---
id: ch-005
title: "Chapter 1.5: Debugging and Logging Tools"
description: "Essential debugging and logging techniques for ROS 2 humanoid robotics development"
tags: [ros2, debugging, logging, tools, development]
---

# Chapter 1.5: Debugging and Logging Tools

## Introduction

Debugging and logging are critical aspects of developing reliable humanoid robotics applications in ROS 2. With complex systems involving multiple nodes, sensors, and actuators, proper debugging techniques and comprehensive logging help identify issues quickly and maintain system reliability. This chapter explores the debugging and logging tools available in ROS 2 and best practices for using them effectively in humanoid robotics applications.

## Learning Outcomes

- Students will understand the importance of proper logging in robotics applications
- Learners will be able to use ROS 2's built-in logging and debugging tools
- Readers will be familiar with debugging techniques for distributed robot systems
- Students will be able to implement appropriate logging levels and formats

## Core Concepts

### ROS 2 Logging System
ROS 2 provides a comprehensive logging system with multiple severity levels:
- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about system operation
- **WARN**: Indications of potential issues
- **ERROR**: Error events that allow the system to continue
- **FATAL**: Critical errors that cause system termination

The logging system automatically includes timestamps, node names, and other contextual information to aid in debugging distributed systems.

### Common Debugging Challenges in Humanoid Robotics
Humanoid robotics systems present unique debugging challenges:
- **Distributed nature**: Issues may span multiple nodes and processes
- **Real-time constraints**: Timing-related issues are common
- **Sensor fusion**: Debugging complex interactions between multiple sensors
- **Safety considerations**: Debugging must not compromise robot safety
- **Hardware dependencies**: Distinguishing between software and hardware issues

### ROS 2 Debugging Tools
Key tools for debugging ROS 2 applications:
- **ros2 topic**: Monitor, publish, and echo topic data
- **ros2 service**: Call services and check service status
- **ros2 action**: Monitor and send action goals
- **rqt**: GUI-based tools for visualization and debugging
- **rviz**: 3D visualization of robot state and sensor data
- **ros2 doctor**: Diagnostic tool for ROS 2 system health

### Best Practices for Logging
Effective logging practices in robotics applications:
- Use appropriate log levels consistently
- Include contextual information in log messages
- Avoid excessive logging that impacts performance
- Structure logs for easy parsing and analysis
- Include timestamps and unique identifiers for correlation

### Debugging Strategies
Approaches for debugging complex robotic systems:
- **Component-based debugging**: Test individual components in isolation
- **Integration testing**: Verify interactions between components
- **Simulation debugging**: Use simulation to reproduce and debug issues
- **Hardware-in-the-loop**: Gradually integrate with real hardware
- **Remote debugging**: Debug robots deployed in different locations

## Simulation Walkthrough

Let's look at how to implement effective logging and debugging in a humanoid robot controller:

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32
    from sensor_msgs.msg import JointState
    from builtin_interfaces.msg import Time
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    import traceback
    import sys
    
    # Custom logging formatter example
    class HumanoidLogger:
        def __init__(self, node):
            self.node = node
            
        def debug(self, msg, *args):
            self.node.get_logger().debug(msg.format(*args) if args else msg)
            
        def info(self, msg, *args):
            self.node.get_logger().info(msg.format(*args) if args else msg)
            
        def warn(self, msg, *args):
            self.node.get_logger().warn(msg.format(*args) if args else msg)
            
        def error(self, msg, *args):
            self.node.get_logger().error(msg.format(*args) if args else msg)
            
        def fatal(self, msg, *args):
            self.node.get_logger().fatal(msg.format(*args) if args else msg)


    class HumanoidController(Node):
        """
        Humanoid robot controller with comprehensive logging and debugging capabilities
        """
        
        def __init__(self):
            super().__init__('humanoid_controller')
            
            # Initialize custom logger
            self.logger = HumanoidLogger(self)
            
            # Create QoS profile for reliable communication
            qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            
            # Publishers for various systems
            self.status_publisher = self.create_publisher(String, 'robot_status', qos_profile)
            self.error_publisher = self.create_publisher(String, 'robot_errors', qos_profile)
            self.joint_publisher = self.create_publisher(JointState, 'monitor_joint_states', qos_profile)
            
            # Subscriber for joint states
            self.joint_subscriber = self.create_subscription(
                JointState,
                'joint_states',
                self.joint_state_callback,
                qos_profile
            )
            
            # Timer for periodic status updates
            self.status_timer = self.create_timer(1.0, self.publish_status)
            
            # Initialize robot state
            self.robot_state = {
                'initialized': False,
                'current_behavior': 'idle',
                'joint_positions': {},
                'error_count': 0,
                'last_error': None
            }
            
            self.logger.info("Humanoid controller initialized")
            
            # Initialize the robot (with error handling)
            try:
                self.initialize_robot()
                self.robot_state['initialized'] = True
                self.logger.info("Robot initialized successfully")
            except Exception as e:
                self.handle_error(f"Failed to initialize robot: {str(e)}")
                
        def initialize_robot(self):
            """Initialize the robot systems with proper error handling"""
            self.logger.info("Initializing robot systems...")
            
            # Simulate initialization steps
            try:
                # Initialize joint controllers
                self.logger.debug("Initializing joint controllers")
                # In a real robot, this would communicate with actual hardware
                
                # Initialize sensor systems
                self.logger.debug("Initializing sensor systems")
                # In a real robot, this would set up sensor connections
                
                # Run self-tests
                self.logger.debug("Running self-tests")
                self.run_self_tests()
                
                self.logger.info("Robot initialization completed")
                
            except Exception as e:
                self.logger.error(f"Initialization error: {str(e)}")
                raise
                
        def run_self_tests(self):
            """Run comprehensive self-tests"""
            self.logger.info("Running self-tests...")
            
            # Test joint movement
            try:
                self.test_joint_movement()
                self.logger.info("Joint movement test passed")
            except Exception as e:
                self.logger.error(f"Joint movement test failed: {str(e)}")
                raise
                
            # Test sensor connectivity
            try:
                self.test_sensor_connectivity()
                self.logger.info("Sensor connectivity test passed")
            except Exception as e:
                self.logger.error(f"Sensor connectivity test failed: {str(e)}")
                raise
                
            self.logger.info("All self-tests passed")
            
        def test_joint_movement(self):
            """Test joint movement capabilities"""
            self.logger.debug("Testing joint movement...")
            
            # Simulate testing each joint
            joint_names = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
            
            for joint_name in joint_names:
                try:
                    # In a real system, this would command the joint to move
                    self.logger.debug(f"Testing joint: {joint_name}")
                    # Simulate successful movement
                    self.logger.debug(f"Joint {joint_name} moved successfully")
                except Exception as e:
                    self.logger.error(f"Error testing joint {joint_name}: {str(e)}")
                    raise
                    
        def test_sensor_connectivity(self):
            """Test sensor connectivity"""
            self.logger.debug("Testing sensor connectivity...")
            
            # Simulate checking sensor connections
            sensors = ['imu', 'lidar', 'camera_front', 'camera_rear']
            
            for sensor in sensors:
                # In a real system, this would verify sensor communication
                self.logger.debug(f"Sensor {sensor} connected and functional")
                
        def joint_state_callback(self, msg):
            """Handle joint state messages with debugging info"""
            try:
                self.logger.debug(f"Received joint state with {len(msg.name)} joints")
                
                # Update internal joint state
                for i, name in enumerate(msg.name):
                    if i < len(msg.position):
                        self.robot_state['joint_positions'][name] = msg.position[i]
                        
                # Log potentially problematic joint states
                for i, name in enumerate(msg.name):
                    if i < len(msg.position):
                        pos = msg.position[i]
                        # Log if joint position is out of expected range
                        if abs(pos) > 3.0:  # radians
                            self.logger.warn(
                                "Joint {} position {} exceeds normal range (>{})".format(
                                    name, pos, 3.0
                                )
                            )
                            
                # Publish joint states to monitoring topic for debugging
                self.joint_publisher.publish(msg)
                
            except Exception as e:
                self.handle_error(f"Error processing joint state: {str(e)}")
                # Log the full traceback for debugging
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                
        def publish_status(self):
            """Publish periodic status information for debugging"""
            try:
                status_msg = String()
                status_msg.data = f"Behavior: {self.robot_state['current_behavior']}, Errors: {self.robot_state['error_count']}"
                
                self.status_publisher.publish(status_msg)
                
                # Log status periodically
                self.logger.info(
                    "Status - Behavior: {}, Errors: {}, Joints: {}".format(
                        self.robot_state['current_behavior'],
                        self.robot_state['error_count'],
                        len(self.robot_state['joint_positions'])
                    )
                )
                
            except Exception as e:
                self.handle_error(f"Error publishing status: {str(e)}")
                
        def handle_error(self, error_msg):
            """Handle errors with appropriate logging and response"""
            self.robot_state['error_count'] += 1
            self.robot_state['last_error'] = error_msg
            
            self.logger.error(f"Error #{self.robot_state['error_count']}: {error_msg}")
            
            # Publish error for other nodes to monitor
            error_msg_obj = String()
            error_msg_obj.data = error_msg
            self.error_publisher.publish(error_msg_obj)
            
            # In a real system, you might trigger safe behaviors here
            # e.g., move to safe position, stop motion, etc.
            
        def execute_behavior(self, behavior_name):
            """Execute a robot behavior with comprehensive logging"""
            self.logger.info(f"Starting behavior: {behavior_name}")
            
            try:
                self.robot_state['current_behavior'] = behavior_name
                
                # Log behavior start
                self.logger.debug(f"Behavior {behavior_name} started")
                
                # Execute behavior logic
                if behavior_name == 'walk_forward':
                    self.execute_walk_forward()
                elif behavior_name == 'turn_left':
                    self.execute_turn_left()
                elif behavior_name == 'idle':
                    self.execute_idle()
                else:
                    self.logger.warn(f"Unknown behavior requested: {behavior_name}")
                    return False
                    
                self.logger.info(f"Behavior {behavior_name} completed successfully")
                return True
                
            except Exception as e:
                self.handle_error(f"Error in behavior {behavior_name}: {str(e)}")
                return False
                
        def execute_walk_forward(self):
            """Execute walking forward behavior with detailed logging"""
            self.logger.debug("Executing walk forward behavior")
            # In a real system, this would implement the walking gait
            
        def execute_turn_left(self):
            """Execute turning left behavior with detailed logging"""
            self.logger.debug("Executing turn left behavior")
            # In a real system, this would implement the turning behavior
            
        def execute_idle(self):
            """Execute idle behavior with detailed logging"""
            self.logger.debug("Executing idle behavior")
            # In a real system, this would maintain idle state


    def main(args=None):
        rclpy.init(args=args)
        
        try:
            controller = HumanoidController()
            
            # Example of triggering different behaviors
            def trigger_behaviors():
                controller.execute_behavior('idle')
                controller.execute_behavior('walk_forward')
                controller.logger.info("All behaviors triggered")
                
            # Trigger behaviors after 3 seconds
            controller.create_timer(3.0, trigger_behaviors)
            
            rclpy.spin(controller)
            
        except KeyboardInterrupt:
            controller.get_logger().info("Interrupted by user")
        except Exception as e:
            print(f"Unhandled exception: {e}")
            print(traceback.format_exc())
        finally:
            controller.destroy_node()
            rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    // Initialize ROS 2 system with debugging capabilities
    Initialize ROS
    Create Node "humanoid_controller"
    
    // Create custom logger with different levels
    logger = CreateCustomLogger(node)
    
    // Create publishers for status and error reporting
    status_publisher = CreatePublisher(String, "robot_status", qos_profile)
    error_publisher = CreatePublisher(String, "robot_errors", qos_profile)
    monitor_publisher = CreatePublisher(JointState, "monitor_joint_states", qos_profile)
    
    // Create subscriber for joint states
    joint_subscriber = CreateSubscriber(JointState, "joint_states", joint_callback)
    
    // Initialize robot state variables
    robot_state = {
        initialized: false,
        current_behavior: "idle",
        joint_positions: {},
        error_count: 0,
        last_error: null
    }
    
    Function initialize_robot():
        Log info: "Initializing robot systems..."
        
        Try:
            Log debug: "Initializing joint controllers"
            // Initialize joint controllers
            
            Log debug: "Initializing sensor systems"
            // Initialize sensor systems
            
            run_self_tests()
            
            Log info: "Robot initialization completed"
            robot_state.initialized = true
        Catch exception:
            Log error: "Failed to initialize robot: " + exception
            Handle error(exception)
        End Try
    End Function
    
    Function run_self_tests():
        Log info: "Running self-tests..."
        
        Try:
            test_joint_movement()
            Log info: "Joint movement test passed"
        Catch exception:
            Log error: "Joint movement test failed: " + exception
            Re-throw exception
        End Try
        
        Try:
            test_sensor_connectivity()
            Log info: "Sensor connectivity test passed"
        Catch exception:
            Log error: "Sensor connectivity test failed: " + exception
            Re-throw exception
        End Try
    End Function
    
    Function joint_callback(joint_message):
        Try:
            Log debug: "Received joint state with " + joint_message.name.length + " joints"
            
            For each joint in joint_message:
                Update robot_state.joint_positions
            End For
            
            // Check for problematic joint states
            For each joint in joint_message:
                If position out of range:
                    Log warn: "Joint " + joint + " position exceeds normal range"
                End If
            End For
            
            Publish joint_message to monitoring topic
        Catch exception:
            Handle error("Error processing joint state: " + exception)
            Log error: Full traceback of exception
        End Try
    End Function
    
    Function handle_error(error_message):
        robot_state.error_count = robot_state.error_count + 1
        robot_state.last_error = error_message
        Log error: "Error #" + robot_state.error_count + ": " + error_message
        
        error_msg = Create String message
        error_msg.data = error_message
        Publish error_msg to error_publisher
    End Function
    
    Function execute_behavior(behavior_name):
        Log info: "Starting behavior: " + behavior_name
        robot_state.current_behavior = behavior_name
        
        Try:
            If behavior_name == "walk_forward":
                execute_walk_forward()
            Else If behavior_name == "turn_left":
                execute_turn_left()
            Else If behavior_name == "idle":
                execute_idle()
            Else:
                Log warn: "Unknown behavior requested: " + behavior_name"
                Return false
            End If
            
            Log info: "Behavior " + behavior_name + " completed successfully"
            Return true
        Catch exception:
            Handle error("Error in behavior " + behavior_name + ": " + exception)
            Return false
        End Try
    End Function
    ```
  </TabItem>
</Tabs>

This implementation demonstrates comprehensive debugging and logging techniques for a humanoid robot controller, including error handling, status reporting, and structured logging at different levels.

## Visual Explanation

```
Debugging and Logging Architecture in ROS 2

+---------------------+    +---------------------+    +---------------------+
|   Robot Node        |    |   Logging System    |    |   Monitoring Tools  |
|                     |    |                     |    |                     |
|  +---------------+  |    |  +---------------+  |    |  +---------------+  |
|  | Debug Logic   |  |<-->|  | ROS2 Logger |  |<-->|  | rqt_console   |  |
|  | - Error       |  |    |  | - DEBUG     |  |    |  | rviz          |  |
|  |   Handling    |  |    |  | - INFO      |  |    |  | ros2 topic    |  |
|  | - Status      |  |    |  | - WARN      |  |    |  | ros2 service  |  |
|  |   Reporting   |  |    |  | - ERROR     |  |    |  +---------------+  |
|  | - State       |  |    |  | - FATAL     |  |    |                     |
|  |   Monitoring  |  |    |  +---------------+  |    |                     |
|  +---------------+  |    |                     |    |                     |
+---------------------+    +---------------------+    +---------------------+
         |                           |                          |
         | rosout topic              |                         |
         |-------------------------> |                         |
         |                           |                         |
+---------------------+    +---------------------+    +---------------------+
|   Error Publisher   |    |   Status Publisher  |    |   Diagnostic Tools  |
|                     |    |                     |    |                     |
|  +---------------+  |    |  +---------------+  |    |  +---------------+  |
|  | Publish       |  |    |  | Publish       |  |    |  | ros2 doctor   |  |
|  | Error Info    |  |    |  | Status Info   |  |    |  | ros2 run      |  |
|  | to /robot_    |  |    |  | to /robot_    |  |    |  | ros2 param    |  |
|  | _errors       |  |    |  | _status       |  |    |  +---------------+  |
|  +---------------+  |    |  +---------------+  |    |                     |
+---------------------+    +---------------------+    +---------------------+

The architecture connects all robot nodes to centralized logging and monitoring tools,
enabling comprehensive debugging and diagnostic capabilities.
```

This architecture shows how all robot nodes connect to centralized logging and monitoring tools, enabling comprehensive debugging and diagnostic capabilities for humanoid robot systems.

## Checklist

- [x] Understand the ROS 2 logging system and its levels
- [x] Know how to implement proper error handling in robot systems
- [x] Can use ROS 2's debugging tools effectively
- [x] Understand debugging strategies for distributed robot systems
- [ ] Self-assessment: Can implement structured logging in robot applications
- [ ] Self-assessment: Understand how to correlate logs from different nodes