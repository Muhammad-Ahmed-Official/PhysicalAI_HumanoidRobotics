---
sidebar_label: 'Chapter 1.2: Nodes, Topics and Message Passing'
---

# Chapter 1.2: Nodes, Topics and Message Passing

## Introduction

Nodes, topics, and message passing form the foundational communication paradigm in ROS 2, enabling distributed computation across multiple processes and machines. The publish-subscribe pattern implemented through topics allows for loose coupling between components, promoting modularity and reusability in robotic systems. Unlike ROS 1's master-based architecture, ROS 2's decentralized approach allows nodes to discover and communicate with each other using the underlying DDS (Data Distribution Service) middleware.

Understanding these concepts is critical for developing effective robotic applications, as they determine how information flows through the system and how different components interact. This chapter explores the implementation of nodes, the design of topics and messages, and best practices for efficient and reliable message passing in complex robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:

- Create and configure ROS 2 nodes in both Python and C++
- Design custom message types for specific robotic applications
- Implement publisher-subscriber communication patterns
- Apply Quality of Service (QoS) policies for different data types
- Debug and monitor message passing in ROS 2 systems
- Optimize message passing for performance and reliability

## Explanation

### Nodes in ROS 2

A node is the fundamental computational unit in ROS 2, representing a single process that performs computation. Nodes are organized into packages and can be written in multiple languages (primarily Python and C++). Each node can contain:

- Publishers: Send messages to topics
- Subscribers: Receive messages from topics
- Services: Provide synchronous request-response functionality
- Action servers: Provide asynchronous goal-feedback-result functionality
- Parameters: Store configuration values

### Topics and Message Passing

Topics are named buses over which nodes exchange messages using a publish-subscribe model. This pattern provides loose coupling between publishers and subscribers:

- Publishers send messages without knowledge of subscribers
- Subscribers receive messages without knowledge of publishers
- Multiple publishers can publish to the same topic
- Multiple subscribers can subscribe to the same topic

### Message Types

Messages are the data structures sent over topics. They have:

- A fixed set of fields with specific data types
- Support for basic types (int, float, boolean, string) and compound types (arrays, nested messages)
- Language-specific bindings generated from .msg definition files
- Standard message types in common packages like std_msgs, sensor_msgs, geometry_msgs

### Quality of Service (QoS)

ROS 2 provides QoS policies to control message delivery characteristics:

- **Reliability**: Reliable (all messages delivered) or best-effort (messages may be dropped)
- **Durability**: Volatile (only new messages) or transient-local (including historical messages)
- **History**: Keep-all or keep-latest messages
- **Depth**: Number of messages to store if using keep-latest
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to determine if a publisher is active

## Example Walkthrough

Consider implementing a perception system for a humanoid robot that processes camera data, detects objects, and shares this information with other nodes.

**Step 1: Creating Custom Message Types**

First, let's create a custom message for detected objects:

```python
# Custom message definition (DetectedObject.msg)
# Time header.stamp
# string header.frame_id
# string object_class
# float64 confidence
# geometry_msgs/Pose pose
# float64[] bbox  # x, y, width, height
```

**Step 2: Implementing a Camera Perception Node**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import cv2
import numpy as np

# Assuming we have a custom message for detected objects
# This would be defined in a .msg file and generated
# For this example, we'll use a simple approach with standard messages

class CameraPerceptionNode(Node):
    def __init__(self):
        super().__init__('camera_perception_node')
        
        # Create publisher for processed images (with detections)
        self.image_pub = self.create_publisher(Image, 'camera/image_processed', 10)
        
        # Create publisher for detected objects
        self.object_pub = self.create_publisher(
            # In a real implementation, this would be a custom message type
            # For now using a standard message as placeholder
            Image,  # Placeholder - should be custom DetectedObjectArray message
            'detected_objects',
            10
        )
        
        # Create subscription for raw camera data
        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10  # Default QoS depth
        )
        
        # Create subscription for camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.info_callback,
            10
        )
        
        # Initialize OpenCV bridge
        self.bridge = CvBridge()
        self.camera_info = None
        
        # Timer for processing loop
        self.timer = self.create_timer(0.1, self.process_frame)  # Process at 10 Hz
        
        self.get_logger().info('Camera Perception Node initialized')

    def info_callback(self, msg):
        """Handle camera information"""
        self.camera_info = msg
        self.get_logger().info('Received camera calibration info')

    def camera_callback(self, msg):
        """Process incoming camera data"""
        # Store the latest image for processing in the timer callback
        self.last_image_msg = msg
        self.get_logger().debug(f'Received camera image: {msg.width}x{msg.height}')

    def process_frame(self):
        """Process the last received image frame"""
        if not hasattr(self, 'last_image_msg'):
            return  # No image received yet
        
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(self.last_image_msg, 'bgr8')
            
            # Perform object detection (simplified - in reality, this would use deep learning)
            processed_image = self.detect_objects(cv_image)
            
            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')
            processed_msg.header = self.last_image_msg.header
            self.image_pub.publish(processed_msg)
            
            # Publish detected objects (simplified)
            self.publish_detected_objects()
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

    def detect_objects(self, image):
        """Detect objects in the image (simplified implementation)"""
        # This is a simplified example - in practice, this would use 
        # a pre-trained neural network like YOLO or SSD
        processed = image.copy()
        
        # Detect red objects as example
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return processed

    def publish_detected_objects(self):
        """Publish detected objects (simplified for this example)"""
        # In a real implementation, this would publish actual detected objects
        # with their poses, classes, and confidence scores
        pass

def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Camera Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 3: Implementing a Navigation Planning Node**

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Time

class NavigationPlannerNode(Node):
    def __init__(self):
        super().__init__('navigation_planner')
        
        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'navigation/goal', 10)
        
        # Create subscriptions with different QoS profiles for different data types
        # Laser scan data with best-effort reliability (can drop frames)
        laser_qos = rclpy.qos.QoSProfile(
            depth=5,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            laser_qos
        )
        
        # Goal commands with reliable delivery
        goal_qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal_command',
            self.goal_command_callback,
            goal_qos
        )
        
        # Create timer for navigation planning
        self.nav_timer = self.create_timer(0.05, self.navigation_callback)  # 20 Hz
        
        self.current_goal = None
        self.obstacle_detected = False
        
        self.get_logger().info('Navigation Planner Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Check for obstacles in front of the robot
        min_distance = min(msg.ranges)
        
        if min_distance < 0.5:  # Less than 50 cm to obstacle
            self.obstacle_detected = True
            self.get_logger().warn('Obstacle detected! Stopping robot.')
        else:
            self.obstacle_detected = False

    def goal_command_callback(self, msg):
        """Receive navigation goal commands"""
        self.current_goal = msg
        self.get_logger().info(f'New goal received: ({msg.pose.position.x}, {msg.pose.position.y})')

    def navigation_callback(self):
        """Main navigation planning loop"""
        if self.current_goal is None:
            # No goal set, stop the robot
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        if self.obstacle_detected:
            # Stop if obstacle is detected
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        # Simple navigation behavior (in reality, you'd implement path planning)
        # Generate simple movement command to approach goal
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward at 0.2 m/s
        cmd.angular.z = 0.0  # No rotation
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Navigation Planner Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 4: Implementing a Robot Control Node**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create publishers for different control interfaces
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)
        
        # Create subscriptions for commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            'joint_commands',
            self.joint_cmd_callback,
            10
        )
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_callback)  # 100 Hz
        
        # Robot state
        self.cmd_vel = Twist()
        self.joint_positions = {}
        self.target_joint_positions = {}
        
        # Initialize joint positions
        self.initialize_joints()
        
        self.get_logger().info('Robot Controller Node initialized')

    def initialize_joints(self):
        """Initialize robot joint positions"""
        # For a humanoid robot, initialize with some default positions
        joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]
        
        initial_positions = [0.0] * len(joint_names)
        
        for name, pos in zip(joint_names, initial_positions):
            self.joint_positions[name] = pos
            self.target_joint_positions[name] = pos

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.cmd_vel = msg
        self.get_logger().debug(f'Received velocity command: linear={msg.linear.x}, angular={msg.angular.z}')

    def joint_cmd_callback(self, msg):
        """Handle joint position commands"""
        for name, pos in zip(msg.name, msg.position):
            if name in self.target_joint_positions:
                self.target_joint_positions[name] = pos
        self.get_logger().debug(f'Received {len(msg.name)} joint commands')

    def control_callback(self):
        """Main control loop"""
        # Update joint positions based on commands
        self.update_joint_positions()
        
        # Publish updated joint states
        self.publish_joint_states()
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Controlling {len(self.joint_positions)} joints, moving with linear: {self.cmd_vel.linear.x}, angular: {self.cmd_vel.angular.z}"
        self.status_pub.publish(status_msg)

    def update_joint_positions(self):
        """Update joint positions based on targets"""
        # Simple proportional control to reach target positions
        for joint_name in self.joint_positions:
            if joint_name in self.target_joint_positions:
                current_pos = self.joint_positions[joint_name]
                target_pos = self.target_joint_positions[joint_name]
                
                # Proportional controller
                error = target_pos - current_pos
                step_size = 0.01  # Radians per control cycle
                
                if abs(error) > step_size:
                    step = np.sign(error) * step_size
                    self.joint_positions[joint_name] += step
                else:
                    self.joint_positions[joint_name] = target_pos

    def publish_joint_states(self):
        """Publish updated joint states"""
        msg = JointState()
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Robot Controller Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 5: Creating a Launch File for the System**

```python
# camera_perception_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        
        # Camera Perception Node
        Node(
            package='robot_perception',
            executable='camera_perception_node',
            name='camera_perception',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'image_topic': 'camera/image_raw'},
                {'publish_rate': 10}
            ],
            remappings=[
                ('camera/image_raw', '/sim_camera/image_raw')
            ],
            output='screen'
        ),
        
        # Navigation Planner Node
        Node(
            package='robot_navigation',
            executable='navigation_planner_node',
            name='nav_planner',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'max_linear_speed': 0.5},
                {'max_angular_speed': 1.0}
            ],
            output='screen'
        ),
        
        # Robot Controller Node
        Node(
            package='robot_control',
            executable='robot_controller_node',
            name='robot_controller',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'control_frequency': 100.0}
            ],
            output='screen'
        )
    ])
```

## Best Practices for Message Passing

### QoS Configuration Guidelines

1. **Sensor Data**: Use best-effort reliability with volatile durability for high-frequency sensor streams
2. **Critical Commands**: Use reliable delivery with appropriate durability for commands
3. **Configuration Data**: Use reliable delivery with transient-local durability to ensure configuration is received by late-joining subscribers
4. **Debug/Logging**: Use best-effort delivery with volatile durability to minimize overhead

### Message Design Principles

1. **Keep Messages Small**: Large messages can impact performance and real-time behavior
2. **Separate Time-Critical from Non-Critical**: Group messages by their timing requirements
3. **Use Semantic Field Names**: Field names should clearly indicate their purpose
4. **Consider Message Evolution**: Design messages anticipating potential future extensions

### Performance Optimization

1. **Monitor Network Usage**: Use tools like `ros2 topic hz` to monitor message rates
2. **Optimize Message Rates**: Don't publish at higher rates than required by subscribers
3. **Use Appropriate QoS**: Match QoS policies to the actual requirements of your application

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ROS 2 Message Passing                             │
│                                                                     │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │   Camera Node   │              │  Perception     │              │
│  │                 │─────────────▶│    Node         │              │
│  │ • Publish raw   │  Topic:      │                 │              │
│  │   images        │  /camera/    │ • Subscribe     │              │
│  │                 │  image_raw   │   /camera/      │              │
│  └─────────────────┘              │   image_raw     │              │
│                                   │ • Publish       │              │
│  ┌─────────────────┐              │   /detected_    │              │
│  │  Navigation     │  Topic:      │   objects       │              │
│  │    Node         │  /detected_  │                 │              │
│  │                 │  objects     │                 │              │
│  │ • Subscribe     │              │                 │              │
│  │   /detected_    │◀─────────────│                 │              │
│  │   objects       │              └─────────────────┘              │
│  │ • Publish       │                                                │
│  │   /cmd_vel      │                                                │
│  └─────────────────┘                                                │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────┐                                                │
│  │  Controller     │                                                │
│  │    Node         │                                                │
│  │                 │                                                │
│  │ • Subscribe     │                                                │
│  │   /cmd_vel      │                                                │
│  │ • Publish       │                                                │
│  │   /joint_states │                                                │
│  └─────────────────┘                                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                 DDS Communication Layer                         ││
│  │  • Message Routing    • Quality of Service                     ││
│  │  • Topic Discovery    • Network Transport                      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Create and configure ROS 2 nodes in both Python and C++
- [ ] Design custom message types for specific robotic applications
- [ ] Implement publisher-subscriber communication patterns
- [ ] Apply Quality of Service (QoS) policies for different data types
- [ ] Debug and monitor message passing in ROS 2 systems
- [ ] Optimize message passing for performance and reliability