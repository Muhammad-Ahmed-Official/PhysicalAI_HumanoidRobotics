---
sidebar_label: 'Chapter 1.1: Introduction to ROS 2 Architecture'
---

# Chapter 1.1: Introduction to ROS 2 Architecture

## Introduction

Robot Operating System 2 (ROS 2) represents a significant evolution from its predecessor, designed specifically for production robotics applications with enhanced security, reliability, and real-time capabilities. Unlike ROS 1's reliance on a central master node, ROS 2 utilizes a distributed architecture based on the Data Distribution Service (DDS) middleware, enabling robust communication in complex robotic systems.

The ROS 2 architecture addresses the challenges of modern robotics, including multi-robot systems, real-time performance requirements, and security considerations essential for commercial deployment. This decentralized approach allows nodes to discover and communicate with each other dynamically without requiring a single point of failure.

This chapter introduces the fundamental architectural concepts of ROS 2, including nodes, topics, services, actions, and the underlying middleware that enables reliable communication in distributed robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the key architectural differences between ROS 1 and ROS 2
- Identify the core components of the ROS 2 architecture
- Explain the role of DDS middleware in ROS 2 communication
- Create and manage ROS 2 nodes in Python and C++
- Implement basic message passing using topics, services, and actions
- Evaluate the security and quality of service features in ROS 2

## Explanation

### ROS 2 Architecture Overview

The ROS 2 architecture is built on several key principles:

1. **Decentralized Communication**: Nodes communicate directly with each other using DDS without a central master.

2. **Quality of Service (QoS) Policies**: Configurable policies control communication behavior such as reliability, durability, and deadline requirements.

3. **Security**: Built-in security features for authentication, encryption, and access control.

4. **Multi-platform Support**: Native support for various operating systems, including Linux, Windows, and macOS, plus real-time systems.

5. **Real-time Support**: Architecture capable of real-time performance with appropriate DDS implementations.

### Core Components

The architecture consists of several key components:

- **Nodes**: Processes that perform computation, the fundamental unit of ROS 2 programs
- **Topics**: Named buses over which nodes exchange messages (publish/subscribe pattern)
- **Services**: Synchronous request/response communication pattern
- **Actions**: Asynchronous request/feedback/response pattern for long-running tasks
- **Parameters**: Configuration values that can be modified at runtime
- **DDS Middleware**: Implements the communication layer between nodes

### Quality of Service (QoS)

QoS policies provide fine-grained control over communication behavior:

- **Reliability**: Reliable (all messages delivered) or best-effort (messages may be dropped)
- **Durability**: Volatile (only new messages) or transient-local (including historical messages)
- **History**: Keep-all or keep-latest messages
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to determine if a publisher is active

## Example Walkthrough

Consider implementing a simple ROS 2 system for a humanoid robot that includes a perception node, a planning node, and an execution node.

**Step 1: Setting up the Development Environment**

```python
# Install ROS 2 (Humble Hawksbill for 22.04 LTS)
# sudo apt update
# sudo apt install ros-humble-desktop
# source /opt/ros/humble/setup.bash

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # Create publishers for processed perception data
        self.object_pub = self.create_publisher(
            PoseStamped, 
            'detected_object', 
            10
        )
        
        # Create subscription for raw camera data
        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10
        )
        
        # Timer for periodic processing
        self.timer = self.create_timer(0.1, self.process_perception)
        
        self.get_logger().info('Perception Node initialized')

    def camera_callback(self, msg):
        """Process incoming camera data"""
        self.get_logger().info(f'Received camera image with shape: {len(msg.data)} bytes')
        # In a real implementation, this would run object detection
        # For this example, we'll simulate a detected object
        
    def process_perception(self):
        """Process perception data and publish results"""
        # Simulate detecting an object
        obj_pose = PoseStamped()
        obj_pose.header.stamp = self.get_clock().now().to_msg()
        obj_pose.header.frame_id = 'base_link'
        obj_pose.pose.position.x = 1.0
        obj_pose.pose.position.y = 0.5
        obj_pose.pose.position.z = 0.0
        obj_pose.pose.orientation.w = 1.0
        
        self.object_pub.publish(obj_pose)

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        
        # Subscription for detected objects
        self.obj_sub = self.create_subscription(
            PoseStamped,
            'detected_object',
            self.object_callback,
            10
        )
        
        # Subscription for robot state
        self.state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.state_callback,
            10
        )
        
        # Publisher for planned trajectory
        self.trajectory_pub = self.create_publisher(
            String,  # In practice, this would be a more complex message type
            'planned_trajectory',
            10
        )
        
        self.get_logger().info('Planning Node initialized')

    def object_callback(self, msg):
        """Handle detected object information"""
        self.get_logger().info(f'Detected object at position: ({msg.pose.position.x}, {msg.pose.position.y})')
        # In a real implementation, this would plan a trajectory to the object
        self.plan_trajectory(msg.pose)

    def state_callback(self, msg):
        """Handle robot state information"""
        self.get_logger().info(f'Received joint states for {len(msg.name)} joints')

    def plan_trajectory(self, target_pose):
        """Plan trajectory to target pose"""
        # Simulate trajectory planning
        plan_msg = String()
        plan_msg.data = f"Go to position: ({target_pose.position.x}, {target_pose.position.y})"
        self.trajectory_pub.publish(plan_msg)

class ExecutionNode(Node):
    def __init__(self):
        super().__init__('execution_node')
        
        # Subscription for planned trajectories
        self.plan_sub = self.create_subscription(
            String,  # Simplified for example
            'planned_trajectory',
            self.plan_callback,
            10
        )
        
        self.get_logger().info('Execution Node initialized')

    def plan_callback(self, msg):
        """Execute planned trajectory"""
        self.get_logger().info(f'Executing: {msg.data}')
        # In a real implementation, this would execute the motion

def main(args=None):
    rclpy.init(args=args)
    
    # Create nodes
    perception_node = PerceptionNode()
    planning_node = PlanningNode()
    execution_node = ExecutionNode()
    
    # Run nodes
    try:
        rclpy.spin_multi_threaded([perception_node, planning_node, execution_node])
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        planning_node.destroy_node()
        execution_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 2: Creating a Custom Message Type**

First, create a package for custom message types:
```bash
mkdir -p ~/ros2_ws/src/custom_msgs
cd ~/ros2_ws/src/custom_msgs
mkdir msg
```

Create a Person.msg file:
```
# Custom message for detecting people
string name
float64 x_position
float64 y_position
float64 certainty
---
string status
```

Package.xml:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>custom_msgs</name>
  <version>0.0.0</version>
  <description>Custom message definitions for robotics project</description>
  <maintainer email="humanoid@todo.todo">humanoid</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>std_msgs</depend>

  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

CMakeLists.txt:
```cmake
cmake_minimum_required(VERSION 3.8)
project(custom_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/Person.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_package()
```

**Step 3: Using Services in ROS 2**

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 4: Using Actions in ROS 2**

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        
        # Create action server with reentrant callback group for concurrent execution
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup())

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()
            
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)
        
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')
        
        return result

def main(args=None):
    rclpy.init(args=args)
    
    fibonacci_action_server = FibonacciActionServer()
    
    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 5: Quality of Service Configuration**

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

class QoSDemoNode(Node):
    def __init__(self):
        super().__init__('qos_demo_node')
        
        # Create QoS profile for sensor data (best effort, volatile, keep last 10)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create QoS profile for critical data (reliable, transient local, keep all)
        critical_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
            depth=1
        )
        
        # Create publisher with specific QoS
        self.sensor_pub = self.create_publisher(Image, 'sensor_data', sensor_qos)
        self.critical_pub = self.create_publisher(String, 'critical_data', critical_qos)
        
        # Create subscription with matching QoS
        self.sensor_sub = self.create_subscription(
            Image, 'sensor_data', self.sensor_callback, sensor_qos
        )
        
        self.get_logger().info('QoS Demo Node initialized')

    def sensor_callback(self, msg):
        self.get_logger().info('Received sensor data with QoS configuration')

def main(args=None):
    rclpy.init(args=args)
    node = QoSDemoNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This implementation demonstrates the core architectural concepts of ROS 2, including nodes, topics, services, actions, and QoS policies that are essential for developing distributed robotic systems.

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROS 2 Architecture                           │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Perception    │    │    Planning     │    │   Execution     │ │
│  │     Node        │    │      Node       │    │      Node       │ │
│  │                 │    │                 │    │                 │ │
│  │ • Camera data   │    │ • Path planning │    │ • Motion        │ │
│  │ • Object det.   │    │ • Trajectory    │    │   control       │ │
│  │ • Sensor fusion │    │ • Collision     │    │ • Task exec.    │ │
│  └─────────────────┘    │   avoidance     │    └─────────────────┘ │
│         │                └─────────────────┘            │         │
│         │                       │                       │         │
│         ▼                       ▼                       ▼         │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    DDS Middleware (Communication)             ││
│  │  • Topic: /camera/image_raw  • Topic: /detected_object        ││
│  │  • Topic: /joint_states      • Topic: /planned_trajectory     ││
│  │  • QoS Policy: Reliability   • QoS Policy: Real-time          ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    ROS 2 Runtime Environment                  ││
│  │  • Node Management      • Lifecycle Management                ││
│  │  • Parameter Server     • Logging & Diagnostics               ││
│  │  • Action Server/Client • Service Server/Client               ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Understand the key architectural differences between ROS 1 and ROS 2
- [ ] Identify the core components of the ROS 2 architecture
- [ ] Explain the role of DDS middleware in ROS 2 communication
- [ ] Create and manage ROS 2 nodes in Python and C++
- [ ] Implement basic message passing using topics, services, and actions
- [ ] Evaluate the security and quality of service features in ROS 2