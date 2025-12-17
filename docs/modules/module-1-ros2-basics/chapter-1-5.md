---
sidebar_label: 'Chapter 1.5: System Integration and Testing'
---

# Chapter 1.5: System Integration and Testing

## Introduction

System integration and testing form the cornerstone of deploying reliable robotic systems in real-world applications. This process involves combining individual ROS 2 nodes, packages, and subsystems into a cohesive whole and verifying that they function correctly together. Effective integration and testing strategies ensure that robotic systems behave as expected under various operating conditions and can gracefully handle failures and edge cases.

In ROS 2, system integration encompasses not only connecting nodes via topics, services, and actions but also coordinating timing, managing resources, and validating the overall system architecture. Testing in ROS 2 involves multiple levels: unit testing of individual components, integration testing of node interactions, and system testing of complete robotic behaviors.

This chapter covers comprehensive approaches to integrating ROS 2 components and testing robotic systems at various levels of complexity, with particular focus on humanoid robotics applications where safety and reliability are paramount.

## Learning Objectives

By the end of this chapter, you will be able to:

- Design integration strategies for complex ROS 2 systems
- Implement comprehensive testing frameworks for ROS 2 nodes and systems
- Perform unit, integration, and system testing of robotic applications
- Create launch files for system-wide testing environments
- Monitor and debug integrated ROS 2 systems
- Evaluate system performance and reliability metrics

## Explanation

### System Integration Strategies

Integration in ROS 2 involves several key strategies:

1. **Component Integration**: Combining individual nodes that perform specific functions
2. **Architecture Integration**: Ensuring proper communication patterns and data flow
3. **Timing Integration**: Coordinating timing-sensitive operations across nodes
4. **Resource Integration**: Managing shared resources like computational power and network bandwidth
5. **Safety Integration**: Ensuring safety-critical paths are properly handled

### Testing Levels in ROS 2

ROS 2 testing follows the standard software testing hierarchy:

- **Unit Testing**: Testing individual functions, classes, and nodes in isolation
- **Integration Testing**: Testing interactions between nodes and components
- **System Testing**: Testing complete robotic behaviors and capabilities
- **Regression Testing**: Ensuring new changes don't break existing functionality

### Testing Approaches

1. **Simulation-based Testing**: Using Gazebo or other simulators for testing without physical hardware
2. **Hardware-in-the-loop Testing**: Testing with actual hardware components
3. **Model-in-the-loop Testing**: Testing with mathematical models of physical systems
4. **Continuous Integration**: Automated testing pipelines for code changes

### Performance Evaluation

Assessing ROS 2 systems involves multiple dimensions:

- **Latency**: Time delays in communication and processing
- **Throughput**: Data processing capacity
- **Reliability**: Consistency and fault tolerance
- **Resource Usage**: CPU, memory, and network utilization
- **Real-time Performance**: Ability to meet timing constraints

## Example Walkthrough

Consider implementing a comprehensive testing framework for a humanoid robot system with perception, planning, and control components.

**Step 1: Creating a Test Node for System Verification**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time
from threading import Thread

class SystemIntegrationTestNode(Node):
    def __init__(self):
        super().__init__('system_integration_test_node')
        
        # QoS profile for testing
        self.test_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        # Publishers for test commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', self.test_qos)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', self.test_qos)
        
        # Subscribers for system status
        self.status_sub = self.create_subscription(
            String, 'robot_status', self.status_callback, self.test_qos
        )
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, self.test_qos
        )
        self.odom_sub = self.create_subscription(
            String, 'odometry', self.odom_callback, self.test_qos  # Placeholder
        )
        
        # Service clients for testing
        self.enable_client = self.create_client(SetBool, 'robot/enable')
        
        # Test state tracking
        self.test_results = {}
        self.current_test = None
        self.system_status = 'unknown'
        self.joint_states = None
        self.odometry_data = None
        
        # Test timer
        self.test_timer = self.create_timer(1.0, self.run_tests)
        
        self.get_logger().info('System Integration Test Node initialized')

    def status_callback(self, msg):
        """Handle system status updates."""
        self.system_status = msg.data
        self.get_logger().debug(f'System status: {msg.data}')

    def joint_callback(self, msg):
        """Handle joint state updates."""
        self.joint_states = msg
        if self.current_test == 'joint_integration':
            self.test_results['joint_integration'] = {
                'timestamp': time.time(),
                'joint_count': len(msg.name),
                'success': len(msg.name) > 0
            }

    def odom_callback(self, msg):
        """Handle odometry updates."""
        self.odometry_data = msg.data

    def run_tests(self):
        """Run various system integration tests."""
        # Test 1: Check system status
        self.test_system_status()
        
        # Test 2: Check joint state publication
        self.test_joint_states()
        
        # Test 3: Check service availability
        self.test_services()
        
        # Test 4: Check communication patterns
        self.test_communication()
        
        # Log results periodically
        self.log_test_results()

    def test_system_status(self):
        """Test that system is reporting status."""
        test_name = 'system_status_check'
        result = {
            'timestamp': time.time(),
            'success': self.system_status != 'unknown',
            'details': f'System status: {self.system_status}'
        }
        self.test_results[test_name] = result
        
        if result['success']:
            self.get_logger().info(f'{test_name}: PASSED - {result["details"]}')
        else:
            self.get_logger().warn(f'{test_name}: FAILED - {result["details"]}')

    def test_joint_states(self):
        """Test that joint states are being published."""
        test_name = 'joint_state_check'
        if self.joint_states:
            result = {
                'timestamp': time.time(),
                'success': len(self.joint_states.name) > 0,
                'details': f'Joint count: {len(self.joint_states.name)}'
            }
        else:
            result = {
                'timestamp': time.time(),
                'success': False,
                'details': 'No joint states received'
            }
        
        self.test_results[test_name] = result
        
        if result['success']:
            self.get_logger().info(f'{test_name}: PASSED - {result["details"]}')
        else:
            self.get_logger().warn(f'{test_name}: FAILED - {result["details"]}')

    def test_services(self):
        """Test that critical services are available."""
        test_name = 'service_availability'
        
        # Check if service client is ready
        if self.enable_client.service_is_ready():
            # Try a simple service call
            req = SetBool.Request()
            req.data = True
            
            future = self.enable_client.call_async(req)
            # Note: In real implementation, we'd wait for the result properly
            result = {
                'timestamp': time.time(),
                'success': True,
                'details': 'Service call initiated'
            }
        else:
            result = {
                'timestamp': time.time(),
                'success': False,
                'details': 'Service not available'
            }
        
        self.test_results[test_name] = result

    def test_communication(self):
        """Test basic communication patterns."""
        test_name = 'communication_check'
        
        # Send a simple command to test communication
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        result = {
            'timestamp': time.time(),
            'success': True,  # Assume success for this test
            'details': 'Command published to cmd_vel'
        }
        
        self.test_results[test_name] = result

    def log_test_results(self):
        """Log a summary of test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        
        self.get_logger().info(f'Test Summary: {passed_tests}/{total_tests} tests passed')

def main(args=None):
    rclpy.init(args=args)
    node = SystemIntegrationTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down System Integration Test Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 2: Implementing Unit Tests for ROS 2 Nodes**

```python
# test_perception_node.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2

class MockPerceptionNode(Node):
    """Mock node for testing perception functionality."""
    
    def __init__(self):
        super().__init__('mock_perception_node')
        
        # Publishers and subscribers
        self.processed_pub = self.create_publisher(String, 'processed_output', 10)
        self.image_sub = self.create_subscription(
            Image, 'test_input', self.image_callback, 10
        )
        
        # Internal state
        self.last_processed = None
        self.bridge = CvBridge()
        
    def image_callback(self, msg):
        """Process incoming image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # Simplified processing - detect red pixels
            processed_result = self.process_image(cv_image)
            
            result_msg = String()
            result_msg.data = f"Detected {processed_result} red pixels"
            self.processed_pub.publish(result_msg)
            self.last_processed = processed_result
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def process_image(self, image):
        """Process image to detect red pixels."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        return np.sum(mask > 0)

class TestPerceptionNode(unittest.TestCase):
    """Unit tests for perception node."""
    
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures."""
        self.node = MockPerceptionNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        """Tear down test fixtures."""
        self.node.destroy_node()
        self.executor.shutdown()

    def test_red_detection(self):
        """Test that the node correctly detects red pixels."""
        # Create test image with red pixels
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [0, 0, 255]  # Red square in BGR
        
        # Convert to ROS Image message
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(test_image, 'bgr8')
        
        # Publish test message
        pub = self.node.create_publisher(Image, 'test_input', 10)
        pub.publish(img_msg)
        
        # Spin to process the message
        start_time = time.time()
        while self.node.last_processed is None and time.time() - start_time < 1.0:
            self.executor.spin_once(timeout_sec=0.1)
        
        # Check the result
        self.assertIsNotNone(self.node.last_processed, "Node did not process image in time")
        self.assertGreater(self.node.last_processed, 0, "Node did not detect red pixels")
        
    def test_image_processing_with_no_red(self):
        """Test that the node handles images with no red pixels."""
        # Create test image with no red pixels
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = [255, 0, 0]  # Blue image in BGR
        
        # Convert to ROS Image message
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(test_image, 'bgr8')
        
        # Publish test message
        pub = self.node.create_publisher(Image, 'test_input', 10)
        pub.publish(img_msg)
        
        # Spin to process the message
        start_time = time.time()
        while self.node.last_processed is None and time.time() - start_time < 1.0:
            self.executor.spin_once(timeout_sec=0.1)
        
        # Check the result
        self.assertIsNotNone(self.node.last_processed, "Node did not process image in time")
        self.assertEqual(self.node.last_processed, 0, "Node detected red pixels in blue image")
        

if __name__ == '__main__':
    import time  # Import time module for the tests
    unittest.main()
```

**Step 3: Creating Integration Tests**

```python
# integration_test.py
import unittest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool
import time

class NavigationIntegrationTest(unittest.TestCase):
    """Integration tests for navigation system components."""
    
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        """Set up integration test environment."""
        self.navigation_bridge = NavigationBridgeNode()
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.navigation_bridge)
        
        # Start executor in a separate thread
        from threading import Thread
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

    def tearDown(self):
        """Tear down integration test environment."""
        self.executor.shutdown()
        self.navigation_bridge.destroy_node()

    def test_navigation_pipeline(self):
        """Test the complete navigation pipeline: goal → plan → execute."""
        # Enable the system
        enable_success = self.navigation_bridge.call_enable_service(True)
        self.assertTrue(enable_success, "Failed to enable system")
        
        # Send a navigation goal
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1.0
        goal.pose.position.y = 1.0
        goal.pose.orientation.w = 1.0
        
        self.navigation_bridge.send_goal(goal)
        
        # Wait for navigation to complete
        success = self.navigation_bridge.wait_for_navigation(10.0)  # 10 second timeout
        self.assertTrue(success, "Navigation did not complete successfully")

class NavigationBridgeNode(Node):
    """Node to facilitate navigation integration testing."""
    
    def __init__(self):
        super().__init__('navigation_bridge_node')
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.status_sub = self.create_subscription(
            String, 'navigation_status', self.status_callback, 10
        )
        
        # Service clients
        self.enable_client = self.create_client(SetBool, 'robot/enable')
        
        # Internal state
        self.navigation_status = 'idle'
        self.navigation_start_time = None
        
    def status_callback(self, msg):
        """Handle navigation status updates."""
        self.navigation_status = msg.data
        self.get_logger().debug(f'Navigation status: {msg.data}')
    
    def call_enable_service(self, enable_flag):
        """Call the robot enable service."""
        if not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Enable service not available')
            return False
        
        req = SetBool.Request()
        req.data = enable_flag
        
        future = self.enable_client.call_async(req)
        
        # Wait for result
        start_time = time.time()
        while not future.done() and time.time() - start_time < 5.0:
            time.sleep(0.1)
        
        if future.done():
            response = future.result()
            return response.successful
        else:
            self.get_logger().error('Enable service call timed out')
            return False
    
    def send_goal(self, goal):
        """Send a navigation goal."""
        self.navigation_start_time = time.time()
        self.goal_pub.publish(goal)
        self.get_logger().info(f'Sent navigation goal to ({goal.pose.position.x}, {goal.pose.position.y})')
    
    def wait_for_navigation(self, timeout):
        """Wait for navigation to complete."""
        start_time = time.time()
        
        while (self.navigation_status != 'completed' and 
               time.time() - start_time < timeout):
            time.sleep(0.1)
        
        return self.navigation_status == 'completed'

if __name__ == '__main__':
    unittest.main()
```

**Step 4: Creating System Tests with Launch Files**

```python
# system_test_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch arguments
    run_tests_arg = DeclareLaunchArgument(
        'run_tests',
        default_value='false',
        description='Run system tests after launch'
    )
    
    # System components
    perception_node = Node(
        package='robot_perception',
        executable='perception_node',
        name='perception_node',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )
    
    planning_node = Node(
        package='robot_planning',
        executable='planning_node',
        name='planning_node',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )
    
    control_node = Node(
        package='robot_control',
        executable='control_node',
        name='control_node',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )
    
    # Test nodes
    integration_test_node = Node(
        package='robot_system_tests',
        executable='system_integration_test_node',
        name='system_integration_test_node',
        parameters=[{'use_sim_time': False}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('run_tests'))
    )
    
    # Performance monitoring node
    perf_monitor_node = Node(
        package='robot_diagnostics',
        executable='performance_monitor',
        name='performance_monitor',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )
    
    return LaunchDescription([
        run_tests_arg,
        perception_node,
        planning_node,
        control_node,
        perf_monitor_node,
        integration_test_node,
    ])
```

**Step 5: Implementing Performance and Stress Testing**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import time
from collections import deque

class PerformanceTestNode(Node):
    def __init__(self):
        super().__init__('performance_test_node')
        
        # Publishers for performance testing
        self.test_pub = self.create_publisher(Image, 'performance_test_topic', 10)
        self.results_pub = self.create_publisher(
            Image, 'performance_results', 1  # QoS = 1 for results
        )
        
        # Internal tracking
        self.bridge = CvBridge()
        self.message_count = 0
        self.message_times = deque(maxlen=1000)  # Keep last 1000 timing measurements
        self.start_time = time.time()
        
        # Performance test parameters
        self.test_duration = 30.0  # seconds
        self.message_rate = 100.0  # Hz
        self.payload_size = (640, 480, 3)  # 640x480 RGB image
        
        # Timer for sending test messages
        self.send_timer = self.create_timer(
            1.0/self.message_rate,
            self.send_test_message
        )
        
        self.get_logger().info(f'Starting performance test: {self.message_rate}Hz, {self.payload_size} images')

    def send_test_message(self):
        """Send test messages for performance evaluation."""
        if time.time() - self.start_time > self.test_duration:
            self.get_logger().info('Performance test completed')
            self.calculate_and_report_performance()
            self.send_timer.cancel()
            return
        
        # Create a test image (simulated sensor data)
        image_data = np.random.randint(0, 255, self.payload_size, dtype=np.uint8)
        img_msg = self.bridge.cv2_to_imgmsg(image_data, 'bgr8')
        img_msg.header = Header()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = f'test_frame_{self.message_count}'
        
        # Record send time
        send_time = time.time()
        self.message_times.append((self.message_count, send_time, 'sent'))
        
        self.test_pub.publish(img_msg)
        self.message_count += 1
        
        # Log progress periodically
        if self.message_count % (int(self.message_rate) * 5) == 0:
            self.get_logger().info(f'Sent {self.message_count} messages')

    def calculate_and_report_performance(self):
        """Calculate and report performance metrics."""
        total_time = time.time() - self.start_time
        expected_messages = int(self.message_rate * self.test_duration)
        
        # Calculate metrics
        delivery_rate = self.message_count / total_time
        efficiency = (self.message_count / expected_messages) * 100
        
        self.get_logger().info('Performance Test Results:')
        self.get_logger().info(f'  Duration: {total_time:.2f}s')
        self.get_logger().info(f'  Messages Sent: {self.message_count}')
        self.get_logger().info(f'  Delivery Rate: {delivery_rate:.2f} msg/s')
        self.get_logger().info(f'  Efficiency: {efficiency:.2f}%')

class StressTestNode(Node):
    def __init__(self):
        super().__init__('stress_test_node')
        
        # Create multiple publishers to stress the system
        self.publishers = []
        for i in range(10):  # 10 different topics
            pub = self.create_publisher(
                Image, 
                f'stress_test_topic_{i}', 
                10
            )
            self.publishers.append(pub)
        
        # Subscriber to monitor system health
        self.status_sub = self.create_subscription(
            Image,  # Using Image as placeholder
            'system_status',
            self.status_callback,
            1
        )
        
        # Timer for stress test
        self.stress_timer = self.create_timer(0.01, self.send_stress_messages)  # 100 Hz per publisher
        self.message_counter = 0
        self.start_time = time.time()
        
        self.get_logger().info('Starting stress test with 10 topics at 100Hz each')

    def status_callback(self, msg):
        """Monitor system status during stress test."""
        # In a real implementation, this would track system metrics
        pass

    def send_stress_messages(self):
        """Send messages on multiple topics."""
        if time.time() - self.start_time > 60.0:  # 1 minute test
            self.get_logger().info(f'Stress test completed. Sent {self.message_counter} total messages')
            self.stress_timer.cancel()
            return
        
        # Send a message on each publisher
        bridge = CvBridge()
        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_msg = bridge.cv2_to_imgmsg(image_data, 'bgr8')
        
        for i, pub in enumerate(self.publishers):
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = f'stress_frame_{self.message_counter}_{i}'
            pub.publish(img_msg)
        
        self.message_counter += len(self.publishers)

def run_performance_test():
    """Run the performance test."""
    rclpy.init()
    
    perf_node = PerformanceTestNode()
    
    try:
        rclpy.spin(perf_node)
    except KeyboardInterrupt:
        perf_node.get_logger().info('Performance test interrupted')
    finally:
        perf_node.destroy_node()
        rclpy.shutdown()

def run_stress_test():
    """Run the stress test."""
    rclpy.init()
    
    stress_node = StressTestNode()
    
    try:
        rclpy.spin(stress_node)
    except KeyboardInterrupt:
        stress_node.get_logger().info('Stress test interrupted')
    finally:
        stress_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='ROS 2 Performance and Stress Testing')
    parser.add_argument('test_type', choices=['performance', 'stress'], 
                       help='Type of test to run')
    
    args = parser.parse_args()
    
    if args.test_type == 'performance':
        run_performance_test()
    elif args.test_type == 'stress':
        run_stress_test()
```

**Step 6: Creating a Complete Testing Framework**

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
import json
import time
from datetime import datetime

class ComprehensiveTestFramework(Node):
    """A comprehensive testing framework for ROS 2 systems."""
    
    def __init__(self):
        super().__init__('comprehensive_test_framework')
        
        # Test configuration
        self.declare_parameter('test_suite', 'basic', 
                              ParameterDescriptor(description='Test suite to run'))
        self.declare_parameter('test_timeout', 60.0,
                              ParameterDescriptor(description='Test timeout in seconds'))
        self.declare_parameter('enable_performance_tests', True,
                              ParameterDescriptor(description='Whether to run performance tests'))
        
        # Publishers and subscribers
        self.test_results_pub = self.create_publisher(String, 'test_results', 10)
        self.test_commands_sub = self.create_subscription(
            String, 'test_commands', self.test_command_callback, 10
        )
        
        # Initialize test suites
        self.test_suites = {
            'basic': self.run_basic_tests,
            'integration': self.run_integration_tests,
            'performance': self.run_performance_tests,
            'regression': self.run_regression_tests
        }
        
        # Test state
        self.current_tests = []
        self.test_results = {}
        self.test_start_time = None
        
        # Parameter callback
        self.add_on_set_parameters_callback(self.param_callback)
        
        self.get_logger().info('Comprehensive Test Framework initialized')

    def param_callback(self, params):
        """Handle parameter changes."""
        for param in params:
            if param.name == 'test_suite':
                self.get_logger().info(f'Test suite changed to: {param.value}')
        
        result = SetParametersResult()
        result.successful = True
        return result

    def test_command_callback(self, msg):
        """Handle test commands."""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '')
            
            if command == 'run_test_suite':
                suite_name = command_data.get('suite', 'basic')
                self.run_test_suite(suite_name)
            elif command == 'get_results':
                self.publish_results()
            elif command == 'reset':
                self.reset_tests()
                
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON command: {msg.data}')

    def run_test_suite(self, suite_name):
        """Run a specific test suite."""
        self.get_logger().info(f'Running test suite: {suite_name}')
        
        if suite_name in self.test_suites:
            # Start test timing
            self.test_start_time = time.time()
            
            # Run the test suite
            self.test_suites[suite_name]()
            
            # Calculate duration
            duration = time.time() - self.test_start_time
            self.get_logger().info(f'Test suite {suite_name} completed in {duration:.2f}s')
            
            # Publish results
            self.publish_results()
        else:
            self.get_logger().error(f'Unknown test suite: {suite_name}')

    def run_basic_tests(self):
        """Run basic functionality tests."""
        self.get_logger().info('Running basic functionality tests...')
        
        # Test 1: Parameter accessibility
        result = self.test_parameters()
        self.test_results['parameter_access'] = result
        
        # Test 2: Publisher/subscriber connectivity
        result = self.test_communication()
        self.test_results['communication'] = result
        
        # Test 3: Service availability
        result = self.test_services()
        self.test_results['services'] = result

    def run_integration_tests(self):
        """Run integration tests."""
        self.get_logger().info('Running integration tests...')
        
        # Test multi-node coordination
        result = self.test_node_coordination()
        self.test_results['node_coordination'] = result
        
        # Test data flow
        result = self.test_data_flow()
        self.test_results['data_flow'] = result

    def run_performance_tests(self):
        """Run performance tests."""
        if self.get_parameter('enable_performance_tests').value:
            self.get_logger().info('Running performance tests...')
            
            # Test message throughput
            result = self.test_message_throughput()
            self.test_results['throughput'] = result
            
            # Test latency
            result = self.test_latency()
            self.test_results['latency'] = result
            
            # Test resource usage
            result = self.test_resource_usage()
            self.test_results['resource_usage'] = result

    def run_regression_tests(self):
        """Run regression tests."""
        self.get_logger().info('Running regression tests...')
        
        # Compare against baseline
        result = self.test_against_baseline()
        self.test_results['regression'] = result

    def test_parameters(self):
        """Test parameter accessibility."""
        try:
            # Try to get a parameter
            param_value = self.get_parameter('test_suite').value
            return {'success': True, 'value': param_value}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_communication(self):
        """Test publisher/subscriber communication."""
        try:
            # This is a simplified test - in reality, you'd implement proper pub/sub verification
            pub = self.create_publisher(String, 'test_topic', 1)
            
            msg = String()
            msg.data = f'test_message_{time.time()}'
            pub.publish(msg)
            
            return {'success': True, 'message_sent': msg.data}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_services(self):
        """Test service availability."""
        # This would check if services are available
        # For this example, we'll return a placeholder
        return {'success': True, 'checked_services': []}

    def test_node_coordination(self):
        """Test coordination between nodes."""
        # This would test that multiple nodes can coordinate properly
        return {'success': True, 'tested_coordinations': ['navigation-planning', 'perception-control']}

    def test_data_flow(self):
        """Test data flow through the system."""
        # This would test that data flows properly from source to destination
        return {'success': True, 'tested_flows': ['camera->perception->planning->control']}

    def test_message_throughput(self):
        """Test message throughput."""
        # This would test how many messages per second the system can handle
        return {'success': True, 'throughput_msgs_per_sec': 1000}

    def test_latency(self):
        """Test system latency."""
        # This would test the time from message send to receive
        return {'success': True, 'avg_latency_ms': 5.2}

    def test_resource_usage(self):
        """Test resource usage."""
        # This would test CPU, memory, and network usage
        return {'success': True, 'cpu_percent': 15.3, 'memory_mb': 250}

    def test_against_baseline(self):
        """Test against baseline performance."""
        # This would compare current performance to a known baseline
        return {'success': True, 'performance_delta_percent': 2.1}

    def publish_results(self):
        """Publish test results."""
        results_msg = String()
        results_msg.data = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': self.get_test_summary()
        })
        
        self.test_results_pub.publish(results_msg)
        self.get_logger().info('Published test results')

    def get_test_summary(self):
        """Get a summary of test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }

    def reset_tests(self):
        """Reset test results."""
        self.test_results = {}
        self.get_logger().info('Test results reset')

def main(args=None):
    rclpy.init(args=args)
    
    framework = ComprehensiveTestFramework()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(framework)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        framework.get_logger().info('Shutting down Comprehensive Test Framework')
    finally:
        framework.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing Best Practices

### Unit Testing Best Practices

1. **Isolate Components**: Test individual nodes and functions in isolation
2. **Use Mock Objects**: Mock dependencies to focus on the component under test
3. **Test Edge Cases**: Include tests for boundary conditions and error scenarios
4. **Parameterized Tests**: Use parameterized tests to check multiple input scenarios
5. **Consistent Naming**: Use descriptive names for test methods

### Integration Testing Best Practices

1. **Test Interfaces**: Focus on testing how components interact
2. **Realistic Scenarios**: Use scenarios that reflect actual usage
3. **Timing Considerations**: Account for timing dependencies between components
4. **Failure Mode Testing**: Test how the system handles component failures
5. **Incremental Integration**: Test integration incrementally rather than all at once

### System Testing Best Practices

1. **End-to-End Scenarios**: Test complete robotic behaviors from start to finish
2. **Performance Requirements**: Verify that the system meets performance requirements
3. **Safety Validation**: Validate that safety mechanisms work correctly
4. **Recovery Testing**: Test system recovery from various failure modes
5. **Load Testing**: Test system behavior under expected load conditions

### Continuous Integration

- Set up automated testing for every code commit
- Use simulation environments for CI testing
- Track performance metrics over time
- Implement test coverage requirements
- Integrate code quality checks

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                   System Integration & Testing                      │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Unit Testing  │    │ Integration     │    │  System         │ │
│  │                 │    │   Testing       │    │   Testing       │ │
│  │ • Individual    │    │ • Component     │    │ • Complete      │ │
│  │   components    │    │   interactions  │    │   behaviors     │ │
│  │ • Isolated      │    │ • Interface     │    │ • Performance   │ │
│  │   validation    │    │   validation    │    │ • Safety        │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│         │                       │                       │         │
│         ▼                       ▼                       ▼         │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Test Execution Pipeline                      ││
│  │  • Code Change    • Automated Testing    • Results Analysis   ││
│  │  • Compilation    • Simulation Tests     • Reporting          ││
│  │  • Deployment     • Hardware Tests       • Alerting           ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Testing Framework                            ││
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  ││
│  │  │ Performance │  │ Stress Test  │  │ Regression &         │  ││
│  │  │ Testing     │  │              │  │ Compliance Testing   │  ││
│  │  │ • Latency   │  │ • Load       │  │ • Safety validation  │  ││
│  │  │ • Throughput│  │ • Resource   │  │ • Requirements check │  ││
│  │  │ • Resource  │  │ • Failure    │  │ • Standards          │  ││
│  │  │   usage     │  │ • Recovery   │  │   compliance         │  ││
│  │  └─────────────┘  └──────────────┘  └──────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Design integration strategies for complex ROS 2 systems
- [ ] Implement comprehensive testing frameworks for ROS 2 nodes and systems
- [ ] Perform unit, integration, and system testing of robotic applications
- [ ] Create launch files for system-wide testing environments
- [ ] Monitor and debug integrated ROS 2 systems
- [ ] Evaluate system performance and reliability metrics