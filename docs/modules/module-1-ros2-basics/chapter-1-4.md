---
sidebar_label: 'Chapter 1.4: Parameters and Lifecycle Management'
---

# Chapter 1.4: Parameters and Lifecycle Management

## Introduction

Parameters and lifecycle management are critical components of ROS 2 that enable dynamic configuration and organized state management of robotic systems. Parameters provide a flexible mechanism for configuring node behavior without recompilation, while the lifecycle management system offers a structured approach to managing node states from initialization to cleanup.

Unlike ROS 1's parameter server approach, ROS 2 parameters are node-local with the ability to share through parameter services and declare specific constraints at runtime. This decentralized approach enhances system reliability by eliminating the single point of failure that existed in ROS 1.

The lifecycle management system in ROS 2 introduces a state machine approach that ensures nodes transition through well-defined states (unconfigured, inactive, active, finalized), enabling proper resource management and coordinated system startup and shutdown. This is particularly important for complex robotic systems where initialization order and resource cleanup are critical.

This chapter explores the implementation and best practices for parameter management and lifecycle management in ROS 2 systems.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement parameter declaration and management in ROS 2 nodes
- Configure nodes using parameters and parameter files
- Implement lifecycle nodes with proper state transitions
- Design parameter interfaces that are robust and user-friendly
- Handle parameter changes dynamically during execution
- Coordinate lifecycle management across multiple nodes

## Explanation

### Parameters in ROS 2

ROS 2 parameters differ from ROS 1 in several key ways:

1. **Node-local**: Parameters are stored within each node rather than in a central parameter server
2. **Type-safe**: Parameters have declared types with validation
3. **Declared**: Parameters must be declared before use with optional constraints
4. **Service-based**: Parameter services allow external tools to get/set parameters

### Parameter Declaration and Usage

Parameters in ROS 2 are declared with:
- Name
- Type
- Default value
- Constraints (min/max for numeric types)
- Description

### Lifecycle Management

The lifecycle management system in ROS 2 provides a structured state machine with these states:

- **Unconfigured**: Node is created but not configured
- **Inactive**: Node is configured but not active
- **Active**: Node is running normally
- **Finalized**: Node is cleaned up and ready for destruction

And these transitions:
- create → configure
- configure → cleanup
- configure → activate
- activate → deactivate
- deactivate → cleanup
- cleanup → shutdown

### Parameter Callbacks

Parameter callbacks allow nodes to react to parameter changes at runtime, enabling dynamic reconfiguration of behavior without restarting nodes.

## Example Walkthrough

Consider implementing a perception node with configurable parameters and proper lifecycle management for a humanoid robot.

**Step 1: Implementing a Parameter-Enabled Perception Node**

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import List

class ParameterizedPerceptionNode(Node):
    def __init__(self):
        super().__init__('parameterized_perception_node')
        
        # Initialize OpenCV bridge
        self.bridge = CvBridge()
        
        # Declare parameters with descriptions and constraints
        self.declare_parameter('image_topic', '/camera/image_raw', 
                              description='Topic from which to receive image data')
        self.declare_parameter('publish_rate', 10.0, 
                              descriptor=rclpy.ParameterDescriptor(
                                  type=Parameter.Type.DOUBLE,
                                  description='Rate at which to process images (Hz)',
                                  floating_point_range=[rclpy.ParameterDescriptor.FloatingPointRange(from_value=1.0, to_value=30.0)]
                              ))
        self.declare_parameter('detection_threshold', 0.5, 
                              descriptor=rclpy.ParameterDescriptor(
                                  type=Parameter.Type.DOUBLE,
                                  description='Threshold for object detection confidence',
                                  floating_point_range=[rclpy.ParameterDescriptor.FloatingPointRange(from_value=0.0, to_value=1.0)]
                              ))
        self.declare_parameter('enable_object_detection', True,
                              description='Whether to perform object detection')
        self.declare_parameter('detection_classes', ['person', 'cup', 'bottle'],
                              description='List of object classes to detect')
        self.declare_parameter('debug_output', False,
                              description='Enable debug output topics')
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.enable_detection = self.get_parameter('enable_object_detection').value
        self.detection_classes = self.get_parameter('detection_classes').value
        self.debug_output = self.get_parameter('debug_output').value
        
        # Initialize node components
        self.setup_subscribers_and_publishers()
        
        # Set parameter callback to handle dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Create timer for processing loop
        self.timer = self.create_timer(1.0/self.publish_rate, self.process_frame)
        
        self.get_logger().info(f'Parameterized Perception Node initialized with parameters:')
        self.get_logger().info(f'  - image_topic: {self.image_topic}')
        self.get_logger().info(f'  - publish_rate: {self.publish_rate} Hz')
        self.get_logger().info(f'  - detection_threshold: {self.detection_threshold}')
        self.get_logger().info(f'  - enable_object_detection: {self.enable_detection}')
        self.get_logger().info(f'  - detection_classes: {self.detection_classes}')
        self.get_logger().info(f'  - debug_output: {self.debug_output}')
    
    def setup_subscribers_and_publishers(self):
        """Setup subscribers and publishers based on parameters."""
        # Create subscription to image topic (determined by parameter)
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # Create publisher for processed images
        self.processed_pub = self.create_publisher(
            Image,
            'processed_image',
            10
        )
        
        # Create publisher for detection results if enabled
        if self.debug_output:
            self.detection_pub = self.create_publisher(
                String,
                'detection_results',
                10
            )
        
        # Create publisher for diagnostics
        self.diag_pub = self.create_publisher(
            String,
            'diagnostics',
            10
        )
    
    def image_callback(self, msg):
        """Store the last received image."""
        self.last_image_msg = msg
    
    def process_frame(self):
        """Process the last received image frame."""
        if not hasattr(self, 'last_image_msg'):
            return  # No image received yet
        
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(self.last_image_msg, 'bgr8')
            
            # Process image based on current parameters
            processed_image = self.process_image(cv_image)
            
            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, 'bgr8')
            processed_msg.header = self.last_image_msg.header
            self.processed_pub.publish(processed_msg)
            
            # Perform detection if enabled
            if self.enable_detection:
                detections = self.detect_objects(cv_image)
                if self.debug_output and hasattr(self, 'detection_pub'):
                    detection_msg = String()
                    detection_msg.data = f'Detected: {detections}'
                    self.detection_pub.publish(detection_msg)
            
            # Publish diagnostics
            diag_msg = String()
            diag_msg.data = f'Processed frame at {self.publish_rate}Hz with threshold {self.detection_threshold}'
            self.diag_pub.publish(diag_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
    
    def process_image(self, image):
        """Apply image processing based on current parameters."""
        # Apply a simple filter based on a parameter
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def detect_objects(self, image):
        """Detect objects in image based on current parameters."""
        # Simplified detection implementation
        # In a real system, this would use a model like YOLO
        detections = []
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect colors that might represent objects in our detection classes
        # Red detection
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and confidence threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # For this example, consider it a "cup" detection
                detections.append({
                    'class': 'cup',
                    'confidence': min(1.0, area / 5000.0),  # Scale confidence based on size
                    'bbox': cv2.boundingRect(contour)
                })
        
        return [d for d in detections if d['confidence'] >= self.detection_threshold]
    
    def parameter_callback(self, params):
        """Handle parameter changes."""
        for param in params:
            if param.name == 'publish_rate' and param.type_ == Parameter.Type.DOUBLE:
                self.publish_rate = param.value
                self.timer.timer_period_ns = int(1e9 / self.publish_rate)  # Update timer period
                self.get_logger().info(f'Updated publish rate to {self.publish_rate} Hz')
                
            elif param.name == 'detection_threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.detection_threshold = param.value
                self.get_logger().info(f'Updated detection threshold to {self.detection_threshold}')
                
            elif param.name == 'enable_object_detection' and param.type_ == Parameter.Type.BOOL:
                self.enable_detection = param.value
                self.get_logger().info(f'Object detection is now {"enabled" if self.enable_detection else "disabled"}')
                
            elif param.name == 'debug_output' and param.type_ == Parameter.Type.BOOL:
                self.debug_output = param.value
                self.get_logger().info(f'Debug output is now {"enabled" if self.debug_output else "disabled"}')
        
        # Return successful result
        result = SetParametersResult()
        result.successful = True
        return result

def main(args=None):
    rclpy.init(args=args)
    node = ParameterizedPerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Parameterized Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 2: Implementing a Lifecycle Node**

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class LifecycleNavigationNode(LifecycleNode):
    def __init__(self, node_name='lifecycle_navigation_node'):
        super().__init__(node_name)
        
        # Define state tracking
        self.subscribers = {}
        self.publishers = {}
        self.timers = {}
        self.is_active = False
        
        # Declare parameters for lifecycle node
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('control_frequency', 50.0)  # Hz
        
        self.get_logger().info('Lifecycle Navigation Node created (unconfigured)')

    # Configuration phase
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Initialize resources but don't start processing."""
        self.get_logger().info('Configuring Lifecycle Navigation Node')
        
        # Create publishers and subscribers but don't start processing
        self.cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            QoSProfile(depth=10)
        )
        
        self.odom_sub = self.create_subscription(
            String,  # Using String as placeholder - would be Odometry in real system
            'odom',
            self.odom_callback,
            QoSProfile(depth=10)
        )
        
        self.get_logger().info('Lifecycle Navigation Node configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Start processing."""
        self.get_logger().info('Activating Lifecycle Navigation Node')
        
        # Activate publishers and subscribers
        self.cmd_pub.enable()
        # Note: In real implementation, we would activate the subscription as well
        
        # Create control timer
        control_freq = self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(
            1.0/control_freq,
            self.control_callback
        )
        
        self.is_active = True
        self.get_logger().info('Lifecycle Navigation Node activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop processing but keep resources."""
        self.get_logger().info('Deactivating Lifecycle Navigation Node')
        
        # Stop timers
        self.control_timer.destroy()
        
        # Stop publishing
        self.cmd_pub.disable()
        
        self.is_active = False
        self.get_logger().info('Lifecycle Navigation Node deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Clean up resources."""
        self.get_logger().info('Cleaning up Lifecycle Navigation Node')
        
        # Destroy all subscribers, publishers, and timers
        self.destroy_publisher(self.cmd_pub)
        self.destroy_subscription(self.odom_sub)
        # Note: In real implementation, we'd destroy all resources
        
        self.get_logger().info('Lifecycle Navigation Node cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final shutdown."""
        self.get_logger().info('Shutting down Lifecycle Navigation Node')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors."""
        self.get_logger().error('Lifecycle Navigation Node encountered an error')
        return TransitionCallbackReturn.SUCCESS

    # Callback methods
    def odom_callback(self, msg):
        """Handle odometry data."""
        if self.is_active:
            self.get_logger().debug(f'Received odometry: {msg.data}')

    def control_callback(self):
        """Main control loop."""
        if not self.is_active:
            return
            
        # In a real implementation, this would run navigation control
        cmd = Twist()
        cmd.linear.x = 0.1  # Move forward slowly
        cmd.angular.z = 0.0  # No rotation
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleNavigationNode()
    
    # Initial state is unconfigured
    node.trigger_configure()
    
    try:
        # Wait for user input to progress through lifecycle
        node.get_logger().info('Node configured. Call node.trigger_activate() to activate.')
        # In practice, lifecycle nodes are managed by lifecycle managers
        node.trigger_activate()
        node.get_logger().info('Node activated. Spinning...')
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Lifecycle Navigation Node')
        node.trigger_deactivate()
        node.trigger_cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 3: Implementing Parameter Files and Launch System**

```python
# navigation_params.yaml
/**:
  ros__parameters:
    max_linear_speed: 0.5
    max_angular_speed: 1.0
    control_frequency: 50.0
    safety_distance: 0.5
    goal_tolerance: 0.1

perception_node:
  ros__parameters:
    image_topic: "/camera/image_raw"
    publish_rate: 10.0
    detection_threshold: 0.6
    enable_object_detection: true
    debug_output: false
```

**Step 4: Creating a Launch File with Parameters**

```python
# parameterized_system_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('robot_system'),
            'config',
            'navigation_params.yaml'
        ]),
        description='Path to parameters file'
    )
    
    # Lifecycle navigation node
    lifecycle_nav_node = LifecycleNode(
        package='robot_navigation',
        executable='lifecycle_navigation_node',
        name='lifecycle_nav_node',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )
    
    # Parameterized perception node
    perception_node = Node(
        package='robot_perception',
        executable='parameterized_perception_node',
        name='perception_node',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )
    
    # Lifecycle manager node
    lifecycle_manager = Node(
        package='lifecycle',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        parameters=[{
            'node_names': ['lifecycle_nav_node'],
            'autostart': True
        }],
        output='screen'
    )
    
    return LaunchDescription([
        params_file_arg,
        lifecycle_nav_node,
        perception_node,
        lifecycle_manager
    ])
```

**Step 5: Implementing Parameter Validation and Best Practices**

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult
from typing import List

class RobustParameterNode(Node):
    def __init__(self):
        super().__init__('robust_parameter_node')
        
        # Define parameter constraints and validation
        self.declare_parameter('robot_name', 'default_robot',
                              description='Name of the robot instance')
        
        self.declare_parameter('control_loop_rate', 50.0,
                              descriptor=rclpy.ParameterDescriptor(
                                  type=Parameter.Type.DOUBLE,
                                  description='Rate of the control loop in Hz',
                                  floating_point_range=[
                                      rclpy.ParameterDescriptor.FloatingPointRange(
                                          from_value=1.0, to_value=500.0, 
                                          step=1.0
                                      )
                                  ]
                              ))
        
        self.declare_parameter('joint_limits', [1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
                              descriptor=rclpy.ParameterDescriptor(
                                  type=Parameter.Type.DOUBLE_ARRAY,
                                  description='Maximum joint limits in radians',
                                  value=[1.57, 1.57, 1.57, 1.57, 1.57, 1.57]
                              ))
        
        self.declare_parameter('robot_ip', '192.168.1.100',
                              descriptor=rclpy.ParameterDescriptor(
                                  type=Parameter.Type.STRING,
                                  description='IP address of the physical robot'
                              ))
        
        # Initialize components based on parameters
        self.setup_components()
        
        # Set parameter callback
        self.add_on_set_parameters_callback(self.validate_parameters)
        
        # Create diagnostic publisher
        self.diag_pub = self.create_publisher(String, 'param_diagnostics', 10)
        
        # Log initial parameter values
        self.log_current_params()
        
        # Create timer to publish diagnostics
        control_rate = self.get_parameter('control_loop_rate').value
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)
        
        self.get_logger().info('Robust Parameter Node initialized')

    def setup_components(self):
        """Initialize node components based on parameters."""
        # Get parameters
        robot_name = self.get_parameter('robot_name').value
        control_rate = self.get_parameter('control_loop_rate').value
        
        # Perform any initialization logic based on parameters
        self.get_logger().info(f'Setting up components for {robot_name} with control rate {control_rate}Hz')

    def validate_parameters(self, parameters):
        """Validate parameters before setting them."""
        result = SetParametersResult()
        result.successful = True
        
        for param in parameters:
            if param.name == 'control_loop_rate':
                if param.type_ != Parameter.Type.DOUBLE:
                    result.successful = False
                    result.reason = f'Parameter {param.name} must be a double'
                    break
                if param.value <= 0 or param.value > 500:
                    result.successful = False
                    result.reason = f'Parameter {param.name} must be between 0 and 500, got {param.value}'
                    break
            
            elif param.name == 'robot_ip':
                if param.type_ != Parameter.Type.STRING:
                    result.successful = False
                    result.reason = f'Parameter {param.name} must be a string'
                    break
                # Basic IP validation
                import ipaddress
                try:
                    ipaddress.ip_address(param.value)
                except ValueError:
                    result.successful = False
                    result.reason = f'Parameter {param.name} is not a valid IP address: {param.value}'
                    break
            
            elif param.name == 'joint_limits':
                if param.type_ != Parameter.Type.DOUBLE_ARRAY:
                    result.successful = False
                    result.reason = f'Parameter {param.name} must be a double array'
                    break
        
        if result.successful:
            # Perform side effects of parameter change
            for param in parameters:
                if param.name == 'control_loop_rate':
                    # Reconfigure timer if rate changed
                    new_rate = param.value
                    self.diag_timer.timer_period_ns = 1_000_000_000  # Update timer (1 sec for diagnostics)
                    self.get_logger().info(f'Control loop rate updated to {new_rate}Hz')
        
        return result

    def log_current_params(self):
        """Log all current parameter values."""
        params = self._parameters
        self.get_logger().info('Current parameters:')
        for name, param in params.items():
            self.get_logger().info(f'  {name}: {param.value}')

    def publish_diagnostics(self):
        """Publish diagnostic information."""
        diag_msg = String()
        robot_name = self.get_parameter('robot_name').value
        control_rate = self.get_parameter('control_loop_rate').value
        
        diag_msg.data = f'Robot: {robot_name}, Control rate: {control_rate}Hz'
        self.diag_pub.publish(diag_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobustParameterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Robust Parameter Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 6: Implementing Parameter Management Tools**

```python
# param_management_tool.py
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import ListParameters, GetParameterTypes, GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from std_msgs.msg import String
import json

class ParameterManager(Node):
    def __init__(self):
        super().__init__('parameter_manager')
        
        # Create clients for parameter services
        self.list_params_client = self.create_client(
            ListParameters, 
            '/parameterized_perception_node/list_parameters'
        )
        self.get_params_client = self.create_client(
            GetParameters,
            '/parameterized_perception_node/get_parameters'
        )
        self.set_params_client = self.create_client(
            SetParameters,
            '/parameterized_perception_node/set_parameters'
        )
        
        # Publisher for parameter updates
        self.param_update_pub = self.create_publisher(String, 'parameter_updates', 10)
        
        # Timer to periodically check parameters
        self.timer = self.create_timer(5.0, self.check_parameters)
        
        self.get_logger().info('Parameter Manager initialized')

    async def check_parameters(self):
        """Periodically check parameters of managed nodes."""
        # Check if services are available
        if not self.list_params_client.service_is_ready():
            self.get_logger().warn("Parameter services not available")
            return
        
        # List parameters
        request = ListParameters.Request()
        future = self.list_params_client.call_async(request)
        
        try:
            response = await future
            self.get_logger().info(f"Node has parameters: {response.result.names}")
            
            # Get values for some parameters
            get_request = GetParameters.Request(names=['publish_rate', 'detection_threshold'])
            get_future = self.get_params_client.call_async(get_request)
            get_response = await get_future
            
            # Process and display parameter values
            for i, value in enumerate(get_response.values):
                param_name = get_request.names[i]
                if value.type == ParameterType.PARAMETER_DOUBLE:
                    self.get_logger().info(f"{param_name}: {value.double_value}")
                elif value.type == ParameterType.PARAMETER_BOOL:
                    self.get_logger().info(f"{param_name}: {value.bool_value}")
                elif value.type == ParameterType.PARAMETER_STRING:
                    self.get_logger().info(f"{param_name}: {value.string_value}")
                    
        except Exception as e:
            self.get_logger().error(f"Error getting parameters: {e}")

    async def update_parameter(self, node_name: str, param_name: str, param_value):
        """Update a parameter on a specific node."""
        # In a real implementation, we would construct the full service name
        # This is a simplified example
        
        # Create parameter to set
        param = Parameter()
        param.name = param_name
        
        # Set parameter value based on type
        if isinstance(param_value, bool):
            param.value.bool_value = param_value
            param.value.type = ParameterType.PARAMETER_BOOL
        elif isinstance(param_value, int):
            param.value.integer_value = param_value
            param.value.type = ParameterType.PARAMETER_INTEGER
        elif isinstance(param_value, float):
            param.value.double_value = param_value
            param.value.type = ParameterType.PARAMETER_DOUBLE
        elif isinstance(param_value, str):
            param.value.string_value = param_value
            param.value.type = ParameterType.PARAMETER_STRING
        else:
            self.get_logger().error(f"Unsupported parameter type for {param_value}")
            return False
        
        # Create set request
        request = SetParameters.Request(parameters=[param])
        
        # Call the service (in real implementation, need to get correct service name)
        future = self.set_params_client.call_async(request)
        
        try:
            response = await future
            success = all(result.successful for result in response.results)
            self.get_logger().info(f"Parameter {param_name} update {'successful' if success else 'failed'}")
            
            # Publish update notification
            update_msg = String()
            update_msg.data = f"Parameter {param_name} updated to {param_value} on {node_name}"
            self.param_update_pub.publish(update_msg)
            
            return success
        except Exception as e:
            self.get_logger().error(f"Error updating parameter: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    manager = ParameterManager()
    
    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        manager.get_logger().info('Shutting down Parameter Manager')
    finally:
        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Parameters and Lifecycle Management

### Parameter Best Practices

1. **Declare Parameters Explicitly**: Always declare parameters with appropriate types and descriptions
2. **Use Parameter Descriptors**: Provide constraints and validation information
3. **Handle Parameter Changes**: Implement callbacks to respond to parameter updates
4. **Validate Input**: Validate parameters to ensure system stability
5. **Document Parameters**: Provide clear descriptions and expected ranges
6. **Group Related Parameters**: Use parameter files to manage related configurations

### Lifecycle Management Best Practices

1. **Use Lifecycle Nodes**: For complex systems requiring coordinated startup/shutdown
2. **Implement All Transitions**: Properly implement all lifecycle transition callbacks
3. **Resource Management**: Properly acquire and release resources in appropriate transitions
4. **State Awareness**: Ensure the node behaves appropriately based on its current state
5. **Error Handling**: Implement proper error handling in lifecycle callbacks
6. **Lifecycle Managers**: Use lifecycle managers for system-level coordination

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Parameters and Lifecycle                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        Parameters                               ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ ││
│  │  │   Declaration   │  │    Validation   │  │   Dynamic       │ ││
│  │  │                 │  │                 │  │   Update        │ ││
│  │  │ • Type          │  │ • Range         │  │ • Callbacks     │ ││
│  │  │ • Constraints   │  │ • Format        │  │ • Validation    │ ││
│  │  │ • Description   │  │ • Dependencies  │  │ • Notification  │ ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Lifecycle States                            ││
│  │                                                                 ││
│  │  unconfigured ──────────configure──────────▶  inactive         ││
│  │       ▲                                           │             ││
│  │       │                                           ▼             ││
│  │    shutdown ◀───────────cleanup──────────────── active          ││
│  │                           ▲                      │             ││
│  │                           │                      ▼             ││
│  │                    deactivate ◀─────────────── activate         ││
│  │                                           ▲           │         ││
│  │                                           └───────────┘         ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                  Lifecycle Transitions                          ││
│  │  • create: Node initialization                                 ││
│  │  • configure: Resource allocation, parameter setup             ││
│  │  • activate: Start processing, enable communication            ││
│  │  • deactivate: Stop processing, maintain resources             ││
│  │  • cleanup: Release resources                                  ││
│  │  • shutdown: Final cleanup                                     ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Implement parameter declaration and management in ROS 2 nodes
- [ ] Configure nodes using parameters and parameter files
- [ ] Implement lifecycle nodes with proper state transitions
- [ ] Design parameter interfaces that are robust and user-friendly
- [ ] Handle parameter changes dynamically during execution
- [ ] Coordinate lifecycle management across multiple nodes