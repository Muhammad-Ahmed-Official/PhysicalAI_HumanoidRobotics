---
sidebar_label: 'Chapter 1.3: Services and Actions'
---

# Chapter 1.3: Services and Actions

## Introduction

Services and actions in ROS 2 provide synchronous and asynchronous request-response communication patterns that complement the publish-subscribe model of topics. While topics enable decoupled data streaming between nodes, services and actions allow for direct function calls and complex task execution with feedback and status reporting.

Services implement a synchronous request-response pattern ideal for quick operations like configuration changes, simple computations, or state queries. Actions extend this concept to support long-running tasks with continuous feedback, goal preemption, and result reporting, making them suitable for navigation, manipulation, and complex robotic behaviors.

Understanding when and how to use services versus actions is critical for developing robust robotic applications that can respond appropriately to requests while maintaining system stability and user experience.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement ROS 2 services for synchronous request-response communication
- Create and use actions for long-running tasks with feedback
- Design appropriate interfaces for services and actions
- Handle action goal preemption and cancellation requests
- Evaluate performance and use case differences between topics, services, and actions
- Debug and monitor service and action communication

## Explanation

### Services in ROS 2

Services implement a synchronous communication pattern where:

1. A service client sends a request to a service server
2. The service server processes the request
3. The service server sends a response back to the client
4. The client receives the response and continues execution

This pattern is ideal for operations that:
- Complete quickly (within milliseconds to seconds)
- Have a clear input-output relationship
- Don't require continuous feedback
- Are triggered by specific events or user commands

### Actions in ROS 2

Actions extend the service concept to support long-running tasks with:

1. **Goal**: Initial request that starts the action
2. **Feedback**: Continuous updates on execution progress
3. **Result**: Final outcome after completion (or failure)
4. **Status**: Current state (active, cancelled, succeeded, aborted)

Actions are appropriate for operations that:
- Take a long time to complete (seconds to minutes)
- Require ongoing feedback to the client
- May need to be preempted or cancelled
- Benefit from status monitoring and reporting

### Key Differences

- **Services**: Synchronous, no feedback during processing, single response
- **Actions**: Asynchronous, continuous feedback, multiple status updates, cancellable

### Service vs Action Use Cases

**Service Use Cases:**
- Setting robot configuration parameters
- Querying current robot state
- Simple computation tasks
- Requesting sensor calibration
- Triggering brief behaviors

**Action Use Cases:**
- Navigation to a goal position
- Manipulation tasks (grasping, placing)
- Complex behaviors with multiple steps
- Calibration processes that take time
- Data collection over extended periods

## Example Walkthrough

Consider implementing a humanoid robot system with navigation and manipulation capabilities using both services and actions.

**Step 1: Implementing a Basic Service**

```python
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from example_interfaces.srv import SetBool
from std_msgs.msg import String
from geometry_msgs.msg import Point

class RobotManagementService(Node):
    def __init__(self):
        super().__init__('robot_management_service')
        
        # Create service to enable/disable robot
        self.enable_service = self.create_service(
            SetBool,
            'robot/enable',
            self.enable_callback
        )
        
        # Create service to query robot status
        self.status_service = self.create_service(
            SetBool,  # Using SetBool for simplicity - in practice, use custom message
            'robot/status',
            self.status_callback
        )
        
        # Create service to set robot home position
        self.home_service = self.create_service(
            SetBool,  # Using SetBool for simplicity - in practice, use custom message
            'robot/set_home',
            self.set_home_callback
        )
        
        # Robot state
        self.is_enabled = False
        self.home_position = Point(x=0.0, y=0.0, z=0.0)
        
        # Publisher for status updates
        self.status_pub = self.create_publisher(String, 'robot/status_updates', 10)
        
        self.get_logger().info('Robot Management Service initialized')

    def enable_callback(self, request, response):
        """Handle robot enable/disable requests"""
        old_state = self.is_enabled
        self.is_enabled = request.data
        
        response.success = True
        response.message = f"Robot {'enabled' if self.is_enabled else 'disabled'}"
        
        self.get_logger().info(response.message)
        
        # Publish status update
        status_msg = String()
        status_msg.data = f"Robot state changed from {'enabled' if old_state else 'disabled'} to {'enabled' if self.is_enabled else 'disabled'}"
        self.status_pub.publish(status_msg)
        
        return response

    def status_callback(self, request, response):
        """Handle robot status requests"""
        response.success = True
        response.message = f"Robot is {'enabled' if self.is_enabled else 'disabled'}. Home position: ({self.home_position.x}, {self.home_position.y}, {self.home_position.z})"
        
        self.get_logger().info(f"Status requested: {response.message}")
        return response

    def set_home_callback(self, request, response):
        """Handle set home position requests"""
        if request.data:
            # In a real system, this might get the current position from localization
            self.home_position.x = 0.0  # Placeholder
            self.home_position.y = 0.0
            self.home_position.z = 0.0
            
            response.success = True
            response.message = f"Home position set to ({self.home_position.x}, {self.home_position.y}, {self.home_position.z})"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "Failed to set home position"
            self.get_logger().warn(response.message)
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotManagementService()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Robot Management Service')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 2: Creating Custom Action Types**

First, let's define custom action messages for navigation and manipulation:

```
# NavigateToPose.action
# Define goal
geometry_msgs/PoseStamped target_pose
float32 tolerance
---
# Define result
bool success
float32 distance_traveled
string message
---
# Define feedback
float32 distance_remaining
float32 progress_percentage
geometry_msgs/Pose current_pose
```

```
# ManipulateObject.action
# Define goal
string object_id
string manipulation_type  # "grasp", "place", "move"
geometry_msgs/Pose target_pose
float32 force_limit
---
# Define result
bool success
string message
---
# Define feedback
string status
float32 progress_percentage
geometry_msgs/Pose current_end_effector_pose
```

**Step 3: Implementing an Action Server for Navigation**

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import math

# For this example, we'll use the standard FollowJointTrajectory action
# from control_msgs.action import FollowJointTrajectory
# In practice, we'd use our custom NavigateToPose action

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        
        # Create action server for navigation
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,  # Using standard action for example - would be NavigateToPose
            'navigate_to_pose',
            execute_callback=self.execute_navigate_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Publishers and subscribers
        self.pose_pub = self.create_publisher(PoseStamped, 'current_pose', 10)
        self.cmd_pub = self.create_publisher(Pose, 'cmd_pose', 10)
        
        # Robot state
        self.current_pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        )
        self.is_navigating = False
        
        self.get_logger().info('Navigation Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject a goal request."""
        self.get_logger().info('Received navigation goal request')
        
        # Accept all goals for this example
        return GoalResponse.ACCEPT
        
    def cancel_callback(self, goal_handle):
        """Accept or reject a cancellation request."""
        self.get_logger().info('Received navigation cancel request')
        return CancelResponse.ACCEPT

    def execute_navigate_callback(self, goal_handle):
        """Execute the navigation goal."""
        self.get_logger().info('Executing navigation goal')
        
        # Set navigation flag
        self.is_navigating = True
        
        # Extract target from goal
        # For this example, we'll simulate navigation to a fixed point
        target_pose = goal_handle.request.trajectory.points[0].positions if goal_handle.request.trajectory.points else None
        
        if target_pose is None:
            self.get_logger().warn('No trajectory points in goal')
            result = FollowJointTrajectory.Result()
            result.error_code = -1  # Error code for invalid goal
            goal_handle.succeed()
            self.is_navigating = False
            return result

        # Simulate navigation with feedback
        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = ['dummy']  # Placeholder
        feedback_msg.actual.positions = [0.0]
        feedback_msg.desired.positions = [1.0]
        
        # Simulate navigation progress
        for i in range(101):  # 0 to 100 percent
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = -2  # Cancelled
                self.is_navigating = False
                self.get_logger().info('Navigation goal cancelled')
                return result
            
            # Update feedback
            feedback_msg.actual.positions[0] = i / 100.0
            feedback_msg.error.positions[0] = abs(1.0 - (i / 100.0))
            
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate movement
            time.sleep(0.05)
            
            # Update current pose (simulated)
            self.current_pose.position.x = (i / 100.0) * 2.0  # Move 2 meters
            current_pose_msg = PoseStamped()
            current_pose_msg.header.stamp = self.get_clock().now().to_msg()
            current_pose_msg.header.frame_id = 'map'
            current_pose_msg.pose = self.current_pose
            self.pose_pub.publish(current_pose_msg)
        
        # Goal reached successfully
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0  # Success
        self.is_navigating = False
        
        self.get_logger().info('Navigation goal completed successfully')
        return result

def main(args=None):
    rclpy.init(args=args)
    
    # Use multithreaded executor to handle concurrent requests
    node = NavigationActionServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Navigation Action Server')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 4: Implementing an Action Client for Navigation**

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped
import time

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')
        
        # Create action client
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,  # Using standard action for example
            'navigate_to_pose'
        )
        
        # Timer to send navigation goals periodically
        self.timer = self.create_timer(5.0, self.send_goal)
        self.goal_count = 0
        
        self.get_logger().info('Navigation Action Client initialized')

    def send_goal(self):
        """Send a navigation goal to the action server."""
        self.get_logger().info(f'Sending navigation goal #{self.goal_count}')
        
        # Wait for the action server to be available
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Action server not available')
            return
        
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        
        # For this example, we'll create a simple trajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        
        trajectory = JointTrajectory()
        trajectory.joint_names = ['dummy_joint']
        
        point = JointTrajectoryPoint()
        point.positions = [self.goal_count * 0.5]  # Different target each time
        point.time_from_start.sec = 5  # 5 seconds to reach goal
        trajectory.points = [point]
        
        goal_msg.trajectory = trajectory
        
        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.goal_count += 1

    def goal_response_callback(self, future):
        """Handle the goal response from the server."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected by server')
            return

        self.get_logger().info('Goal accepted by server, waiting for result')
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.actual.positions[0]:.2f}')

    def get_result_callback(self, future):
        """Handle the action result."""
        result = future.result().result
        self.get_logger().info(f'Received result: {result.error_code}')
        
        # Check if navigation was successful
        if result.error_code == 0:
            self.get_logger().info('Navigation completed successfully')
        else:
            self.get_logger().info(f'Navigation failed with error code: {result.error_code}')

def main(args=None):
    rclpy.init(args=args)
    client = NavigationActionClient()
    
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Shutting down Navigation Action Client')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 5: Implementing a Manipulation Action Server**

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String
from rclpy.callback_groups import ReentrantCallbackGroup
import time

class ManipulationActionServer(Node):
    def __init__(self):
        super().__init__('manipulation_action_server')
        
        # Create action server for manipulation
        # Using standard action type for this example - would use custom ManipulateObject
        from control_msgs.action import FollowJointTrajectory
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,  # Would be custom ManipulateObject action
            'manipulate_object',
            execute_callback=self.execute_manipulation_callback,
            goal_callback=self.manipulation_goal_callback,
            cancel_callback=self.manipulation_cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'manipulation_status', 10)
        
        # Robot state
        self.is_manipulating = False
        self.held_object = None
        
        self.get_logger().info('Manipulation Action Server initialized')

    def manipulation_goal_callback(self, goal_request):
        """Accept or reject a manipulation goal request."""
        self.get_logger().info('Received manipulation goal request')
        
        # Accept goals if not currently manipulating
        if not self.is_manipulating:
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT
        
    def manipulation_cancel_callback(self, goal_handle):
        """Accept or reject a manipulation cancellation request."""
        self.get_logger().info('Received manipulation cancel request')
        return CancelResponse.ACCEPT

    def execute_manipulation_callback(self, goal_handle):
        """Execute the manipulation goal."""
        self.get_logger().info('Executing manipulation goal')
        
        # Set manipulation flag
        self.is_manipulating = True
        
        # In a real implementation, this would:
        # 1. Plan the manipulation sequence (approach, grasp, lift, move, place)
        # 2. Execute each step with feedback
        # 3. Handle errors and recovery
        
        # For simulation, we'll implement a sequence of manipulation steps
        manipulation_steps = [
            "Approaching object",
            "Grasping object", 
            "Lifting object",
            "Moving object",
            "Placing object"
        ]
        
        # Publish initial status
        status_msg = String()
        status_msg.data = "Starting manipulation sequence"
        self.status_pub.publish(status_msg)
        
        # Simulate manipulation with feedback
        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = ['manipulator_joint_1', 'manipulator_joint_2']
        feedback_msg.actual.positions = [0.0, 0.0]
        feedback_msg.desired.positions = [0.0, 0.0]
        
        for i, step in enumerate(manipulation_steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = -2  # Cancelled
                self.is_manipulating = False
                self.get_logger().info('Manipulation goal cancelled')
                return result
            
            # Update feedback
            feedback_msg.actual.positions[0] = (i + 1) / len(manipulation_steps)
            feedback_msg.actual.positions[1] = (i + 1) / len(manipulation_steps) * 0.5
            feedback_msg.error.positions[0] = 0.1  # Placeholder error
            feedback_msg.error.positions[1] = 0.05
            
            goal_handle.publish_feedback(feedback_msg)
            
            # Publish status
            status_msg.data = f"Step {i+1}/{len(manipulation_steps)}: {step}"
            self.status_pub.publish(status_msg)
            
            self.get_logger().info(f'Executing: {step}')
            
            # Simulate time for each step
            time.sleep(1.0)
        
        # Finalize manipulation
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0  # Success
        self.is_manipulating = False
        
        # Update status
        status_msg.data = "Manipulation completed successfully"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info('Manipulation goal completed successfully')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationActionServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Manipulation Action Server')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Step 6: Complete System Integration with Services and Actions**

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from example_interfaces.srv import SetBool
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import String

class IntegratedRobotSystem(Node):
    def __init__(self):
        super().__init__('integrated_robot_system')
        
        # Create callback group for handling concurrent callbacks
        callback_group = ReentrantCallbackGroup()
        
        # Services
        self.enable_service = self.create_service(
            SetBool,
            'robot/enable',
            self.enable_callback,
            callback_group=callback_group
        )
        
        # Action servers
        self.nav_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'navigate_to_pose',
            execute_callback=self.execute_navigation_callback,
            goal_callback=self.nav_goal_callback,
            cancel_callback=self.nav_cancel_callback,
            callback_group=callback_group
        )
        
        self.manip_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'manipulate_object',
            execute_callback=self.execute_manipulation_callback,
            goal_callback=self.manip_goal_callback,
            cancel_callback=self.manip_cancel_callback,
            callback_group=callback_group
        )
        
        # Action clients
        self.nav_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            'navigate_to_pose',
            callback_group=callback_group
        )
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        
        # System state
        self.is_enabled = False
        self.current_task = None
        
        self.get_logger().info('Integrated Robot System initialized')

    def enable_callback(self, request, response):
        """Handle system enable/disable requests."""
        old_state = self.is_enabled
        self.is_enabled = request.data
        
        response.success = True
        response.message = f"System {'enabled' if self.is_enabled else 'disabled'}"
        
        # Publish status update
        status_msg = String()
        status_msg.data = f"System state changed from {'enabled' if old_state else 'disabled'} to {'enabled' if self.is_enabled else 'disabled'}"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(response.message)
        return response

    # Navigation action methods
    def nav_goal_callback(self, goal_request):
        if not self.is_enabled:
            self.get_logger().warn('Navigation request rejected - system disabled')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def nav_cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_navigation_callback(self, goal_handle):
        self.current_task = 'navigation'
        self.get_logger().info('Executing navigation task')
        
        # Simulate navigation
        from time import sleep
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = -2  # Cancelled
                self.current_task = None
                return result
            
            # Publish feedback
            feedback_msg = FollowJointTrajectory.Feedback()
            feedback_msg.joint_names = ['dummy']
            feedback_msg.actual.positions = [i / 10.0]
            goal_handle.publish_feedback(feedback_msg)
            
            sleep(0.1)
        
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0  # Success
        self.current_task = None
        return result

    # Manipulation action methods
    def manip_goal_callback(self, goal_request):
        if not self.is_enabled:
            self.get_logger().warn('Manipulation request rejected - system disabled')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def manip_cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_manipulation_callback(self, goal_handle):
        self.current_task = 'manipulation'
        self.get_logger().info('Executing manipulation task')
        
        # Simulate manipulation
        from time import sleep
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = -2  # Cancelled
                self.current_task = None
                return result
            
            # Publish feedback
            feedback_msg = FollowJointTrajectory.Feedback()
            feedback_msg.joint_names = ['manipulator_joint']
            feedback_msg.actual.positions = [i / 10.0 * 1.5]  # Simulate joint movement
            goal_handle.publish_feedback(feedback_msg)
            
            sleep(0.1)
        
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = 0  # Success
        self.current_task = None
        return result

def main(args=None):
    rclpy.init(args=args)
    
    # Use multithreaded executor to handle all callbacks
    node = IntegratedRobotSystem()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Integrated Robot System')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Services and Actions

### Service Design Guidelines

1. **Quick Responses**: Services should complete quickly (under 1 second). For longer operations, use actions.
2. **Simple Interface**: Keep request/response messages simple and focused.
3. **Error Handling**: Always return meaningful error messages in the response.
4. **State Management**: Services should not store state between calls unless designed to do so.

### Action Design Guidelines

1. **Long-Running Tasks**: Use actions for operations taking more than a second or requiring feedback.
2. **Feedback Frequency**: Don't send feedback too frequently; balance between responsiveness and network load.
3. **Goal Preemption**: Design actions to handle goal cancellation requests gracefully.
4. **State Reporting**: Provide clear status updates on the task progress.

### When to Use Each Pattern

- **Topic**: Continuous data streaming, sensor data, state publishing
- **Service**: Synchronous requests for quick operations (configuration, queries)
- **Action**: Long-running tasks with progress feedback (navigation, manipulation)

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Services vs Actions                               │
│                                                                     │
│  ┌─────────────────┐    Service Request-Response    ┌─────────────┐│
│  │    Client       │───────────────────────────────▶│    Server   ││
│  │                 │                                 │             ││
│  │ • Send request  │◀──────── Response ─────────────│ • Process   ││
│  │ • Wait for      │                                 │   request   ││
│  │   response      │                                 │ • Return    ││
│  │ • Continue      │                                 │   response  ││
│  └─────────────────┘                                 └─────────────┘│
│                                                                     │
│  ┌─────────────────┐    Action Goal-Feedback-Result    ┌─────────┐ │
│  │    Client       │──────── Goal ────────────────────▶│  Server │ │
│  │                 │                                   │         │ │
│  │ • Send goal     │                                   │ • Execute│ │
│  │ • Listen for    │◀────── Feedback ──────────────────│   goal   │ │
│  │   feedback      │                                   │ • Update │ │
│  │ • Get result    │◀────── Result ────────────────────│   state  │ │
│  │ • Handle        │                                   │ • Handle│ │
│  │   cancellation  │                                   │   cancel │ │
│  └─────────────────┘                                   └─────────┘ │
│                                                                     │
│  Service Characteristics:     Action Characteristics:              │
│  • Synchronous               • Asynchronous                       │
│  • Single request-response   • Goal, feedback, result             │
│  • Quick operations          • Long-running operations            │
│  • No progress updates       • Continuous progress updates        │
│  • Not cancellable           • Cancellable                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Implement ROS 2 services for synchronous request-response communication
- [ ] Create and use actions for long-running tasks with feedback
- [ ] Design appropriate interfaces for services and actions
- [ ] Handle action goal preemption and cancellation requests
- [ ] Evaluate performance and use case differences between topics, services, and actions
- [ ] Debug and monitor service and action communication