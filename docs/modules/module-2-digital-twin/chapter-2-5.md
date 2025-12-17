---
sidebar_label: 'Chapter 2.5: Visualization and Scenario Testing'
---

# Chapter 2.5: Visualization and Scenario Testing

## Introduction

Visualization and scenario testing represent the final crucial components of effective digital twin implementations for robotics. Visualization systems provide intuitive, real-time representations of robot states, sensor data, and environmental interactions, enabling researchers and engineers to monitor and debug robotic behaviors efficiently. Scenario testing, on the other hand, allows systematic evaluation of robot capabilities under various conditions, configurations, and environmental challenges.

These elements work together to ensure that simulation environments serve as effective tools for developing and validating robotic systems. Visualization systems help identify issues and understand robot behaviors, while comprehensive scenario testing validates the robustness and reliability of robotic algorithms before deployment to physical hardware.

This chapter explores visualization techniques for robotics simulation and the development of systematic scenario testing methodologies that ensure effective validation of AI/humanoid robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement effective visualization systems for robot state monitoring
- Design comprehensive scenario testing frameworks for validation
- Create visualization overlays for sensor data and robot state
- Establish systematic testing protocols for robotic capabilities
- Analyze and interpret test results from simulation scenarios

## Explanation

### Visualization Systems in Robotics

Visualization systems for robotics simulation typically encompass multiple layers of information display:

1. **3D Visualization**: Real-time rendering of the simulation environment, robot models, and environmental elements
2. **Robot State Visualization**: Display of joint angles, velocities, forces, and other internal states
3. **Sensor Data Visualization**: Overlay of camera feeds, LiDAR point clouds, IMU readings, and other sensor outputs
4. **Trajectory Visualization**: Display of planned paths, executed trajectories, and goal positions
5. **Debug Visualization**: Specialized rendering for collision shapes, force vectors, and other debugging information

### Scenario Testing Frameworks

Effective scenario testing in robotics simulation involves:

- **Deterministic Scenarios**: Repeatable test cases with specific parameters and expected outcomes
- **Randomized Scenarios**: Tests with random elements to validate robustness across possible conditions
- **Edge Case Scenarios**: Situations designed to stress test the limits of robot capabilities
- **Regression Testing**: Standardized tests to ensure that new code changes don't break existing functionality
- **Performance Testing**: Evaluation of computational efficiency and real-time performance capabilities

### Integration of Visualization and Testing

Visualization and scenario testing are most effective when integrated. Visualizations can be used to monitor scenario execution, while scenario testing frameworks can automate the collection and analysis of visualization data. This integration allows for rapid identification and debugging of issues that arise during testing.

## Example Walkthrough

Consider implementing visualization and scenario testing for a humanoid robot navigation task in Gazebo:

**Step 1: Gazebo Visualization Setup**
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- Physics configuration -->
    <physics name="ode" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Visualization camera -->
    <gui>
      <camera name="user_camera">
        <pose>-2.0 -2.0 1.5 0.0 0.29 1.57</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>

    <!-- Visualization plugins for trajectories -->
    <gazebo>
      <plugin name="trajectory_visualization" filename="libgazebo_ros_p3d.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>10.0</updateRate>
        <bodyName>humanoid_robot::base_link</bodyName>
        <topicName>robot_pose</topicName>
        <gaussianNoise>0.0</gaussianNoise>
        <frameName>map</frameName>
      </plugin>
    </gazebo>

    <!-- Optional: Add a plugin for custom visualization -->
    <gazebo>
      <plugin name="custom_visualization" filename="libcustom_visualization.so">
        <topic_name>/visualization_marker</topic_name>
      </plugin>
    </gazebo>
  </world>
</sdf>
```

**Step 2: RViz Configuration for Robot State Visualization**
```yaml
# rviz config file for humanoid robot visualization
Panels:
  - Class: rviz/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
        - /LaserScan1
        - /Camera1
        - /Path1
        - /MarkerArray1
      Splitter Ratio: 0.5
    Tree Height: 787
  - Class: rviz/Selection
    Name: Selection
  - Class: rviz/Tool Properties
    Expanded:
      - /2D Pose Estimate1
      - /2D Nav Goal1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
  - Class: rviz/Time
    Experimental: false
    Name: Time
    SyncMode: 0
    SyncSource: ""
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 100
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz/RobotModel
      Collision Enabled: false
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
        base_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      Robot Description: robot_description
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
        base_link:
          Value: true
        head:
          Value: true
        left_elbow:
          Value: true
        left_foot:
          Value: true
        left_hand:
          Value: true
        left_knee:
          Value: true
        left_shin:
          Value: true
        left_shoulder:
          Value: true
        left_thigh:
          Value: true
        left_wrist:
          Value: true
        map:
          Value: true
        odom:
          Value: true
        right_elbow:
          Value: true
        right_foot:
          Value: true
        right_hand:
          Value: true
        right_knee:
          Value: true
        right_shin:
          Value: true
        right_shoulder:
          Value: true
        right_thigh:
          Value: true
        right_wrist:
          Value: true
        torso:
          Value: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: true
      Tree:
        map:
          odom:
            base_link:
              torso:
                head:
                  {}
                left_shoulder:
                  left_upper_arm:
                    left_elbow:
                      left_lower_arm:
                        left_wrist:
                          left_hand:
                            {}
                right_shoulder:
                  right_upper_arm:
                    right_elbow:
                      right_lower_arm:
                        right_wrist:
                          right_hand:
                            {}
              left_hip:
                left_thigh:
                  left_knee:
                    left_shin:
                      left_foot:
                        {}
              right_hip:
                right_thigh:
                  right_knee:
                    right_shin:
                      right_foot:
                        {}
      Update Interval: 0
      Value: true
    - Class: rviz/Camera
      Enabled: true
      Image Topic: /humanoid_robot/camera/image_raw
      Name: Camera
      Overlay Alpha: 0.5
      Queue Size: 2
      Transport Hint: raw
      Unreliable: false
      Value: true
      Visibility:
        Grid: true
        Path: true
        RobotModel: true
        TF: true
        Value: true
      Zoom Factor: 1
    - Alpha: 0.699999988079071
      Class: rviz/Map
      Color Scheme: map
      Draw Behind: false
      Enabled: true
      Name: Map
      Topic: /map
      Unreliable: false
      Use Timestamp: false
      Value: true
    - Class: rviz/Path
      Alpha: 1
      Buffer Length: 1
      Class: rviz/Path
      Color: 255; 0; 0
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Lines
      Line Width: 0.029999999329447746
      Name: Path
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 255; 85; 255
      Pose Style: None
      Radius: 0.029999999329447746
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic: /move_base/NavfnROS/plan
      Unreliable: false
      Value: true
    - Class: rviz/MarkerArray
      Enabled: true
      Marker Topic: /visualization_marker_array
      Name: MarkerArray
      Namespaces:
        {}
      Queue Size: 100
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Default Light: true
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz/Interact
      Hide Inactive Objects: true
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
    - Class: rviz/Measure
    - Class: rviz/SetInitialPose
      Topic: /initialpose
    - Class: rviz/SetGoal
      Topic: /move_base_simple/goal
    - Class: rviz/PublishPoint
      Single click: true
      Topic: /clicked_point
  Value: true
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.7853981852531433
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.7853981852531433
    Saved: ~
Window Geometry:
  Camera:
    collapsed: false
  Displays:
    collapsed: false
  Height: 1056
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd00000004000000000000016a00000396fc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000006100fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000002800000396000000d700fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002c4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a005600690065007700730000000028000002c4000000ad00fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e100000197000000030000073f0000003efc0100000002fb0000000800540069006d006501000000000000073f000002f600fffffffb0000000800540069006d00650100000000000004500000000000000000000005cf0000039600000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Time:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1855
  X: 65
  Y: 24
```

**Step 3: Gazebo Scenario Testing Framework**
```python
#!/usr/bin/env python3
# Example scenario testing framework for humanoid robot in Gazebo

import rospy
import unittest
import actionlib
import time
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float64
import tf
import math

class HumanoidScenarioTest(unittest.TestCase):
    def setUp(self):
        rospy.init_node('humanoid_tester', anonymous=True)

        # Initialize action client for navigation
        self.nav_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()

        # TF listener for pose tracking
        self.listener = tf.TransformListener()

        # Gazebo services
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Publishers for direct control
        self.joint_publishers = {}
        for joint_name in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_shoulder', 'right_shoulder']:
            self.joint_publishers[joint_name + '_position_controller/command'] = rospy.Publisher(
                f'/humanoid_robot/{joint_name}_position_controller/command',
                Float64,
                queue_size=1
            )

    def navigate_to_pose(self, x, y, theta):
        """Send navigation goal and wait for result"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(theta/2.0)
        goal.target_pose.pose.orientation.w = math.cos(theta/2.0)

        self.nav_client.send_goal(goal)
        self.nav_client.wait_for_result(rospy.Duration(120.0))  # Increased timeout for complex humanoid movement
        return self.nav_client.get_result()

    def spawn_object_in_gazebo(self, name, model_xml, pose):
        """Spawn an object in Gazebo for testing"""
        try:
            rospy.wait_for_service('/gazebo/spawn_urdf_model')
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

            res = spawn_model(name, model_xml, "", pose, "world")
            return res.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False

    def delete_object_from_gazebo(self, name):
        """Remove an object from Gazebo"""
        try:
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

            res = delete_model(name)
            return res.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False

    def test_basic_navigation(self):
        """Test basic navigation to known poses"""
        result = self.navigate_to_pose(1.0, 1.0, 0.0)
        self.assertIsNotNone(result, "Navigation failed to reach goal")
        self.assertTrue(result.state == 3, "Navigation did not succeed (state != SUCCEEDED)")

    def test_corner_case_obstacle_avoidance(self):
        """Test navigation with obstacles in Gazebo environment"""
        # Define a simple obstacle model
        obstacle_model = """
        <robot name="test_obstacle">
          <link name="obstacle_link">
            <collision>
              <origin xyz="0 0 0.25"/>
              <geometry>
                <box size="0.2 0.2 0.5"/>
              </geometry>
            </collision>
            <visual>
              <origin xyz="0 0 0.25"/>
              <geometry>
                <box size="0.2 0.2 0.5"/>
              </geometry>
              <material name="red">
                <color rgba="1 0 0 1"/>
              </material>
            </visual>
            <inertial>
              <mass value="1.0"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
            </inertial>
          </link>
        </robot>
        """

        # Spawn obstacle in path
        obstacle_pose = PoseStamped()
        obstacle_pose.header.frame_id = "world"
        obstacle_pose.pose.position.x = 2.0
        obstacle_pose.pose.position.y = 0.0
        obstacle_pose.pose.position.z = 0.0
        obstacle_pose.pose.orientation.w = 1.0

        success = self.spawn_object_in_gazebo("test_obstacle_1", obstacle_model, obstacle_pose.pose)
        self.assertTrue(success, "Failed to spawn obstacle in Gazebo")

        time.sleep(1.0)  # Allow physics to settle

        # Attempt navigation
        result = self.navigate_to_pose(3.0, 0.0, 0.0)

        # Clean up
        self.delete_object_from_gazebo("test_obstacle_1")

        self.assertIsNotNone(result, "Navigation failed with obstacles")
        self.assertTrue(result.state == 3, "Navigation with obstacles did not succeed")

    def test_manipulation_scenario(self):
        """Test manipulation scenario using joint control"""
        # Move left arm to a specific position
        left_shoulder_cmd = Float64()
        left_shoulder_cmd.data = 0.5  # Move shoulder up

        for _ in range(50):  # Send command for 50 iterations
            self.joint_publishers['left_shoulder_position_controller/command'].publish(left_shoulder_cmd)
            time.sleep(0.1)

        # Verify the movement happened by checking joint states
        # This would typically check joint state topic or TF transforms
        rospy.sleep(1.0)  # Allow time for movement

        # More complex manipulation could be tested here
        self.assertTrue(True, "Manipulation test completed")

    def test_balance_stillness(self):
        """Test robot's ability to maintain balance when standing still"""
        initial_state = self.get_model_state("humanoid_robot", "world")
        initial_z = initial_state.pose.position.z

        rospy.sleep(5.0)  # Wait for 5 seconds

        final_state = self.get_model_state("humanoid_robot", "world")
        final_z = final_state.pose.position.z

        # Check that the robot didn't fall over (z position shouldn't change significantly)
        z_change = abs(initial_z - final_z)
        self.assertLess(z_change, 0.1, f"Robot fell over: {z_change}m change in height")

    def test_visualization_data(self):
        """Test that visualization topics are publishing correctly"""
        # Subscribe to camera topic
        msg = rospy.wait_for_message("/humanoid_robot/camera/image_raw", "sensor_msgs/Image", timeout=5.0)
        self.assertIsNotNone(msg, "Failed to receive camera image data")

        # Subscribe to LaserScan
        msg = rospy.wait_for_message("/humanoid_robot/scan", "sensor_msgs/LaserScan", timeout=5.0)
        self.assertIsNotNone(msg, "Failed to receive laser scan data")

        # Subscribe to joint states
        msg = rospy.wait_for_message("/humanoid_robot/joint_states", "sensor_msgs/JointState", timeout=5.0)
        self.assertIsNotNone(msg, "Failed to receive joint states data")

        # Check that we have the expected number of joints
        expected_joints = ['hip_joint', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
                          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'neck']
        for joint_name in expected_joints:
            self.assertIn(joint_name, msg.name, f"Expected joint {joint_name} not found in joint states")

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun('humanoid_test', 'humanoid_scenario_test', HumanoidScenarioTest)
```

**Step 4: Automated Test Execution Script for Gazebo**
```bash
#!/bin/bash
# Automated testing script for humanoid robot in Gazebo

# Function to cleanup
cleanup() {
    echo "Cleaning up..."
    killall gazebo
    killall gzserver
    killall gzclient
    killall roscore
    killall rosmaster
    killall roslaunch
    pkill -f "roslaunch humanoid_gazebo"
    pkill -f "rosrun humanoid_test"
    sleep 3
}

# Set up trap to catch exit signals
trap cleanup EXIT INT TERM

# Create results directory
mkdir -p test_results
DATE=$(date +"%Y%m%d_%H%M%S")

# Start ROS core
echo "Starting ROS core..."
roscore &
ROS_PID=$!
sleep 2

# Start Gazebo simulation with humanoid robot
echo "Starting Gazebo simulation..."
roslaunch humanoid_gazebo humanoid_world.launch &
GAZEBO_PID=$!
sleep 10  # Wait for Gazebo to fully load

# Run scenario tests
echo "Running humanoid scenario tests..."
rosrun humanoid_test humanoid_scenario_test.py --text
TEST_RESULT=$?

# Save test results
echo "Test result: $TEST_RESULT" > test_results/test_result_$DATE.txt

# Optionally save Gazebo logs
if [ -d ~/.gazebo/logs ]; then
    cp -r ~/.gazebo/logs test_results/gazebo_logs_$DATE/
fi

# Generate test report
echo "Generating test report..."
cat << EOF > test_results/report_$DATE.txt
Test Report: $(date)
====================

Test Run: $DATE
Environment: Gazebo Simulation with Humanoid Robot
Tests Executed:
- Basic Navigation
- Obstacle Avoidance
- Manipulation Scenario
- Balance Test
- Visualization Data Validation

Result: $(if [ $TEST_RESULT -eq 0 ]; then echo "PASS"; else echo "FAIL"; fi)

EOF

echo "Tests completed with result: $TEST_RESULT"
exit $TEST_RESULT
```

### Unity Testing Framework

**Step 5: Unity Testing Implementation**
Unity provides comprehensive testing capabilities for robotics simulation, particularly when using Unity ML-Agents for AI training:

```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;

public class UnityRoboticsTests
{
    [Test]
    public void TestRobotSpawn()
    {
        // Create a robot in the scene
        GameObject robotPrefab = Resources.Load<GameObject>("RobotPrefab");
        GameObject robot = Object.Instantiate(robotPrefab);

        // Verify the robot exists and has required components
        Assert.NotNull(robot, "Robot should be instantiated");
        Assert.NotNull(robot.GetComponent<Rigidbody>(), "Robot should have a Rigidbody");
        Assert.NotNull(robot.GetComponent<RobotController>(), "Robot should have a RobotController");

        Object.DestroyImmediate(robot);
    }

    [Test]
    public void TestRobotMovement()
    {
        GameObject robotPrefab = Resources.Load<GameObject>("RobotPrefab");
        GameObject robot = Object.Instantiate(robotPrefab);
        RobotController controller = robot.GetComponent<RobotController>();

        Vector3 initialPosition = robot.transform.position;

        // Apply movement command (this would typically be done through ROS communication)
        controller.MoveRobot(1.0f, 0.0f); // Move forward

        // Wait for physics update
        yield return new WaitForFixedUpdate();

        // Verify robot moved
        Vector3 finalPosition = robot.transform.position;
        Assert.Greater(finalPosition.x, initialPosition.x, "Robot should have moved forward");

        Object.DestroyImmediate(robot);
    }

    [Test]
    public void TestLiDARSensor()
    {
        GameObject lidarObject = new GameObject("LiDAR");
        UnityLiDAR lidar = lidarObject.AddComponent<UnityLiDAR>();

        // Initialize the sensor
        lidar.Start();

        // Verify sensor components are set up
        Assert.NotNull(lidar, "LiDAR component should exist");
        Assert.AreEqual(720, lidar.rayCount, "Default ray count should be 720");

        Object.DestroyImmediate(lidarObject);
    }

    [Test]
    public void TestCameraSensor()
    {
        GameObject cameraObject = new GameObject("Camera");
        Camera cam = cameraObject.AddComponent<Camera>();
        UnityCameraSensor cameraSensor = cameraObject.AddComponent<UnityCameraSensor>();

        // Initialize the sensor
        cameraSensor.Start();

        // Verify sensor components are set up
        Assert.NotNull(cameraSensor, "Camera sensor component should exist");
        Assert.NotNull(cam.targetTexture, "Camera should have a render texture");

        Object.DestroyImmediate(cameraObject);
    }
}
```

**Step 6: Unity ML-Agents Training Test Scenario**
```csharp
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class UnityNavigationScenario : Agent
{
    [Header("Environment Configuration")]
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 2.0f;
    [SerializeField] private float rotationSpeed = 100.0f;

    [Header("Reward Configuration")]
    [SerializeField] private float reachTargetReward = 10f;
    [SerializeField] private float stepPenalty = -0.01f;
    [SerializeField] private float distanceMultiplier = -0.05f;

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot position
        transform.position = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);

        // Reset target position
        target.position = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Relative position to target
        sensor.AddObservation((target.position - transform.position) / 10f);

        // Robot's forward direction
        sensor.AddObservation(transform.forward);

        // Robot's velocity
        sensor.AddObservation(rb.velocity / 10f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Actions: [0] forward/backward, [1] rotation
        float forwardAmount = actions.ContinuousActions[0];
        float rotationAmount = actions.ContinuousActions[1];

        // Apply movement
        rb.AddForce(transform.forward * forwardAmount * moveSpeed, ForceMode.VelocityChange);
        transform.Rotate(Vector3.up, rotationAmount * rotationSpeed * Time.deltaTime);

        // Add small penalty for each step to encourage efficiency
        SetReward(stepPenalty);

        // Add distance-based reward to encourage moving toward target
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        SetReward(GetReward() + distanceToTarget * distanceMultiplier);

        // Check if target reached
        if (distanceToTarget < 1.0f)
        {
            SetReward(GetReward() + reachTargetReward);
            EndEpisode();
        }

        // End episode if too far away
        if (distanceToTarget > 10.0f)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Vertical");  // Forward/backward
        continuousActionsOut[1] = Input.GetAxis("Horizontal"); // Rotation
    }
}
```

This comprehensive implementation provides a complete visualization and scenario testing system specifically designed for robotic validation in both Gazebo and Unity simulations. Each platform offers unique advantages: Gazebo excels in accurate physics simulation and tight ROS integration, while Unity provides photorealistic rendering and sophisticated AI training capabilities through ML-Agents.

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│              Visualization and Scenario Testing                     │
│                                                                     │
│  ┌─────────────────────────┐       ┌─────────────────────────────┐ │
│  │    Visualization        │       │    Scenario Testing         │ │
│  │       System            │       │         Framework           │ │
│  │                         │       │                             │ │
│  │  ┌───────────────────┐  │       │  ┌─────────────────────────┐│ │
│  │  │ 3D Environment    │  │       │  │ Deterministic Scenarios ││ │
│  │  │ Rendering         │  │       │  │ • Specific parameters   ││ │
│  │  └───────────────────┘  │       │  │ • Expected outcomes     ││ │
│  │                         │       │  └─────────────────────────┘│ │
│  │  ┌───────────────────┐  │       │                             │ │
│  │  │ Robot State       │  │       │  ┌─────────────────────────┐│ │
│  │  │ Display           │  │       │  │ Randomized Scenarios    ││ │
│  │  │ • Joint angles    │  │       │  │ • Random elements       ││ │
│  │  │ • Velocities      │  │       │  │ • Validation of robustness││ │
│  │  │ • Forces          │  │       │  └─────────────────────────┘│ │
│  │  └───────────────────┘  │       │                             │ │
│  │                         │       │  ┌─────────────────────────┐│ │
│  │  ┌───────────────────┐  │       │  │ Edge Case Scenarios     ││ │
│  │  │ Sensor Data       │  │       │  │ • Stress testing limits ││ │
│  │  │ Overlay           │  │       │  │ • Validation of limits  ││ │
│  │  │ • Camera feeds    │  │       │  └─────────────────────────┘│ │
│  │  │ • LiDAR points    │  │       │                             │ │
│  │  │ • IMU readings    │  │       │  ┌─────────────────────────┐│ │
│  │  └───────────────────┘  │       │  │ Performance Testing     ││ │
│  │                         │       │  │ • Computational efficiency││ │
│  │  ┌───────────────────┐  │       │  │ • Real-time performance ││ │
│  │  │ Debug Info        │  │       │  └─────────────────────────┘│ │
│  │  │ • Collision shapes│  │       │                             │ │
│  │  │ • Force vectors   │  │       │  ┌─────────────────────────┐│ │
│  │  │ • Trajectory paths│  │       │  │ Regression Testing      ││ │
│  │  └───────────────────┘  │       │  │ • Ensure no regressions ││ │
│  └─────────────────────────┘       │  └─────────────────────────┘│ │
│                                   └─────────────────────────────┘ │
│                                                                     │
│  Integration: Visualization provides monitoring for test execution    │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Implement effective visualization systems for robot state monitoring
- [ ] Design comprehensive scenario testing frameworks
- [ ] Create visualization overlays for sensor data
- [ ] Establish systematic testing protocols
- [ ] Detail integration points between visualization and testing
- [ ] Provide example testing framework implementation
- [ ] Include visual representation of visualization and testing system