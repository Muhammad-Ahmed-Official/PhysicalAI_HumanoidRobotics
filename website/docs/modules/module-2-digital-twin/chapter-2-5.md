---
id: chapter-2-5
title: "Chapter 2.5: Visualization and Scenario Testing"
description: "Implementing visualization tools and conducting scenario tests in robotics simulation"
tags: [visualization, testing, scenarios, simulation]
---

# Chapter 2.5: Visualization and Scenario Testing

## Introduction

Visualization and scenario testing are critical components of the simulation-first approach to humanoid robotics development. This chapter explores techniques for effectively visualizing robot behavior and conducting comprehensive scenario tests in simulation environments to validate robot capabilities before physical deployment.

## Learning Outcomes

- Students will understand the importance of visualization in robotics simulation
- Learners will be able to implement effective visualization tools for robot behavior
- Readers will be familiar with scenario testing methodologies
- Students will be able to design and execute comprehensive test scenarios

## Core Concepts

Effective visualization and scenario testing in robotics simulation encompass several key areas:

1. **Visual Debugging**: Providing visual feedback for robot states, paths, and interactions
2. **Sensor Visualization**: Displaying sensor data in intuitive formats
3. **Scenario Design**: Creating diverse test scenarios that cover operational requirements
4. **Performance Metrics**: Defining and measuring metrics for robot performance
5. **Data Recording**: Capturing simulation data for analysis and debugging

Comprehensive scenario testing should include both nominal operation conditions and edge cases to ensure robust robot behavior.

## Simulation Walkthrough

Implementing visualization and scenario testing in both Gazebo and Unity:

<Tabs>
  <TabItem value="gazebo" label="Gazebo Visualization/Testing">
    ```xml
    <!-- Example of visualization elements in a Gazebo world -->
    <sdf version="1.6">
      <world name="test_scenarios">
        <!-- Visual markers and displays -->
        <model name="path_marker">
          <pose>0 0 0.05 0 0 0</pose>
          <link name="link">
            <visual name="visual">
              <geometry>
                <box>
                  <size>0.1 0.1 0.01</size>
                </box>
              </geometry>
              <material>
                <script>
                  <name>Gazebo/Red</name>
                </script>
              </material>
            </visual>
          </link>
        </model>
        
        <!-- Scenario-specific lighting -->
        <light name="test_area_light" type="spot">
          <pose>0 -5 5 0 0.5 0</pose>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
          <attenuation>
            <range>10</range>
            <constant>0.5</constant>
            <linear>0.1</linear>
            <quadratic>0.01</quadratic>
          </attenuation>
          <direction>-0.1 0.8 -0.5</direction>
          <spot>
            <inner_angle>0.5</inner_angle>
            <outer_angle>1.0</outer_angle>
            <falloff>1.0</falloff>
          </spot>
        </light>
      </world>
    </sdf>
    ```
    
    ```python
    # Python script for scenario testing with visualization
    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    import matplotlib.pyplot as plt
    import numpy as np
    
    class ScenarioTester:
        def __init__(self):
            # Initialize ROS node
            rospy.init_node('scenario_tester')
            
            # Connect to joint trajectory controller
            self.controller_client = actionlib.SimpleActionClient(
                '/humanoid/joint_trajectory_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction
            )
            self.controller_client.wait_for_server()
            
            # Initialize visualization
            self.fig, self.ax = plt.subplots()
            self.trajectory_x = []
            self.trajectory_y = []
            
        def run_navigation_scenario(self):
            # Define a simple navigation scenario
            waypoints = [
                (0.0, 0.0, 0.0),    # Start position
                (1.0, 0.0, 0.0),    # Move forward 1m
                (1.0, 1.0, 1.57),   # Turn and move right
                (0.0, 1.0, 1.57),   # Move back
                (0.0, 0.0, 0.0),    # Return to start
            ]
            
            for i, (x, y, theta) in enumerate(waypoints):
                rospy.loginfo(f"Executing waypoint {i+1}: ({x}, {y}, {theta})")
                
                # Send trajectory goal
                goal = self.create_trajectory_goal(x, y, theta)
                self.controller_client.send_goal(goal)
                
                # Record position for visualization
                self.trajectory_x.append(x)
                self.trajectory_y.append(y)
                
                # Wait for execution
                self.controller_client.wait_for_result(rospy.Duration(10.0))
                
            # Visualize the trajectory
            self.visualize_trajectory()
        
        def create_trajectory_goal(self, x, y, theta):
            # Create a simple trajectory goal
            goal = FollowJointTrajectoryGoal()
            traj = JointTrajectory()
            traj.joint_names = ["joint1", "joint2", "joint3"]  # Example joint names
            
            # Add trajectory points
            point = JointTrajectoryPoint()
            point.positions = [x, y, theta]
            point.time_from_start = rospy.Duration(2.0)
            traj.points.append(point)
            
            goal.trajectory = traj
            return goal
            
        def visualize_trajectory(self):
            # Plot the robot's trajectory
            self.ax.plot(self.trajectory_x, self.trajectory_y, 'b-', linewidth=2, label='Robot Path')
            self.ax.scatter(self.trajectory_x[0], self.trajectory_y[0], color='green', s=100, label='Start', zorder=5)
            self.ax.scatter(self.trajectory_x[-1], self.trajectory_y[-1], color='red', s=100, label='End', zorder=5)
            
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_title('Robot Trajectory Visualization')
            self.ax.grid(True)
            self.ax.legend()
            
            plt.show()
    
    if __name__ == '__main__':
        tester = ScenarioTester()
        tester.run_navigation_scenario()
    ```
  </TabItem>
  <TabItem value="unity" label="Unity Visualization/Testing">
    ```
    // Unity Scenario Testing and Visualization
    
    1. Visualization Components:
       - Trajectory Line Renderer: Shows robot path
       - Gizmos: Real-time debugging of transforms
       - Custom Render Textures: For sensor visualization
       - UI Elements: For displaying metrics and status
    
    2. Scenario Testing Framework:
       public class ScenarioTestManager : MonoBehaviour
       {
           public RobotController robot;
           public List<TestScenario> scenarios;
           public float testTimeout = 60f;  // seconds
           
           void Start()
           {
               StartCoroutine(RunAllScenarios());
           }
           
           IEnumerator RunAllScenarios()
           {
               foreach (TestScenario scenario in scenarios)
               {
                   Debug.Log($"Starting scenario: {scenario.name}");
                   
                   // Setup scenario
                   scenario.Setup();
                   
                   // Run scenario with timeout
                   float startTime = Time.time;
                   while (!scenario.IsCompleted() && 
                          Time.time - startTime < testTimeout)
                   {
                       yield return null;  // Wait for next frame
                   }
                   
                   // Evaluate results
                   bool success = scenario.Evaluate();
                   Debug.Log($"Scenario {scenario.name}: {(success ? "PASS" : "FAIL")}");
                   
                   // Cleanup
                   scenario.Cleanup();
               }
           }
       }
    
    3. Example Test Scenario:
       public class NavigationScenario : TestScenario
       {
           public Transform targetLocation;
           public float tolerance = 0.1f;
           
           public override void Setup()
           {
               // Position robot at start location
               robot.transform.position = startPosition.position;
           }
           
           public override bool IsCompleted()
           {
               // Check if robot reached target
               return Vector3.Distance(robot.transform.position, 
                                      targetLocation.position) < tolerance;
           }
           
           public override bool Evaluate()
           {
               // Return true if completed within time and constraints
               return IsCompleted() && completionTime < maxAllowedTime;
           }
       }
    ```
  </TabItem>
</Tabs>

## Visual Explanation

```
[Visualization and Scenario Testing Diagram]

Simulation Environment
┌─────────────────────────────────────────────────────────┐
│                    3D Visualization                     │
│  ┌─────────────┐      ←─ Trajectory Display            │
│  │ Humanoid    │                                        │
│  │ Robot       │ ←─ Real-time Position & Status         │
│  │ Model       │                                        │
│  └─────────────┘                                        │
│         │                                               │
│         │ Actual Position                               │
│    ┌────▼────┐                                         │
│    │ Path    │  ←─ Robot Path Visualization            │
│    │ Marker  │                                         │
│    └─────────┘                                         │
│         │                                               │
│         ▼                                               │
│  Scenario Test Framework                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Test: Navigate to Target                        │   │
│  │ ┌─────────────────┐  →  [ ] Start at position   │   │
│  │ │    Robot        │     [ ] Move to target      │   │
│  │ │                 │     [ ] Avoid obstacles     │   │
│  │ │     Target      │     [ ] Reach within time   │   │
│  │ └─────────────────┘                             │   │
│  │ Status: [ RUNNING ]  Success: [   % ]           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Performance Metrics                            │   │
│  │ - Time to complete: 15.3s                     │   │
│  │ - Energy consumption: 4.2 J                   │   │
│  │ - Path efficiency: 87%                        │   │
│  │ - Collision avoidance: PASS                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

Effective visualization and scenario testing provide insights
into robot behavior and performance in various situations.
```

## Checklist

- [x] Understand the importance of visualization in simulation
- [x] Know how to implement scenario testing frameworks
- [x] Can visualize robot trajectories and states
- [ ] Designed comprehensive test scenarios
- [ ] Implemented performance metrics tracking
- [ ] Self-assessment: How would you design a scenario to test your robot's robustness to unexpected obstacles?