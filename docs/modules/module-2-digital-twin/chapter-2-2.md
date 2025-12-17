---
sidebar_label: 'Chapter 2.2: 3D Simulation Environment Setup'
---

# Chapter 2.2: 3D Simulation Environment Setup

## Introduction

Setting up a 3D simulation environment is a crucial step in developing and testing AI-controlled robots. These environments serve as digital laboratories where researchers can experiment with robot behaviors, sensor configurations, and AI algorithms without the constraints and risks of the physical world. A well-designed simulation environment accurately represents the physical properties, visual characteristics, and interaction dynamics that will be encountered by the real robot.

Modern simulation environments like Gazebo and Unity offer sophisticated tools for creating realistic worlds with detailed physics, lighting, and sensory conditions. These platforms support a variety of robot models and can simulate complex scenarios ranging from simple indoor navigation to challenging outdoor terrains.

This chapter explores the process of configuring 3D simulation environments, including selecting appropriate platforms, defining world parameters, and ensuring fidelity between simulated and real environments.

## Learning Objectives

By the end of this chapter, you will be able to:

- Evaluate different 3D simulation platforms for robotics applications
- Configure essential parameters for a 3D simulation environment
- Set up realistic physics and rendering properties
- Import and customize 3D models for robot and environment components
- Validate simulation-environment correspondence to real-world conditions

## Explanation

### Popular Simulation Platforms

Three-dimensional simulation environments for robotics typically fall into two categories:

1. **Robotics-Specific Simulators**: Gazebo, Webots, MuJoCo, and PyBullet are tailored specifically for robotics applications. They provide robust physics engines, sensor simulation, and tight integration with ROS/ROS2.

2. **Game Engine-Based Simulators**: Unity ML-Agents and Unreal Engine (with CARLA or AirSim) leverage gaming engines to create visually realistic environments with high-quality graphics and complex scenarios.

### Configuration Elements

A complete 3D simulation environment setup involves several key configuration elements:

- **World Definition**: Physical dimensions, terrain properties, obstacles, and lighting conditions
- **Physics Engine**: Gravity, friction coefficients, collision detection parameters, and real-time simulation settings
- **Sensor Models**: Cameras, LiDAR, IMU, GPS, and other sensors with realistic noise and error profiles
- **Robot Models**: URDF/SDF descriptions for kinematics, dynamics, and visual appearance

### Fidelity Considerations

Simulation fidelity refers to how accurately the virtual environment replicates real-world conditions. High-fidelity simulations are computationally expensive but provide more reliable results when transferring behaviors to real hardware. Low-fidelity simulations run faster but may not adequately reflect real-world challenges.

The key to successful simulation environments lies in balancing computational efficiency with fidelity appropriate for the intended use case.

## Example Walkthrough

Consider setting up a 3D simulation environment for a humanoid robot designed for indoor navigation and object manipulation. This can be done using either Gazebo or Unity, each with their own advantages.

### Gazebo Simulation Setup

**Step 1: Platform Setup**
- Install Gazebo Classic (version 11) with ROS 2 Humble Hawksbill
- Verify installation with `gazebo --version`
- Set up the Gazebo model path environment variable

**Step 2: World Creation**
Create a new world file (`.world`) in the `~/.gazebo/worlds` directory or package-specific location:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Lighting setup -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
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
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Simple room with walls -->
    <model name="room_walls">
      <static>true</static>
      <link name="floor">
        <collision><geometry><box><size>10 10 0.1</size></box></geometry></collision>
        <visual><geometry><box><size>10 10 0.1</size></box></geometry></visual>
      </link>
      <link name="wall_1">
        <pose>0 -5 2.5 0 0 0</pose>
        <collision><geometry><box><size>10 0.1 5</size></box></geometry></collision>
        <visual><geometry><box><size>10 0.1 5</size></box></geometry></visual>
      </link>
      <link name="wall_2">
        <pose>0 5 2.5 0 0 0</pose>
        <collision><geometry><box><size>10 0.1 5</size></box></geometry>
        <visual><geometry><box><size>10 0.1 5</size></box></geometry></visual>
      </link>
      <link name="wall_3">
        <pose>-5 0 2.5 0 0 0</pose>
        <collision><geometry><box><size>0.1 10 5</size></box></geometry>
        <visual><geometry><box><size>0.1 10 5</size></box></geometry></visual>
      </link>
      <link name="wall_4">
        <pose>5 0 2.5 0 0 0</pose>
        <collision><geometry><box><size>0.1 10 5</size></box></geometry>
        <visual><geometry><box><size>0.1 10 5</size></box></geometry></visual>
      </link>
    </model>

    <!-- Sample obstacle -->
    <model name="obstacle_box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision><geometry><box><size>0.5 0.5 1</size></box></geometry></collision>
        <visual><geometry><box><size>0.5 0.5 1</size></box></geometry></visual>
      </link>
    </model>

    <!-- Optional: Include a sample model from Gazebo's model database -->
    <include>
      <uri>model://cylinder</uri>
      <pose>0 3 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

**Step 3: Robot Model Integration**
- Create a URDF file for your humanoid robot
- Include Gazebo-specific tags to define sensors, plugins, and materials
- Launch the world with your robot using a ROS launch file

Example launch file:
```xml
<launch>
  <!-- Load the robot description parameter -->
  <param name="robot_description" textfile="$(find your_robot_description)/urdf/your_robot.urdf" />

  <!-- Spawn the robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-param robot_description -urdf -model your_robot -x 0 -y 0 -z 1" />

  <!-- Start Gazebo with the world file -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find your_robot_gazebo)/worlds/humanoid_lab.world" />
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>
</launch>
```

**Step 4: Gazebo-Specific Sensor Configuration**
Mount sensors with Gazebo plugins:
```xml
<!-- Camera sensor example -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera_sensor">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>your_robot/camera</cameraName>
      <imageTopicName>image_raw</imageTopicName>
      <cameraInfoTopicName>camera_info</cameraInfoTopicName>
      <frameName>camera_link</frameName>
    </plugin>
  </sensor>
</gazebo>

<!-- LiDAR sensor example -->
<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topicName>scan</topicName>
      <frameName>lidar_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### Unity Simulation Setup

**Step 1: Unity Project Setup**
- Install Unity 2021.3 LTS or later
- Import Unity Robotics Hub and ML-Agents packages
- Set up ROS# for ROS/ROS2 communication

**Step 2: Environment Creation**
Create a 3D environment in Unity with appropriate physics settings:

```csharp
using UnityEngine;

public class UnityRoboticsEnvironment : MonoBehaviour
{
    [Header("Environment Configuration")]
    [SerializeField] private float gravity = -9.81f;
    [SerializeField] private PhysicMaterial defaultMaterial;
    [SerializeField] private GameObject[] robotPrefabs;

    [Header("Simulation Settings")]
    [SerializeField] private float physicsUpdateRate = 50f; // Hz

    void Start()
    {
        SetupEnvironment();
    }

    private void SetupEnvironment()
    {
        // Configure physics settings
        Physics.gravity = new Vector3(0, gravity, 0);
        Time.fixedDeltaTime = 1f / physicsUpdateRate;

        // Set default physics material if provided
        if (defaultMaterial != null)
        {
            PhysicMaterial[] materials = FindObjectsOfType<PhysicMaterial>();
            foreach (PhysicMaterial material in materials)
            {
                if (material.name == "Default Physics Material")
                {
                    // Update default material settings
                    material.staticFriction = defaultMaterial.staticFriction;
                    material.dynamicFriction = defaultMaterial.dynamicFriction;
                    material.bounciness = defaultMaterial.bounciness;
                }
            }
        }
    }

    public void SpawnRobot(int robotTypeIndex)
    {
        if (robotTypeIndex < 0 || robotTypeIndex >= robotPrefabs.Length)
        {
            Debug.LogError($"Invalid robot type index: {robotTypeIndex}");
            return;
        }

        GameObject robot = Instantiate(robotPrefabs[robotTypeIndex]);
        robot.transform.position = Vector3.zero + Vector3.up;
    }
}
```

**Step 3: Robot Controller Implementation**
Create a robot controller to handle movement and sensor data:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class RobotController : MonoBehaviour
{
    [SerializeField] private float speed = 1.0f;
    [SerializeField] private float rotationSpeed = 1.0f;

    private ROSConnection ros;
    private string robotTopic = "robot/cmd_vel";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(robotTopic);
    }

    void Update()
    {
        // Process input
        float moveVertical = Input.GetAxis("Vertical");
        float moveHorizontal = Input.GetAxis("Horizontal");

        // Create twist message
        TwistMsg cmd = new TwistMsg
        {
            linear = new Vector3Msg { x = moveVertical * speed, y = 0, z = 0 },
            angular = new Vector3Msg { x = 0, y = 0, z = moveHorizontal * rotationSpeed }
        };

        // Publish to ROS
        ros.Publish(robotTopic, cmd);
    }
}
```

**Step 4: Sensor Simulation**
Implement various sensors like camera and LiDAR:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

[RequireComponent(typeof(Camera))]
public class UnityCameraSensor : MonoBehaviour
{
    [SerializeField] private string cameraTopic = "robot/camera/image_raw";
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 480;

    private Camera cam;
    private RenderTexture renderTexture;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        cam = GetComponent<Camera>();
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = renderTexture;

        ros.RegisterPublisher<ImageMsg>(cameraTopic);
    }

    void Update()
    {
        // Capture image from camera
        RenderTexture.active = renderTexture;
        Texture2D texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS Image message
        ImageMsg imageMsg = new ImageMsg
        {
            header = new std_msgs.HeaderMsg { frame_id = "camera_frame" },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3),
            data = texture2D.GetRawTextureData<byte>()
        };

        // Publish image
        ros.Publish(cameraTopic, imageMsg);

        // Clean up
        Destroy(texture2D);
    }
}
```

**Step 5: Environment Validation**
- Build and run the Unity simulation
- Use ROS tools to verify that sensor data and control messages are being exchanged correctly
- Test various scenarios to validate the simulation behavior

Both Gazebo and Unity offer powerful simulation environments with different strengths - Gazebo excels in robotics-specific physics simulation and ROS integration, while Unity provides photorealistic rendering and complex AI training scenarios through ML-Agents.

## Visual Representation

```
┌─────────────────────────────────────────────────────────────┐
│                    3D Simulation Environment                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                                                      │   │
│  │              Virtual Laboratory                      │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │         Humanoid Robot                      │    │   │
│  │  │   o    Joint Structure                      │    │   │
│  │  │  /|\\   Motor Actuators                     │    │   │
│  │  │  / \\   Sensors (Cam, LiDAR, IMU)           │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │      ▲                                             │   │
│  │      │                                             │   │
│  │  ┌───┼─────────────────────────────────────────┐   │   │
│  │  │   │    Indoor Environment                   │   │   │
│  │  │   │                                         │   │   │
│  │  │   │  ┌──────┐       ▲                       │   │   │
│  │  │   │  │Table │    Obstacles                  │   │   │
│  │  │   │  └──────┘                               │   │   │
│  │  │   │      ▲                                  │   │   │
│  │  │   │  ┌───┴──┐        ┌───────────────────┐  │   │   │
│  │  │   │  │Chair │        │  Door Opening     │  │   │   │
│  │  │   │  └──────┘        │                   │  │   │   │
│  │  │   │                   └───────────────────┘  │   │   │
│  │  └────────────────────────────────────────────────┘   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Physics Engine: Accurate force simulation                  │
│  Rendering: Realistic lighting and textures                 │
│  Sensors: Realistic noise and limitations                   │
└─────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Evaluate different 3D simulation platforms and their features
- [ ] Configure essential parameters for 3D simulation environment
- [ ] Set up realistic physics and rendering properties
- [ ] Detail the process for importing 3D models
- [ ] Outline fidelity considerations for real-world correspondence
- [ ] Provide step-by-step example of environment setup
- [ ] Include visual representation of simulation environment components