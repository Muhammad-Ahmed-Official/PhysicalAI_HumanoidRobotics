# Unity Simulation Examples for Robotics

## Introduction

Unity offers powerful capabilities for robotics simulation using packages like Unity Robotics Hub and Unity ML-Agents. These tools enable researchers to create highly realistic simulation environments with advanced physics, detailed rendering, and sophisticated AI integration.

## Unity Robotics Setup

### Required Packages

- Unity 2021.3 LTS or later
- Unity Robotics Hub
- Unity ML-Agents Toolkit
- ROS# (for ROS/ROS2 communication)

### Basic Robot Controller in Unity

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

### Unity Sensor Simulation

Unity can simulate various sensors using built-in components and custom scripts:

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
            step = (uint)(imageWidth * 3), // 3 bytes per pixel
            data = texture2D.GetRawTextureData<byte>()
        };
        
        // Publish image
        ros.Publish(cameraTopic, imageMsg);
        
        // Clean up
        Destroy(texture2D);
    }
}
```

### LiDAR Simulation in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityLiDAR : MonoBehaviour
{
    [SerializeField] private string lidarTopic = "robot/scan";
    [SerializeField] private int rayCount = 720;
    [SerializeField] private float fieldOfView = 360f;
    [SerializeField] private float maxDistance = 30f;
    
    private ROSConnection ros;
    private RaycastHit[] raycastHits;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<LaserScanMsg>(lidarTopic);
        raycastHits = new RaycastHit[rayCount];
    }
    
    void Update()
    {
        float angleStep = fieldOfView / rayCount;
        float currentAngle = -fieldOfView / 2f;
        
        LaserScanMsg scanMsg = new LaserScanMsg
        {
            header = new std_msgs.HeaderMsg { frame_id = "lidar_frame" },
            angle_min = Mathf.Deg2Rad * (-fieldOfView / 2f),
            angle_max = Mathf.Deg2Rad * (fieldOfView / 2f),
            angle_increment = Mathf.Deg2Rad * angleStep,
            time_increment = 0f,
            scan_time = 0f,
            range_min = 0.1f,
            range_max = maxDistance,
            ranges = new float[rayCount],
            intensities = new float[rayCount]
        };
        
        for (int i = 0; i < rayCount; i++)
        {
            Vector3 direction = Quaternion.Euler(0, currentAngle, 0) * transform.forward;
            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxDistance))
            {
                scanMsg.ranges[i] = hit.distance;
            }
            else
            {
                scanMsg.ranges[i] = float.PositiveInfinity;
            }
            
            currentAngle += angleStep;
        }
        
        ros.Publish(lidarTopic, scanMsg);
    }
}
```

## Unity Physics Simulation

Unity's physics engine (PhysX) provides realistic simulation of physical interactions:

```csharp
using UnityEngine;

public class PhysicsModel : MonoBehaviour
{
    [Header("Physical Properties")]
    [SerializeField] private float mass = 1f;
    [SerializeField] private float friction = 0.5f;
    [SerializeField] private float bounciness = 0.1f;
    
    [Header("Simulation Parameters")]
    [SerializeField] private bool useGravity = true;
    [SerializeField] private float gravityScale = 1f;
    
    private Rigidbody rb;
    private PhysicMaterial physicMaterial;
    
    void Start()
    {
        SetupRigidbody();
        SetupPhysicsMaterial();
    }
    
    private void SetupRigidbody()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }
        
        rb.mass = mass;
        rb.useGravity = useGravity;
        rb.gravityScale = gravityScale;
    }
    
    private void SetupPhysicsMaterial()
    {
        if (GetComponent<Collider>() != null)
        {
            physicMaterial = new PhysicMaterial("RobotMaterial");
            physicMaterial.staticFriction = friction;
            physicMaterial.dynamicFriction = friction;
            physicMaterial.bounceCombine = PhysicMaterialCombine.Average;
            physicMaterial.frictionCombine = PhysicMaterialCombine.Average;
            
            GetComponent<Collider>().material = physicMaterial;
        }
    }
}
```

## Unity ML-Agents for Robot Training

Unity ML-Agents allows training AI behaviors for robots:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RobotAgent : Agent
{
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 1f;
    
    private Rigidbody rb;
    
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }
    
    public override void OnEpisodeBegin()
    {
        // Reset position
        transform.position = new Vector3(Random.Range(-5f, 5f), 0.5f, Random.Range(-5f, 5f));
        
        // Move target to new random position
        target.position = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe position relative to target
        sensor.AddObservation(transform.position - target.position);
        
        // Observe velocity
        sensor.AddObservation(rb.velocity);
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        
        rb.AddForce(new Vector3(moveX, 0, moveZ) * moveSpeed);
        
        // Reward based on distance to target
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        SetReward(-distanceToTarget * 0.01f); // Negative reward for distance
        
        // End episode when close to target
        if (distanceToTarget < 1.0f)
        {
            SetReward(1.0f);
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
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

## Unity Simulation Environment Setup

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

## Unity-ROS Integration

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;

public class UnityROSIntegration : MonoBehaviour
{
    [Header("ROS Settings")]
    [SerializeField] private string rosIP = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;
    
    [Header("Robot Topics")]
    [SerializeField] private string jointStatesTopic = "joint_states";
    [SerializeField] private string cmdVelTopic = "cmd_vel";
    
    private ROSConnection ros;
    
    void Start()
    {
        // Configure ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
        
        // Register publishers and subscribers
        ros.RegisterPublisher<Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.JointStateMsg>(jointStatesTopic);
        ros.RegisterSubscriber<Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.TwistMsg>(cmdVelTopic, OnCmdVelReceived);
    }
    
    private void OnCmdVelReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.TwistMsg cmdVel)
    {
        // Process velocity command from ROS
        Debug.Log($"Received velocity command: linear={cmdVel.linear}, angular={cmdVel.angular}");
        
        // Update robot movement based on command
        ProcessVelocityCommand(cmdVel);
    }
    
    private void ProcessVelocityCommand(Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.TwistMsg cmd)
    {
        // Implement command processing logic
        // This would typically control robot joints or movement
    }
    
    void Update()
    {
        // Publish joint states periodically
        PublishJointStates();
    }
    
    private void PublishJointStates()
    {
        // Create and publish joint state message
        var jointState = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.JointStateMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new TimeMsg { sec = (int)Time.time, nanosec = (int)((Time.time % 1) * 1e9) },
                frame_id = "base_link"
            },
            name = System.Array.Empty<string>(),
            position = System.Array.Empty<double>(),
            velocity = System.Array.Empty<double>(),
            effort = System.Array.Empty<double>()
        };
        
        ros.Publish(jointStatesTopic, jointState);
    }
}
```

## Visualization in Unity

Unity provides powerful visualization tools that can display robot state, sensor data, and environment information:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class UnityVisualization : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField] private Text statusText;
    [SerializeField] private Text positionText;
    [SerializeField] private Text sensorDataText;
    
    [Header("Visualization Settings")]
    [SerializeField] private GameObject pathPrefab;
    [SerializeField] private GameObject targetMarker;
    [SerializeField] private GameObject robotModel;
    
    void Update()
    {
        UpdateStatusDisplay();
        UpdatePositionDisplay();
        UpdateSensorDisplay();
    }
    
    private void UpdateStatusDisplay()
    {
        if (statusText != null)
        {
            // Show simulation status
            statusText.text = $"Simulation Time: {Time.time:F2}s\nStatus: Running";
        }
    }
    
    private void UpdatePositionDisplay()
    {
        if (positionText != null && robotModel != null)
        {
            Vector3 pos = robotModel.transform.position;
            positionText.text = $"Position: X={pos.x:F2}, Y={pos.y:F2}, Z={pos.z:F2}";
        }
    }
    
    private void UpdateSensorDisplay()
    {
        if (sensorDataText != null)
        {
            // Display simulated sensor data
            sensorDataText.text = $"Camera: Active\nLiDAR: 720 rays\nIMU: OK";
        }
    }
    
    public void VisualizePath(Vector3[] waypoints)
    {
        // Create visual path representation
        for (int i = 0; i < waypoints.Length - 1; i++)
        {
            GameObject pathSegment = Instantiate(pathPrefab);
            LineRenderer line = pathSegment.GetComponent<LineRenderer>();
            if (line != null)
            {
                line.SetPosition(0, waypoints[i]);
                line.SetPosition(1, waypoints[i+1]);
            }
        }
    }
}
```

## Unity Simulation Workflow

To set up a complete Unity simulation for robotics:

1. **Environment Setup**: Create your 3D environment with appropriate lighting, physics materials, and colliders
2. **Robot Model Import**: Import your robot model (preferably as URDF via Unity Robotics package)
3. **Sensor Implementation**: Add sensor scripts to simulate camera, LiDAR, IMU, etc.
4. **Control System**: Implement control interfaces that communicate with ROS or standalone AI
5. **Testing Framework**: Create scenarios to test robot behavior in various conditions
6. **Integration**: Connect to ROS/ROS2 using Unity's ROS TCP Connector

Unity's high-fidelity rendering and physics simulation make it ideal for testing complex perception systems and visual scenarios that require photorealistic rendering or complex lighting conditions.