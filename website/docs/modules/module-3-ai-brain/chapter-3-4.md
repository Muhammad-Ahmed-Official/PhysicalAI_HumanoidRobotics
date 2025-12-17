---
id: chapter-3-4
title: "Chapter 3.4: Simulation-to-Real Bridging Techniques"
description: "Techniques for transferring learned behaviors from simulation to physical robots"
tags: [sim-to-real, domain-randomization, transfer-learning, robotics]
---

# Chapter 3.4: Simulation-to-Real Bridging Techniques

## Introduction

Simulation-to-real (Sim-to-Real) transfer is a critical challenge in humanoid robotics, enabling behaviors learned in simulation to be effectively applied to physical robots. This chapter explores various techniques to bridge the reality gap between simulated and physical environments, ensuring that skills developed in simulation can be successfully deployed on real robots.

## Learning Outcomes

- Students will understand the challenges of simulation-to-real transfer
- Learners will be able to implement domain randomization techniques
- Readers will be familiar with system identification methods
- Students will know how to validate and adapt simulation models for real robots

## Core Concepts

Simulation-to-real bridging involves several key areas:

1. **Reality Gap**: The difference between simulated and real environments that affects transfer performance
2. **Domain Randomization**: Randomizing simulation parameters to improve transferability
3. **System Identification**: Calibrating simulation models to match physical robot behavior
4. **Adaptive Control**: Adjusting robot behaviors based on real-world feedback
5. **Progressive Training**: Gradually increasing simulation fidelity during training

These techniques are essential for the simulation-first approach to humanoid robotics development to be practical.

## Simulation Walkthrough

Implementing simulation-to-real bridging techniques:

<Tabs>
  <TabItem value="python" label="Python Implementation">
    ```python
    import numpy as np
    import rospy
    import random
    from std_msgs.msg import Float32
    from sensor_msgs.msg import JointState, Imu
    from geometry_msgs.msg import Vector3
    import pickle
    from scipy import stats
    
    class DomainRandomization:
        def __init__(self):
            rospy.init_node('domain_randomizer')
            
            # Physics parameters to randomize
            self.physics_params = {
                'gravity': {'low': -9.81*1.1, 'high': -9.81*0.9, 'type': 'uniform'},
                'friction': {'low': 0.3, 'high': 0.9, 'type': 'uniform'},
                'mass_variance': {'low': 0.9, 'high': 1.1, 'type': 'uniform'},
                'com_offset': {'low': -0.02, 'high': 0.02, 'type': 'uniform'},
            }
            
            # Sensor parameters to randomize
            self.sensor_params = {
                'camera_noise': {'low': 0.001, 'high': 0.01, 'type': 'uniform'},
                'imu_drift': {'low': -0.001, 'high': 0.001, 'type': 'uniform'},
                'delay_min': 0.001,
                'delay_max': 0.010,
            }
            
            # Initialize randomization bounds
            self.current_params = self.randomize_all()
            
        def randomize_all(self):
            """Randomize all parameters according to defined distributions"""
            randomized_params = {}
            
            for param_name, config in self.physics_params.items():
                if config['type'] == 'uniform':
                    randomized_params[param_name] = np.random.uniform(
                        config['low'], config['high']
                    )
                elif config['type'] == 'normal':
                    randomized_params[param_name] = np.random.normal(
                        config['mean'], config['std']
                    )
            
            for param_name, config in self.sensor_params.items():
                if param_name in ['camera_noise', 'imu_drift']:
                    if config['type'] == 'uniform':
                        randomized_params[param_name] = np.random.uniform(
                            config['low'], config['high']
                        )
            
            return randomized_params
        
        def get_current_params(self):
            """Get current randomized parameters"""
            return self.current_params
        
        def update_parameters(self):
            """Update parameters to new randomized values"""
            self.current_params = self.randomize_all()
            return self.current_params
    
    class SystemIdentification:
        def __init__(self):
            rospy.init_node('system_id')
            
            # Subscribe to real robot data
            self.joint_state_sub = rospy.Subscriber(
                '/robot/joint_states', JointState, self.joint_state_callback
            )
            self.imu_sub = rospy.Subscriber(
                '/robot/imu', Imu, self.imu_callback
            )
            
            # Store data for system identification
            self.joint_data = []
            self.imu_data = []
            self.real_commands = []
            self.simulation_predictions = []
            
            # Model parameters to identify
            self.model_params = {
                'mass': {'estimated': 1.0, 'real': None},
                'inertia': {'estimated': 0.1, 'real': None},
                'friction_coeff': {'estimated': 0.1, 'real': None},
                'com_offset': {'estimated': [0, 0, 0], 'real': None}
            }
            
        def joint_state_callback(self, data):
            """Store real robot joint state data"""
            self.joint_data.append({
                'position': data.position,
                'velocity': data.velocity,
                'effort': data.effort,
                'timestamp': rospy.Time.now()
            })
        
        def imu_callback(self, data):
            """Store real robot IMU data"""
            self.imu_data.append({
                'linear_acceleration': [data.linear_acceleration.x, 
                                       data.linear_acceleration.y, 
                                       data.linear_acceleration.z],
                'angular_velocity': [data.angular_velocity.x, 
                                    data.angular_velocity.y, 
                                    data.angular_velocity.z],
                'orientation': [data.orientation.x, 
                               data.orientation.y, 
                               data.orientation.z, 
                               data.orientation.w],
                'timestamp': rospy.Time.now()
            })
        
        def collect_data(self, duration=60):
            """Collect data for system identification"""
            rospy.loginfo(f"Collecting data for system identification for {duration} seconds")
            
            start_time = rospy.Time.now()
            while (rospy.Time.now() - start_time).to_sec() < duration:
                rospy.sleep(0.1)  # Collect at 10Hz
            
            rospy.loginfo(f"Collected {len(self.joint_data)} joint samples and {len(self.imu_data)} IMU samples")
        
        def identify_parameters(self):
            """Identify system parameters by comparing simulation and real data"""
            # This is a simplified example - in practice this would involve more complex
            # optimization algorithms to match simulated behavior to real behavior
            
            # Calculate discrepancies between simulation predictions and real data
            if len(self.joint_data) > 100:  # Need sufficient data
                # Example: estimate actual mass based on acceleration response
                positions = [sample['position'][0] for sample in self.joint_data]  # Example for first joint
                velocities = [sample['velocity'][0] for sample in self.joint_data]
                
                # Calculate approximate acceleration
                accelerations = []
                for i in range(2, len(velocities)):
                    dt = 0.1  # Assume 10Hz data collection
                    accel = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(accel)
                
                # Simplified system ID: adjust mass estimate based on acceleration
                # differences between simulation and reality
                avg_accel = sum(accelerations) / len(accelerations)
                
                # Update mass estimate (this is a simplified example)
                if avg_accel != 0:
                    self.model_params['mass']['real'] = self.model_params['mass']['estimated'] * 9.81 / abs(avg_accel)
            
            rospy.loginfo("System identification completed")
            return self.model_params
    
    class Sim2RealTransferValidator:
        def __init__(self):
            rospy.init_node('sim2real_validator')
            
            # Performance metrics
            self.metrics = {
                'tracking_accuracy': [],
                'execution_time': [],
                'energy_efficiency': [],
                'stability': []
            }
            
            # Publishers for validation results
            self.accuracy_pub = rospy.Publisher('/sim2real/tracking_accuracy', Float32, queue_size=1)
            self.stability_pub = rospy.Publisher('/sim2real/stability', Float32, queue_size=1)
            
        def validate_transfer(self, sim_trajectory, real_trajectory):
            """Validate transfer performance by comparing trajectories"""
            if len(sim_trajectory) == 0 or len(real_trajectory) == 0:
                rospy.logwarn("Empty trajectories for validation")
                return None
            
            # Calculate tracking accuracy (average distance between trajectories)
            min_len = min(len(sim_trajectory), len(real_trajectory))
            total_error = 0.0
            
            for i in range(min_len):
                sim_point = sim_trajectory[i]
                real_point = real_trajectory[i]
                
                # Calculate distance in joint space
                error = np.linalg.norm(np.array(sim_point) - np.array(real_point))
                total_error += error
            
            avg_accuracy = total_error / min_len
            self.metrics['tracking_accuracy'].append(avg_accuracy)
            
            # Publish accuracy metric
            accuracy_msg = Float32()
            accuracy_msg.data = avg_accuracy
            self.accuracy_pub.publish(accuracy_msg)
            
            # Calculate stability metric based on joint oscillations
            stability = self.calculate_stability(real_trajectory)
            self.metrics['stability'].append(stability)
            
            stability_msg = Float32()
            stability_msg.data = stability
            self.stability_pub.publish(stability_msg)
            
            return {
                'tracking_accuracy': avg_accuracy,
                'stability': stability,
                'transfer_success': avg_accuracy < 0.1 and stability > 0.7  # Thresholds
            }
        
        def calculate_stability(self, trajectory):
            """Calculate stability metric based on joint oscillations"""
            if len(trajectory) < 10:
                return 0.0  # Need sufficient data points
            
            # Calculate joint velocity variance (less variance = more stable)
            velocities = []
            for i in range(1, len(trajectory)):
                dt = 0.1  # Assume 10Hz control
                vel = (np.array(trajectory[i]) - np.array(trajectory[i-1])) / dt
                velocities.append(vel)
            
            vel_array = np.array(velocities)
            stability = 1.0 / (1.0 + np.std(vel_array))  # Higher std dev = less stable
            
            return min(stability, 1.0)  # Clamp to [0, 1]
        
        def get_transfer_metrics(self):
            """Get summary of transfer metrics"""
            if not self.metrics['tracking_accuracy']:
                return "No validation data available"
            
            avg_accuracy = sum(self.metrics['tracking_accuracy']) / len(self.metrics['tracking_accuracy'])
            avg_stability = sum(self.metrics['stability']) / len(self.metrics['stability'])
            
            return {
                'average_tracking_accuracy': avg_accuracy,
                'average_stability': avg_stability,
                'total_validations': len(self.metrics['tracking_accuracy']),
                'success_rate': sum(1 for m in self.metrics['tracking_accuracy'] if m < 0.1) / len(self.metrics['tracking_accuracy'])
            }
    
    class AdaptiveBehaviorTransfer:
        def __init__(self):
            rospy.init_node('adaptive_transfer')
            
            # Parameters for adaptation
            self.adaptation_params = {
                'learning_rate': 0.01,
                'max_correction': 0.1,  # rad for joint angles
                'min_performance': 0.7,  # minimum performance threshold
            }
            
            # Correction factors for each joint
            self.correction_factors = {
                'left_hip_joint': 0.0,
                'right_hip_joint': 0.0,
                'left_knee_joint': 0.0,
                'right_knee_joint': 0.0,
                'left_ankle_joint': 0.0,
                'right_ankle_joint': 0.0,
            }
            
        def adapt_behavior(self, original_behavior, performance_feedback):
            """Adapt behavior based on performance feedback from real robot"""
            if performance_feedback < self.adaptation_params['min_performance']:
                # Apply corrections to improve performance
                for joint, correction in self.correction_factors.items():
                    # Adjust correction based on performance error
                    performance_error = self.adaptation_params['min_performance'] - performance_feedback
                    adjustment = self.adaptation_params['learning_rate'] * performance_error
                    
                    # Apply bounded adjustment
                    self.correction_factors[joint] += np.clip(
                        adjustment, 
                        -self.adaptation_params['max_correction'], 
                        self.adaptation_params['max_correction']
                    )
                
                # Create adapted behavior with corrections
                adapted_behavior = []
                for trajectory_point in original_behavior:
                    # Apply corrections to each point in the trajectory
                    corrected_point = [
                        pos + self.correction_factors.get(f'joint_{i}', 0.0)
                        for i, pos in enumerate(trajectory_point)
                    ]
                    adapted_behavior.append(corrected_point)
                
                rospy.loginfo(f"Adapted behavior with performance improvement: {performance_feedback:.3f}")
                return adapted_behavior
            else:
                # Performance is acceptable, return original behavior
                return original_behavior
    
    # Example usage of the system
    if __name__ == '__main__':
        # Initialize components
        domain_rand = DomainRandomization()
        sys_id = SystemIdentification()
        validator = Sim2RealTransferValidator()
        adapter = AdaptiveBehaviorTransfer()
        
        rospy.loginfo("Sim-to-Real transfer system initialized")
        
        # Example: Collect real robot data for system identification
        rospy.sleep(2)  # Wait for subscribers to connect
        sys_id.collect_data(duration=30)  # Collect for 30 seconds
        identified_params = sys_id.identify_parameters()
        
        rospy.loginfo(f"Identified parameters: {identified_params}")
        
        # Example: Validate a trajectory transfer
        sim_traj = [[0.1*i, 0.2*i, 0.15*i] for i in range(50)]  # Example trajectory
        real_traj = [[0.09*i, 0.19*i, 0.14*i] for i in range(50)]  # Slightly different real trajectory
        
        validation_result = validator.validate_transfer(sim_traj, real_traj)
        rospy.loginfo(f"Validation result: {validation_result}")
        
        # Example: Adapt behavior based on feedback
        original_behavior = [[0.1, 0.2, 0.15] for _ in range(10)]
        adapted_behavior = adapter.adapt_behavior(original_behavior, 0.6)  # Low performance feedback
        
        rospy.loginfo(f"Original: {original_behavior[0]}, Adapted: {adapted_behavior[0]}")
        
        # Keep nodes running
        rospy.spin()
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode Implementation">
    ```
    // Simulation-to-Real Transfer System
    
    Class DomainRandomization:
        Initialize:
            - Define physics parameters to randomize (mass, friction, etc.)
            - Define sensor parameters to randomize (noise, delay, etc.)
            - Set randomization ranges and distributions
        
        Randomize Parameters():
            // Randomly adjust simulation parameters within defined ranges
            for each parameter in physics_params:
                if parameter.type == "uniform":
                    parameter.value = random.uniform(parameter.low, parameter.high)
                else if parameter.type == "normal":
                    parameter.value = random.normal(parameter.mean, parameter.std)
            
            for each parameter in sensor_params:
                parameter.value = random.uniform(parameter.low, parameter.high)
        
        Apply To Simulation():
            // Apply randomized parameters to physics engine
            set_physics_gravity(randomized_gravity)
            set_joint_friction(randomized_friction)
            set_sensor_noise(randomized_noise)
    
    Class SystemIdentification:
        Initialize:
            - Setup subscribers for real robot data (joint states, IMU, etc.)
            - Initialize data collection buffers
            - Set model parameters with initial estimates
        
        Collect Data(duration):
            // Collect synchronized data from real robot
            while collection_time < duration:
                store_joint_states()
                store_imu_data()
                store_command_data()
                sleep(sampling_rate)
        
        Identify Parameters():
            // Compare simulation predictions with real robot data
            // to adjust model parameters
            
            // Example algorithm:
            for each joint in robot.joints:
                sim_response = simulate_joint_response(model_params[joint])
                real_response = measure_joint_response()
                
                // Calculate parameter correction
                correction = compute_parameter_correction(
                    sim_response, real_response, model_params[joint])
                
                model_params[joint].value += correction
        
        Update Simulation():
            // Apply identified parameters to simulation
            for each parameter in model_params:
                update_simulation_parameter(parameter.name, parameter.real_value)
    
    Class Sim2RealValidator:
        Initialize:
            - Setup publishers for validation metrics
            - Initialize metric buffers
        
        Validate Transfer(sim_trajectory, real_trajectory):
            // Compare simulated and real execution trajectories
            if length(sim_trajectory) == 0 or length(real_trajectory) == 0:
                return error("Empty trajectories")
            
            // Calculate tracking accuracy
            total_error = 0
            min_length = min(length(sim_trajectory), length(real_trajectory))
            
            for i from 0 to min_length:
                error = distance(sim_trajectory[i], real_trajectory[i])
                total_error += error
            
            tracking_accuracy = total_error / min_length
            
            // Calculate stability metric
            stability = calculate_stability(real_trajectory)
            
            return {
                tracking_accuracy: tracking_accuracy,
                stability: stability,
                transfer_success: tracking_accuracy < threshold && stability > threshold
            }
        
        Calculate Stability(trajectory):
            // Calculate stability based on joint oscillations
            velocities = compute_velocities(trajectory)
            velocity_variance = compute_variance(velocities)
            stability = 1.0 / (1.0 + velocity_variance)
            return clamp(stability, 0, 1)
    
    Class AdaptiveTransfer:
        Initialize:
            - Set adaptation parameters (learning rate, correction limits, etc.)
            - Initialize correction factors for joints
        
        Adapt Behavior(original_behavior, performance_feedback):
            if performance_feedback < min_performance_threshold:
                // Calculate needed adjustments
                performance_error = min_performance_threshold - performance_feedback
                adjustment = learning_rate * performance_error
                
                // Update correction factors
                for each joint in correction_factors:
                    correction_factors[joint] += clamp(adjustment, -max_correction, max_correction)
                
                // Apply corrections to behavior
                adapted_behavior = []
                for each trajectory_point in original_behavior:
                    corrected_point = apply_corrections(trajectory_point, correction_factors)
                    adapted_behavior.append(corrected_point)
                
                return adapted_behavior
            else:
                return original_behavior  // Performance is acceptable
    
    Class ProgressiveTraining:
        Initialize:
            - Define training phases with increasing fidelity
            - Set phase transition criteria
        
        Execute Progressive Training():
            // Train in multiple phases with increasing realism
            for each phase in training_phases:
                // Adjust simulation fidelity
                adjust_simulation_fidelity(phase.fidelity_level)
                
                // Train behavior in current phase
                trained_behavior = train_behavior_in_phase(phase)
                
                // Validate transfer to next higher fidelity
                validation_result = validate_transfer_to_phase(trained_behavior, phase.next)
                
                if validation_result.success:
                    continue_to_next_phase()
                else:
                    // Remain in current phase and continue training
                    continue
    
    // Main integration example
    sim2real_system = New Sim2RealSystem()
    
    // Step 1: Perform system identification
    real_robot_data = collect_robot_data(duration=60s)
    identified_params = system_identification.identify_parameters(real_robot_data)
    update_simulation_with(identified_params)
    
    // Step 2: Apply domain randomization during training
    for episode in training_episodes:
        randomized_params = domain_randomization.randomize_parameters()
        apply_to_simulation(randomized_params)
        
        trained_policy = train_policy_in_simulation()
    
    // Step 3: Validate transfer to reality
    validation_result = sim2real_validator.validate_transfer(
        simulated_trajectory, real_robot_trajectory)
    
    if not validation_result.success:
        // Adapt behavior based on feedback
        adapted_policy = adaptive_transfer.adapt_behavior(
            original_policy, validation_result.performance)
    
    return final_policy_for_real_robot
    ```
  </TabItem>
</Tabs>

## Visual Explanation

```
[Simulation-to-Real Transfer Process]

Simulation Environment
┌─────────────────────────────────────────────────────────┐
│  ┌─────────────────┐    Domain Randomization           │
│  │  Base Model    │  ┌─────────────────────────────┐   │
│  │ (Robot URDF)   │  │ • Vary friction coefficients│   │
│  │                │  │ • Randomize masses          │   │
│  │                │  │ • Adjust sensor noise       │   │
│  │                │  │ • Change environmental      │   │
│  └─────────────────┘  │   properties               │   │
│            │          └─────────────────────────────┘   │
│            ▼                                          ▲ │
│  ┌─────────────────┐    ┌─────────────────────────┐   │
│  │ Randomized      │────│ Training Loop          │   │
│  │ Simulations     │    │ • Execute behaviors    │   │
│  │ (Many variants) │    │ • Collect data         │   │
│  └─────────────────┘    │ • Optimize policies    │   │
│                         └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│     System Identification & Validation                  │
│  ┌─────────────────┐    ┌─────────────────────────┐   │
│  │ Real Robot      │    │ Performance Metrics     │   │
│  │ Data Collection │───▶│ • Trajectory accuracy   │   │
│  │ • Joint states  │    │ • Stability measures    │   │
│  │ • IMU data      │    │ • Energy efficiency     │   │
│  │ • Sensor data   │    │ • Success rates         │   │
│  └─────────────────┘    └─────────────────────────┘   │
└───────────────────────────────┬─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│        Adaptive Transfer & Correction                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Original Behavior     Adapted Behavior          │   │
│  │   (From Simulation)    (For Real Robot)         │   │
│  │ ┌─────────────────┐  ┌─────────────────────────┐ │ │
│  │ │ • Joint angles  │  │ • Corrected angles    │ │ │
│  │ │ • Timing        │  │ • Adjusted timing     │ │ │
│  │ │ • Force profiles│  │ • Compensated forces  │ │ │
│  │ └─────────────────┘  └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────┬─┘ │
│                                                    │   │
└────────────────────────────────────────────────────┼───┘
                                                     │
                                             ┌───────▼────────┐
                                             │ Physical Robot │
                                             │ Execution      │
                                             └────────────────┘

Simulation-to-real transfer uses domain randomization, system
identification, and adaptive techniques to bridge the gap between
virtual and physical robot execution.
```

## Checklist

- [x] Understand the challenges of simulation-to-real transfer
- [x] Know how to implement domain randomization techniques
- [x] Understand system identification methods
- [ ] Implemented parameter validation for transfer
- [ ] Created adaptive behavior correction system
- [ ] Self-assessment: How would you design a progressive training approach that gradually increases simulation fidelity?