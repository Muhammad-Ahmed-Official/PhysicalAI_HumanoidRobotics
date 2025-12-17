---
id: chapter-3-2
title: "Chapter 3.2: Motion Planning and Decision-Making"
description: "Implementing motion planning algorithms and decision-making systems for humanoid robots"
tags: [motion-planning, decision-making, path-planning, robotics]
---

# Chapter 3.2: Motion Planning and Decision-Making

## Introduction

Motion planning and decision-making are critical components for autonomous humanoid robots, enabling them to navigate complex environments and execute tasks efficiently. This chapter explores algorithms and systems for planning robot movements and making decisions based on environmental conditions and task requirements.

## Learning Outcomes

- Students will understand fundamental motion planning algorithms for humanoid robots
- Learners will be able to implement path planning systems
- Readers will be familiar with decision-making frameworks for robotics
- Students will understand how to integrate motion planning with perception systems

## Core Concepts

Motion planning and decision-making in humanoid robots involve several key areas:

1. **Path Planning**: Algorithms to find optimal paths from current location to goal
2. **Trajectory Generation**: Creating smooth, dynamic-feasible motions
3. **Decision Trees**: Structured approaches for robot behavior selection
4. **Reactive vs. Deliberative Systems**: Real-time vs. planned responses
5. **Multi-objective Optimization**: Balancing competing requirements (safety, efficiency, etc.)

Effective motion planning must consider the robot's physical constraints, environmental obstacles, and task requirements simultaneously.

## Simulation Walkthrough

Implementing motion planning and decision-making for a humanoid robot:

<Tabs>
  <TabItem value="python" label="Python Implementation">
    ```python
    import numpy as np
    import rospy
    from geometry_msgs.msg import Pose, Point, Quaternion
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker
    import heapq
    from collections import defaultdict
    import math
    
    class MotionPlanner:
        def __init__(self):
            rospy.init_node('motion_planner')
            
            # Publishers for path visualization
            self.path_pub = rospy.Publisher('/humanoid/planned_path', Path, queue_size=1)
            self.marker_pub = rospy.Publisher('/humanoid/path_markers', Marker, queue_size=1)
            
            # Robot parameters
            self.robot_radius = 0.3  # meters
            self.step_size = 0.1     # meters
            self.max_reach = 1.5     # meters (for manipulation)
            
        def plan_path_rrt(self, start_pose, goal_pose, environment_map):
            """
            Rapidly-exploring Random Tree (RRT) path planning algorithm
            """
            start = (start_pose.position.x, start_pose.position.y)
            goal = (goal_pose.position.x, goal_pose.position.y)
            
            # Initialize tree with start node
            tree = [start]
            parent_map = {start: None}
            
            # RRT algorithm
            for i in range(10000):  # max iterations
                # Sample random point
                rand_point = self.sample_random_point(environment_map)
                
                # Find nearest node in tree
                nearest_node = self.find_nearest_node(tree, rand_point)
                
                # Extend tree toward random point
                new_node = self.extend_toward_point(nearest_node, rand_point, self.step_size)
                
                # Check for collisions
                if self.is_collision_free(nearest_node, new_node, environment_map) and \
                   self.is_in_environment_bounds(new_node, environment_map):
                    
                    # Add new node to tree
                    tree.append(new_node)
                    parent_map[new_node] = nearest_node
                    
                    # Check if new node is near goal
                    if self.distance(new_node, goal) < 0.5:  # goal tolerance
                        # Extract path
                        path = self.extract_path(parent_map, new_node, start, goal)
                        return path
            
            # If no path found
            rospy.logwarn("No path found using RRT")
            return None
        
        def sample_random_point(self, env_map):
            """Sample a random point in the environment"""
            # In practice, bias toward goal occasionally
            if np.random.random() < 0.1:  # 10% chance to sample goal
                return (env_map.goal_x, env_map.goal_y)
            else:
                # Sample random point in environment bounds
                x = np.random.uniform(env_map.min_x, env_map.max_x)
                y = np.random.uniform(env_map.min_y, env_map.max_y)
                return (x, y)
        
        def find_nearest_node(self, tree, point):
            """Find the node in the tree closest to the given point"""
            nearest = tree[0]
            min_dist = self.distance(nearest, point)
            
            for node in tree[1:]:
                dist = self.distance(node, point)
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            
            return nearest
        
        def extend_toward_point(self, from_node, to_point, step_size):
            """Extend from from_node toward to_point by step_size"""
            dx = to_point[0] - from_node[0]
            dy = to_point[1] - from_node[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= step_size:
                return to_point
            else:
                ratio = step_size / distance
                new_x = from_node[0] + dx * ratio
                new_y = from_node[1] + dy * ratio
                return (new_x, new_y)
        
        def is_collision_free(self, node1, node2, env_map):
            """Check if path between node1 and node2 is collision-free"""
            # Simple check - in practice would use more sophisticated collision detection
            # Check several points along the path
            steps = int(self.distance(node1, node2) / 0.1)  # Check every 0.1m
            for i in range(steps + 1):
                t = i / steps
                x = node1[0] + (node2[0] - node1[0]) * t
                y = node1[1] + (node2[1] - node1[1]) * t
                
                if env_map.is_occupied(x, y, self.robot_radius):
                    return False
            
            return True
        
        def extract_path(self, parent_map, goal_node, start, actual_goal):
            """Extract path from tree by backtracking from goal_node"""
            path = [actual_goal]  # Add actual goal position
            current = goal_node
            
            while current != start:
                path.append(current)
                current = parent_map[current]
            
            path.append(start)
            path.reverse()
            return path
        
        def distance(self, p1, p2):
            """Calculate Euclidean distance between two points"""
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def is_in_environment_bounds(self, node, env_map):
            """Check if node is within environment bounds"""
            x, y = node
            return (env_map.min_x <= x <= env_map.max_x and 
                    env_map.min_y <= y <= env_map.max_y)
    
    class DecisionMaker:
        def __init__(self):
            rospy.init_node('decision_maker')
            
            # Initialize decision-making parameters
            self.behavior_weights = {
                'safety': 0.4,
                'efficiency': 0.3,
                'task_completion': 0.3
            }
            
        def make_decision(self, perception_data, goal, robot_state):
            """
            Make decisions based on perception, goals, and robot state
            """
            # Define possible actions
            possible_actions = self.get_possible_actions(perception_data, robot_state)
            
            # Evaluate each action
            best_action = None
            best_score = float('-inf')
            
            for action in possible_actions:
                score = self.evaluate_action(action, perception_data, goal, robot_state)
                if score > best_score:
                    best_score = score
                    best_action = action
            
            return best_action
        
        def get_possible_actions(self, perception_data, robot_state):
            """Get list of possible actions based on current situation"""
            actions = []
            
            # If there are obstacles ahead, consider alternatives
            obstacles_ahead = self.get_obstacles_ahead(perception_data)
            if obstacles_ahead:
                actions.extend(['turn_left', 'turn_right', 'wait', 'move_backward'])
            else:
                actions.extend(['move_forward', 'move_to_goal'])
            
            # If there are objects of interest, consider interaction
            objects_of_interest = self.get_objects_of_interest(perception_data)
            if objects_of_interest:
                actions.extend(['approach_object', 'grasp_object', 'identify_object'])
            
            # Default actions
            actions.extend(['idle', 'scan_environment'])
            
            return actions
        
        def evaluate_action(self, action, perception_data, goal, robot_state):
            """Evaluate an action based on multiple criteria"""
            # Get individual scores for different criteria
            safety_score = self.evaluate_safety(action, perception_data, robot_state)
            efficiency_score = self.evaluate_efficiency(action, goal, robot_state)
            task_completion_score = self.evaluate_task_completion(action, goal, robot_state)
            
            # Combine scores using weighted sum
            total_score = (
                self.behavior_weights['safety'] * safety_score +
                self.behavior_weights['efficiency'] * efficiency_score +
                self.behavior_weights['task_completion'] * task_completion_score
            )
            
            return total_score
        
        def evaluate_safety(self, action, perception_data, robot_state):
            """Evaluate how safe an action is"""
            # Calculate safety score (0-1 scale)
            if action == 'move_forward':
                obstacles = self.get_obstacles_ahead(perception_data)
                if obstacles and any(dist < 0.5 for dist in obstacles.values()):
                    return 0.1  # Very unsafe
            elif action == 'grasp_object':
                # Check if robot can safely reach the object
                object_pos = perception_data.get('object_position')
                if object_pos:
                    reach_dist = self.calculate_distance(robot_state['position'], object_pos)
                    if reach_dist > self.max_reach:
                        return 0.2  # Not safe to attempt
                    
            return 0.9  # Generally safe
        
        def evaluate_efficiency(self, action, goal, robot_state):
            """Evaluate how efficient an action is toward goal"""
            # Calculate efficiency score (0-1 scale)
            if action == 'move_to_goal':
                # Moving directly to goal is most efficient
                return 1.0
            elif action == 'move_forward':
                # Moving forward if it gets closer to goal
                new_pos = self.estimate_new_position(robot_state['position'], 'forward')
                if self.calculate_distance(new_pos, goal) < self.calculate_distance(robot_state['position'], goal):
                    return 0.8
            elif action == 'turn_left' or action == 'turn_right':
                # Turning might help avoid obstacles or get closer to goal
                return 0.5
            
            return 0.3  # Less efficient actions
        
        def evaluate_task_completion(self, action, goal, robot_state):
            """Evaluate how well an action contributes to task completion"""
            # Calculate task completion score (0-1 scale)
            if action == 'move_to_goal' and self.is_at_goal(robot_state['position'], goal):
                return 1.0
            elif action == 'grasp_object':
                # If goal is to grasp an object
                if robot_state['goal_type'] == 'grasp':
                    return 0.9
            elif action == 'identify_object':
                # If goal involves object recognition
                if robot_state['goal_type'] == 'identify':
                    return 0.8
            
            return 0.2  # Less relevant to current task
    
    # Example usage
    if __name__ == '__main__':
        planner = MotionPlanner()
        decision_maker = DecisionMaker()
        
        # Example: Plan path and make decisions
        start_pose = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))
        goal_pose = Pose(Point(5, 5, 0), Quaternion(0, 0, 0, 1))
        env_map = None  # Would be an actual environment map
        
        # Plan path
        path = planner.plan_path_rrt(start_pose, goal_pose, env_map)
        if path:
            print("Path found:", path)
        else:
            print("No path found")
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode Implementation">
    ```
    // Motion Planning and Decision Making System
    
    Class MotionPlanner:
        Initialize:
            - Set robot physical parameters (radius, step size, etc.)
            - Initialize path planning algorithms (RRT*, A*, etc.)
            - Setup visualization publishers
            
        Plan Path(start, goal, environment):
            // Use appropriate planning algorithm based on environment
            if environment is static and known:
                path = run_astar(start, goal, environment)
            elif environment is dynamic or unknown:
                path = run_rrt(start, goal, environment)
            
            // Optimize path for robot dynamics
            optimized_path = smooth_path_dynamically(path)
            
            return optimized_path
        
        Run A* Algorithm(start_node, goal_node, environment):
            open_set = PriorityQueue()
            open_set.add(start_node, heuristic(start_node, goal_node))
            
            closed_set = Set()
            g_score = Map(node -> infinity)  // Cost from start
            g_score[start_node] = 0
            
            f_score = Map(node -> infinity)  // Estimated total cost
            f_score[start_node] = heuristic(start_node, goal_node)
            
            while open_set is not empty:
                current = open_set.pop_lowest_f_score()
                
                if current == goal_node:
                    return reconstruct_path(current)
                
                closed_set.add(current)
                
                for neighbor in get_neighbors(current, environment):
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g_score = g_score[current] + distance(current, neighbor)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                    elif tentative_g_score >= g_score[neighbor]:
                        continue
                    
                    // This path to neighbor is better
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node)
            
            return null  // No path found
    
    Class DecisionMaker:
        Initialize:
            - Set behavior weights (safety, efficiency, task_completion)
            - Initialize decision-making models
            
        Make Decision(perception_data, goal, robot_state):
            // Get possible actions based on current situation
            possible_actions = get_possible_actions(perception_data, robot_state)
            
            // Evaluate each action using utility function
            best_action = null
            best_utility = -infinity
            
            for action in possible_actions:
                utility = evaluate_action(action, perception_data, goal, robot_state)
                
                if utility > best_utility:
                    best_utility = utility
                    best_action = action
            
            return best_action
        
        Evaluate Action(action, perception_data, goal, robot_state):
            // Calculate multi-objective utility score
            safety_score = evaluate_safety(action, perception_data, robot_state)
            efficiency_score = evaluate_efficiency(action, perception_data, goal, robot_state)
            task_score = evaluate_task_relevance(action, goal, robot_state)
            
            // Weighted combination of scores
            utility = (safety_weight * safety_score + 
                      efficiency_weight * efficiency_score + 
                      task_weight * task_score)
            
            return utility
    
    Class Integrated System:
        Initialize:
            - Initialize perception module
            - Initialize motion planner
            - Initialize decision maker
            
        Main Loop:
            while robot_operational:
                // Get current state
                perception_data = get_perception_data()
                robot_state = get_robot_state()
                current_goal = get_current_goal()
                
                // Make high-level decisions
                high_level_action = decision_maker.make_decision(
                    perception_data, current_goal, robot_state)
                
                // Plan motion based on decision
                if high_level_action in ['move_to', 'navigate']:
                    planned_path = motion_planner.plan_path(
                        robot_state.position, current_goal.position, 
                        get_environment_map())
                    
                    // Execute path following
                    execute_path_following(planned_path)
                
                // Handle other types of actions
                elif high_level_action == 'grasp_object':
                    execute_grasping_behavior()
                
                // Sleep to maintain control loop frequency
                sleep(control_loop_time)
    ```
  </TabItem>
</Tabs>

## Visual Explanation

```
[Motion Planning and Decision Making Architecture]

Perception Data
┌─────────────────┐
│ Objects,        │
│ Obstacles,      │
│ Humans, etc.    │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                  Decision Maker                         │
│  ┌─────────────┐   ┌─────────────────────────────────┐ │
│  │ Action      │   │ Evaluate Action Criteria:       │ │
│  │ Selection   │◀──│ • Safety (avoid collisions)     │ │
│  └─────────────┘   │ • Efficiency (shortest path)    │ │
│          │         │ • Task completion (goal reached)│ │
│          ▼         └─────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Possible Actions:                               │   │
│  │ • Move forward                                  │   │
│  │ • Turn left/right                               │   │
│  │ • Approach object                               │   │
│  │ • Grasp object                                  │   │
│  │ • Wait for clear path                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Motion Planner                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Path Planning Algorithms:                       │   │
│  │ • A* for known environments                     │   │
│  │ • RRT* for unknown/dynamic envs                 │   │
│  │ • Trajectory optimization                       │   │
│  └─────────────────────────────────────────────────┘   │
│  │                                                     │
│  │  Start                    Goal                      │
│  │    ● ────────────────────── ●                      │
│  │         ↑              ↑                           │
│  │   Obstacle        Obstacle                        │
│  │                                                     │
│  │ Path: ●─────/\\/─────/\\/─────●                   │
│  │       ↑    Path      ↑                             │
│  │    Smoothed        Optimized                       │
│  └─────────────────────────────────────────────────┬───┘
│                                                    │
└────────────────────────────────────────────────────┼─────┐
                                                     │     │
                                                     ▼     ▼
                                         ┌─────────────────────────┐
                                         │    Robot Control        │
                                         │                         │
                                         │ • Execute planned path  │
                                         │ • Adjust for real-time  │
                                         │   obstacles            │
                                         │ • Maintain balance      │
                                         │ • Execute actions       │
                                         └─────────────────────────┘

Motion planning and decision-making systems work together to
enable the humanoid robot to navigate efficiently while
making intelligent choices based on environmental conditions.
```

## Checklist

- [x] Understand fundamental motion planning algorithms
- [x] Know how to implement path planning systems
- [x] Understand decision-making frameworks
- [ ] Implemented basic A* or RRT path planning
- [ ] Created decision evaluation functions
- [ ] Self-assessment: How would you modify the system to handle dynamic obstacles that move during execution?