---
sidebar_label: 'Chapter 4.2: Action Planning from Language Commands'
---

# Chapter 4.2: Action Planning from Language Commands

## Introduction

Action planning from language commands represents the critical bridge between understanding human instructions and executing appropriate robotic behaviors. This process involves translating abstract linguistic descriptions into concrete, executable action sequences that account for environmental constraints, robot capabilities, and task requirements. In Vision-Language-Action (VLA) systems, action planning must seamlessly integrate natural language understanding with perception and control to enable robots to perform complex, multi-step tasks based on human instructions.

The challenge lies in decomposing high-level language commands into primitive robot actions while considering the rich context of the environment and the dynamic nature of real-world tasks. For humanoid robots, action planning must account for complex kinematics, balance requirements, and the need for natural, human-like movements. Effective action planning systems must handle ambiguity in language, generate robust plans that adapt to changing conditions, and ensure safe, reliable execution.

This chapter explores methodologies for translating language commands into executable robot actions, covering symbolic planning approaches, neural planning methods, and hybrid systems that leverage both for effective command execution.

## Learning Objectives

By the end of this chapter, you will be able to:

- Design action planning systems that translate language commands into primitive robot actions
- Implement symbolic planners that handle multi-step tasks and constraints
- Develop neural planning approaches for complex manipulation and navigation
- Integrate language understanding with action selection and execution
- Handle uncertainty and adapt planning during execution
- Evaluate the effectiveness of language-to-action translation systems

## Explanation

### Language-to-Action Translation Process

The conversion of natural language commands to robot actions involves several stages:

1. **Command Parsing**: Breaking down the natural language into actionable components (verbs, objects, locations, constraints)

2. **World State Interpretation**: Understanding the current state of the environment and how it relates to the command

3. **Action Selection**: Choosing appropriate primitive actions that achieve the commanded goal

4. **Plan Construction**: Sequencing actions to form a coherent plan that handles dependencies and constraints

5. **Execution and Monitoring**: Executing the plan and adapting as needed based on feedback

### Types of Action Plans

Robotic action plans can be categorized based on their structure and complexity:

- **Primitive Actions**: Basic robot capabilities (move, grasp, place, etc.)
- **Compound Actions**: Combinations of primitives (fetch, deliver, follow)
- **Task-Level Actions**: High-level behaviors (set table, clean room)
- **Contingent Plans**: Plans with alternative branches for handling failures

### Planning Challenges

Action planning from language commands faces several challenges:

- **Language Ambiguity**: Commands often contain underspecified or ambiguous elements
- **World Understanding**: The robot must maintain an accurate model of its environment
- **Embodied Constraints**: Robot kinematics and dynamics limit possible actions
- **Dynamic Environments**: Conditions may change during plan execution
- **Human Interaction**: Plans may need to adapt based on human feedback

### Planning Approaches

Different approaches to action planning include:

- **Symbolic Planning**: Using formal logic and classical planning algorithms
- **Neural Planning**: Learning action sequences from data using neural networks
- **Hybrid Planning**: Combining symbolic and neural approaches
- **Reactive Planning**: Simple condition-action rules for specific situations

## Example Walkthrough

Consider implementing an action planning system for a humanoid robot that needs to execute complex tasks from language commands, such as "Bring me the red mug from the kitchen counter and place it on the table."

**Step 1: Command Parsing and Semantic Grounding**

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time

@dataclass
class ParsedCommand:
    """Represents a parsed language command"""
    action_sequence: List[str]
    objects: List[Dict[str, Any]]
    locations: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]

@dataclass
class RobotAction:
    """Represents an executable robot action"""
    name: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    duration: float = 1.0  # Estimated duration in seconds

class CommandParser:
    """Parses natural language commands into structured representations"""
    
    def __init__(self):
        self.action_vocabulary = {
            'navigate': ['go', 'move', 'walk', 'go to', 'travel to'],
            'grasp': ['pick up', 'take', 'grasp', 'grab', 'lift'],
            'place': ['place', 'put', 'set down', 'release'],
            'find': ['find', 'locate', 'search for', 'look for'],
            'follow': ['follow', 'accompany', 'go with'],
            'transport': ['bring', 'deliver', 'carry', 'move'],
            'inspect': ['look at', 'examine', 'check']
        }
        
        self.object_types = {
            'container': ['mug', 'cup', 'bowl', 'glass', 'box', 'basket'],
            'furniture': ['table', 'counter', 'desk', 'chair', 'couch'],
            'food': ['apple', 'banana', 'sandwich', 'snack'],
            'tool': ['fork', 'spoon', 'knife', 'pen', 'book']
        }
        
        self.location_types = {
            'room': ['kitchen', 'living room', 'bedroom', 'office', 'bathroom'],
            'surface': ['table', 'counter', 'desk', 'shelf', 'couch'],
            'furniture': ['cabinet', 'drawer', 'refrigerator', 'oven']
        }
    
    def parse_command(self, command: str) -> ParsedCommand:
        """Parse a natural language command into structured form"""
        cmd_lower = command.lower()
        
        # Identify actions
        actions = []
        for action, synonyms in self.action_vocabulary.items():
            for synonym in synonyms:
                if synonym in cmd_lower:
                    actions.append(action)
        
        # Identify objects
        objects = []
        for obj_type, obj_names in self.object_types.items():
            for name in obj_names:
                if name in cmd_lower:
                    # Check for attributes like color or size
                    attributes = self.extract_attributes(cmd_lower, name)
                    objects.append({
                        'type': obj_type,
                        'name': name,
                        'attributes': attributes,
                        'position': None  # Will be determined by perception
                    })
        
        # Identify locations
        locations = []
        for loc_type, loc_names in self.location_types.items():
            for name in loc_names:
                if name in cmd_lower:
                    locations.append({
                        'type': loc_type,
                        'name': name,
                        'coordinates': None  # Will be determined by navigation system
                    })
        
        # Identify constraints
        constraints = []
        if 'carefully' in cmd_lower or 'gently' in cmd_lower:
            constraints.append({'type': 'handling', 'value': 'gentle'})
        if 'quickly' in cmd_lower or 'fast' in cmd_lower:
            constraints.append({'type': 'speed', 'value': 'fast'})
        
        # Determine dependencies based on command structure
        dependencies = self.extract_dependencies(actions, objects, locations)
        
        return ParsedCommand(
            action_sequence=actions,
            objects=objects,
            locations=locations,
            constraints=constraints,
            dependencies=dependencies
        )
    
    def extract_attributes(self, command: str, obj_name: str) -> List[str]:
        """Extract attributes like color, size for an object"""
        attributes = []
        
        # Find the object in the command
        obj_start = command.find(obj_name)
        if obj_start == -1:
            return attributes
        
        # Look for descriptors in a small window around the object
        context_start = max(0, obj_start - 30)
        context_end = min(len(command), obj_start + len(obj_name) + 20)
        context = command[context_start:context_end]
        
        # Extract colors
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown']
        for color in colors:
            if color in context:
                attributes.append(color)
        
        # Extract sizes
        sizes = ['small', 'large', 'big', 'little', 'medium', 'huge', 'tiny']
        for size in sizes:
            if size in context:
                attributes.append(size)
        
        return attributes
    
    def extract_dependencies(self, actions: List[str], objects: List[Dict], locations: List[Dict]) -> List[Dict]:
        """Extract dependencies between actions, objects, and locations"""
        dependencies = []
        
        # Navigation must happen before manipulation
        if 'navigate' in actions and any(action in ['grasp', 'place'] for action in actions):
            dependencies.append({
                'from': 'navigate',
                'to': [a for a in actions if a in ['grasp', 'place']],
                'type': 'spatial_prerequisite'
            })
        
        # Find must happen before grasp
        if 'find' in actions and 'grasp' in actions:
            dependencies.append({
                'from': 'find',
                'to': 'grasp',
                'type': 'perception_prerequisite'
            })
        
        # Transport involves both grasp and place
        if 'transport' in actions or ('grasp' in actions and 'place' in actions):
            dependencies.append({
                'from': 'grasp',
                'to': 'place',
                'type': 'sequential_dependency'
            })
        
        return dependencies

class ActionPlanner:
    """Plans robot actions based on parsed commands and world state"""
    
    def __init__(self, perception_system, navigation_system, manipulation_system):
        self.perception = perception_system
        self.navigation = navigation_system
        self.manipulation = manipulation_system
        self.parser = CommandParser()
        
        # Define primitive robot actions
        self.primitive_actions = {
            'move_to': {
                'parameters': ['destination'],
                'preconditions': ['robot_mobile'],
                'effects': ['robot_at_destination']
            },
            'detect_object': {
                'parameters': ['object_type', 'search_location'],
                'preconditions': ['camera_operational'],
                'effects': ['object_location_known']
            },
            'grasp_object': {
                'parameters': ['object_id', 'grasp_pose'],
                'preconditions': ['object_reachable', 'gripper_free'],
                'effects': ['object_grasped', 'gripper_occupied']
            },
            'place_object': {
                'parameters': ['placement_position', 'release_type'],
                'preconditions': ['object_grasped'],
                'effects': ['object_placed', 'gripper_free']
            },
            'navigate_to_region': {
                'parameters': ['region_name'],
                'preconditions': ['robot_mobile'],
                'effects': ['robot_in_region']
            }
        }
    
    def plan_from_command(self, command: str) -> List[RobotAction]:
        """Generate an action plan from a natural language command"""
        parsed_command = self.parser.parse_command(command)
        
        # Ground the command in the current world state
        grounded_command = self.ground_command(parsed_command)
        
        # Generate the action sequence
        action_plan = self.generate_action_sequence(grounded_command)
        
        return action_plan
    
    def ground_command(self, parsed_command: ParsedCommand) -> ParsedCommand:
        """Ground the parsed command in the current world state"""
        # Update object positions based on perception
        for obj in parsed_command.objects:
            # Query perception system for object location
            obj_info = self.perception.find_object(
                obj_type=obj['type'], 
                attributes=obj['attributes']
            )
            if obj_info:
                obj['position'] = obj_info['position']
                obj['id'] = obj_info['id']
        
        # Update location coordinates based on navigation map
        for loc in parsed_command.locations:
            # Query navigation system for location coordinates
            coords = self.navigation.get_location_coordinates(loc['name'])
            if coords:
                loc['coordinates'] = coords
        
        return parsed_command
    
    def generate_action_sequence(self, grounded_command: ParsedCommand) -> List[RobotAction]:
        """Generate an executable action sequence"""
        actions = []
        
        # Process each high-level action in sequence
        for i, action_type in enumerate(grounded_command.action_sequence):
            if action_type == 'navigate':
                # Plan navigation to the first specified location
                if grounded_command.locations:
                    target_location = grounded_command.locations[0]
                    nav_action = RobotAction(
                        name='navigate_to_region',
                        parameters={'destination': target_location['coordinates']},
                        preconditions=['robot_mobile'],
                        effects=['robot_at_destination']
                    )
                    actions.append(nav_action)
            
            elif action_type == 'find':
                # Plan to search for the first specified object
                if grounded_command.objects:
                    target_obj = grounded_command.objects[0]
                    find_action = RobotAction(
                        name='detect_object',
                        parameters={
                            'object_type': target_obj['name'],
                            'search_location': target_obj.get('position', [0, 0, 0])
                        },
                        preconditions=['camera_operational'],
                        effects=['object_location_known']
                    )
                    actions.append(find_action)
            
            elif action_type == 'grasp':
                # Plan to grasp the first specified object
                if grounded_command.objects:
                    target_obj = grounded_command.objects[0]
                    grasp_action = RobotAction(
                        name='grasp_object',
                        parameters={
                            'object_id': target_obj.get('id'),
                            'grasp_pose': self.calculate_grasp_pose(target_obj)
                        },
                        preconditions=['object_reachable', 'gripper_free'],
                        effects=['object_grasped', 'gripper_occupied']
                    )
                    actions.append(grasp_action)
            
            elif action_type == 'place':
                # Plan to place object at the first specified location
                if grounded_command.locations:
                    target_location = grounded_command.locations[0]
                    place_action = RobotAction(
                        name='place_object',
                        parameters={
                            'placement_position': target_location['coordinates'],
                            'release_type': 'careful' if 'carefully' in [c['value'] for c in grounded_command.constraints] else 'normal'
                        },
                        preconditions=['object_grasped'],
                        effects=['object_placed', 'gripper_free']
                    )
                    actions.append(place_action)
        
        # Add any necessary transport actions (grasp then navigate then place)
        if 'transport' in grounded_command.action_sequence and len(grounded_command.locations) >= 2:
            # Transport from first location to second location
            transport_actions = self.generate_transport_actions(grounded_command)
            actions.extend(transport_actions)
        
        # Add constraints and handling requirements
        self.apply_constraints(actions, grounded_command.constraints)
        
        return actions
    
    def generate_transport_actions(self, grounded_command: ParsedCommand) -> List[RobotAction]:
        """Generate actions for transport tasks (fetch and carry)"""
        transport_actions = []
        
        # First, navigate to object location (or find and go to it)
        if grounded_command.objects:
            obj = grounded_command.objects[0]
            if obj.get('position'):
                # Navigate to object
                transport_actions.append(RobotAction(
                    name='navigate_to_region',
                    parameters={'destination': obj['position']},
                    preconditions=['robot_mobile'],
                    effects=['robot_at_object']
                ))
            
            # Grasp the object
            transport_actions.append(RobotAction(
                name='grasp_object',
                parameters={
                    'object_id': obj.get('id', 'unknown'),
                    'grasp_pose': self.calculate_grasp_pose(obj)
                },
                preconditions=['object_reachable', 'gripper_free'],
                effects=['object_grasped', 'gripper_occupied']
            ))
        
        # Navigate to destination location
        if len(grounded_command.locations) >= 2:
            dest_location = grounded_command.locations[1]  # Assuming "bring X from Y to Z" format
            transport_actions.append(RobotAction(
                name='navigate_to_region',
                parameters={'destination': dest_location['coordinates']},
                preconditions=['robot_mobile', 'object_grasped'],
                effects=['robot_at_destination', 'object_transport_completed']
            ))
        
        # Place the object
        transport_actions.append(RobotAction(
            name='place_object',
            parameters={
                'placement_position': dest_location['coordinates'],
                'release_type': 'careful' if 'carefully' in [c['value'] for c in grounded_command.constraints] else 'normal'
            },
            preconditions=['object_grasped'],
            effects=['object_placed', 'gripper_free']
        ))
        
        return transport_actions
    
    def calculate_grasp_pose(self, obj: Dict[str, Any]) -> Dict[str, float]:
        """Calculate an appropriate grasp pose for an object"""
        # This would use perception information and grasp planning
        # For simplicity, returning a basic grasp pose
        return {
            'position': [obj['position'][0], obj['position'][1], obj['position'][2] + 0.1],  # Just above object
            'orientation': [0, 0, 0, 1]  # Default orientation
        }
    
    def apply_constraints(self, actions: List[RobotAction], constraints: List[Dict[str, Any]]):
        """Apply constraints to the action plan"""
        for constraint in constraints:
            if constraint['type'] == 'handling' and constraint['value'] == 'gentle':
                # Modify all grasp and place actions to be gentle
                for action in actions:
                    if action.name in ['grasp_object', 'place_object']:
                        action.parameters['force_limit'] = 'low'
                        action.duration *= 1.5  # Add extra time for careful handling
            elif constraint['type'] == 'speed' and constraint['value'] == 'fast':
                # Adjust navigation speed
                for action in actions:
                    if action.name == 'navigate_to_region':
                        action.parameters['max_speed'] = 'high'
```

**Step 2: Implement Neural Planning Component**

```python
import torch
import torch.nn as nn
import numpy as np

class NeuralActionPlanner(nn.Module):
    """Neural network for learning action sequences from language and perception"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Embedding layer for language input
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for processing language sequence
        self.language_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Perception processing
        self.perception_fc = nn.Sequential(
            nn.Linear(512, hidden_dim),  # Perception feature dimension
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combined processing
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layers for action selection
        self.action_selector = nn.Linear(hidden_dim, 20)  # 20 possible primitive actions
        self.navigation_planner = nn.Linear(hidden_dim, 2)  # x, y coordinates
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
    
    def forward(self, language_input, perception_input):
        """
        Forward pass to generate action plans from language and perception
        
        Args:
            language_input: [batch_size, seq_len] - Tokenized language input
            perception_input: [batch_size, feature_dim] - Perception features
        Returns:
            action_logits: [batch_size, num_actions] - Logits for possible actions
            navigation_target: [batch_size, 2] - Coordinates for navigation
        """
        # Process language input
        embedded_lang = self.word_embedding(language_input)
        lang_features, _ = self.language_lstm(embedded_lang)
        
        # Apply attention to focus on relevant parts
        lang_features = lang_features.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        attended_lang, _ = self.attention(lang_features, lang_features, lang_features)
        attended_lang = attended_lang.transpose(0, 1)  # [batch, seq_len, hidden_dim]
        
        # Take mean across sequence dimension
        lang_repr = attended_lang.mean(dim=1)  # [batch, hidden_dim]
        
        # Process perception input
        perc_repr = self.perception_fc(perception_input)
        
        # Combine language and perception
        combined_repr = torch.cat([lang_repr, perc_repr], dim=1)
        combined_features = self.combined_fc(combined_repr)
        
        # Generate action logits
        action_logits = self.action_selector(combined_features)
        navigation_target = self.navigation_planner(combined_features)
        
        return action_logits, navigation_target
    
    def plan_from_language_and_perception(self, language_tokens, perception_features):
        """
        Plan actions directly from language and perception inputs
        """
        self.eval()
        with torch.no_grad():
            action_logits, nav_target = self.forward(
                torch.tensor([language_tokens]), 
                torch.tensor([perception_features]).float()
            )
            
            # Convert to action probabilities
            action_probs = torch.softmax(action_logits[0], dim=0)
            
            # Select most probable action
            action_idx = torch.argmax(action_probs).item()
            
            return {
                'action_idx': action_idx,
                'action_prob': action_probs[action_idx].item(),
                'navigation_target': nav_target[0].tolist()
            }

class HybridActionPlanner:
    """Combines symbolic and neural planning approaches"""
    
    def __init__(self, neural_planner: NeuralActionPlanner):
        self.neural_planner = neural_planner
        self.symbolic_planner = ActionPlanner(None, None, None)  # Will be initialized later
        self.action_vocabulary = {
            0: 'move_forward',
            1: 'turn_left',
            2: 'turn_right',
            3: 'grasp',
            4: 'release',
            5: 'navigate_to',
            6: 'search_for',
            7: 'avoid_obstacle',
            # ... more actions
        }
    
    def plan_with_fallback(self, command: str, perception_features: np.ndarray) -> List[RobotAction]:
        """Plan using neural approach with symbolic fallback"""
        try:
            # Try neural planning first
            language_tokens = self.tokenize_command(command)
            neural_plan = self.neural_planner.plan_from_language_and_perception(
                language_tokens, perception_features
            )
            
            # Convert neural output to robot actions
            robot_actions = self.convert_neural_to_robot_action(neural_plan)
            
            # Validate the plan
            if self.validate_plan(robot_actions):
                return robot_actions
        except Exception:
            print("Neural planning failed, falling back to symbolic planning")
        
        # Fallback to symbolic planning
        return self.symbolic_planner.plan_from_command(command)
    
    def tokenize_command(self, command: str) -> List[int]:
        """Convert command string to token indices"""
        # This would use a proper tokenizer
        # For simplicity, returning placeholder tokens
        words = command.lower().split()
        # Map each word to a token ID (simplified)
        return [hash(word) % 1000 for word in words]  # Placeholder mapping
    
    def convert_neural_to_robot_action(self, neural_output: Dict) -> List[RobotAction]:
        """Convert neural planner output to robot actions"""
        action_idx = neural_output['action_idx']
        action_name = self.action_vocabulary.get(action_idx, 'unknown')
        
        if action_name == 'navigate_to':
            return [RobotAction(
                name='navigate_to_region',
                parameters={'destination': neural_output['navigation_target']},
                preconditions=['robot_mobile'],
                effects=['robot_at_destination']
            )]
        elif action_name == 'grasp':
            return [RobotAction(
                name='grasp_object',
                parameters={'object_id': 'target_object', 'grasp_pose': [0, 0, 0, 1]},
                preconditions=['object_reachable', 'gripper_free'],
                effects=['object_grasped', 'gripper_occupied']
            )]
        else:
            # Default action
            return [RobotAction(
                name=action_name,
                parameters={},
                preconditions=[],
                effects=[]
            )]
    
    def validate_plan(self, actions: List[RobotAction]) -> bool:
        """Validate if the plan is reasonable"""
        # Check if plan is not empty
        if not actions:
            return False
        
        # Check if actions have required parameters
        for action in actions:
            if not action.name:
                return False
        
        # Add more validation checks as needed
        return True
```

**Step 3: Implement Plan Execution and Monitoring**

```python
import asyncio
from enum import Enum

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PlanExecutor:
    """Executes action plans and monitors their progress"""
    
    def __init__(self, robot_interface):
        self.robot = robot_interface
        self.current_plan = []
        self.current_action_index = 0
        self.status = ExecutionStatus.PENDING
        self.execution_history = []
    
    def execute_plan(self, action_plan: List[RobotAction], timeout: float = 120.0) -> ExecutionStatus:
        """Execute a plan and return the final status"""
        self.current_plan = action_plan
        self.current_action_index = 0
        self.status = ExecutionStatus.RUNNING
        
        start_time = time.time()
        
        while (self.current_action_index < len(self.current_plan) and 
               self.status == ExecutionStatus.RUNNING and
               time.time() - start_time < timeout):
            
            action = self.current_plan[self.current_action_index]
            
            # Execute the action
            action_result = self.execute_action(action)
            
            if action_result == ExecutionStatus.SUCCESS:
                # Move to next action
                self.current_action_index += 1
            else:
                # Handle failure
                self.status = ExecutionStatus.FAILED
                self.execution_history.append({
                    'action': action,
                    'result': action_result,
                    'timestamp': time.time()
                })
                break
        
        # Update final status
        if self.status == ExecutionStatus.RUNNING:
            if self.current_action_index >= len(self.current_plan):
                self.status = ExecutionStatus.SUCCESS
            else:
                self.status = ExecutionStatus.CANCELLED
        
        return self.status
    
    def execute_action(self, action: RobotAction) -> ExecutionStatus:
        """Execute a single robot action"""
        print(f"Executing action: {action.name} with params: {action.parameters}")
        
        try:
            # Check preconditions
            if not self.check_preconditions(action.preconditions):
                print(f"Preconditions not met for action {action.name}")
                return ExecutionStatus.FAILED
            
            # Execute the specific action
            if action.name == 'navigate_to_region':
                result = self.robot.navigate_to_position(
                    action.parameters['destination']
                )
            elif action.name == 'grasp_object':
                result = self.robot.grasp_object(
                    action.parameters['object_id'],
                    action.parameters['grasp_pose']
                )
            elif action.name == 'place_object':
                result = self.robot.place_object(
                    action.parameters['placement_position'],
                    action.parameters.get('release_type', 'normal')
                )
            elif action.name == 'detect_object':
                result = self.robot.detect_object(
                    action.parameters['object_type'],
                    action.parameters['search_location']
                )
            else:
                # Unknown action
                result = False
            
            # Update effects if successful
            if result:
                self.apply_effects(action.effects)
                return ExecutionStatus.SUCCESS
            else:
                return ExecutionStatus.FAILED
                
        except Exception as e:
            print(f"Error executing action {action.name}: {e}")
            return ExecutionStatus.FAILED
    
    def check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if action preconditions are met"""
        # This would check the current robot state
        # For simulation, we'll assume preconditions are met
        return True
    
    def apply_effects(self, effects: List[str]):
        """Apply the effects of an action to the world state"""
        # This would update the world model
        # For now, just record the effects
        pass

class ReactivePlanner:
    """Handles reactive responses to unexpected situations during plan execution"""
    
    def __init__(self, base_planner: ActionPlanner, executor: PlanExecutor):
        self.base_planner = base_planner
        self.executor = executor
        self.reactive_rules = []
        self.setup_reactive_rules()
    
    def setup_reactive_rules(self):
        """Define reactive rules for handling exceptions"""
        self.reactive_rules = [
            {
                'condition': 'object_not_found',
                'action': 'search_in_alternative_location'
            },
            {
                'condition': 'path_obstructed', 
                'action': 'find_alternative_path'
            },
            {
                'condition': 'grasp_failed',
                'action': 'retry_with_different_grasp'
            },
            {
                'condition': 'user_intervention',
                'action': 'pause_and_wait_for_instruction'
            }
        ]
    
    def handle_exception(self, exception_type: str, context: Dict[str, Any]) -> Optional[List[RobotAction]]:
        """Handle an exception that occurs during plan execution"""
        for rule in self.reactive_rules:
            if rule['condition'] == exception_type:
                return self.execute_reactive_action(rule['action'], context)
        return None  # No reactive action found
    
    def execute_reactive_action(self, action_name: str, context: Dict[str, Any]) -> Optional[List[RobotAction]]:
        """Execute a reactive action"""
        if action_name == 'search_in_alternative_location':
            # Plan to search in a different location
            alternative_location = self.find_alternative_location(context.get('target_object'))
            if alternative_location:
                return [RobotAction(
                    name='navigate_to_region',
                    parameters={'destination': alternative_location},
                    preconditions=['robot_mobile'],
                    effects=['robot_at_search_location']
                )]
        
        elif action_name == 'find_alternative_path':
            # Plan to find a different route
            start_pos = context.get('start_position')
            goal_pos = context.get('goal_position')
            alternative_path = self.find_path_around_obstacle(start_pos, goal_pos)
            if alternative_path:
                return [RobotAction(
                    name='navigate_to_region',
                    parameters={'destination': alternative_path[-1]},
                    preconditions=['robot_mobile'],
                    effects=['robot_at_goal']
                )]
        
        elif action_name == 'retry_with_different_grasp':
            # Plan to try grasping differently
            return [RobotAction(
                name='grasp_object',
                parameters={
                    'object_id': context.get('object_id'),
                    'grasp_pose': self.calculate_alternative_grasp_pose(context.get('object_info'))
                },
                preconditions=['object_reachable', 'gripper_free'],
                effects=['object_grasped', 'gripper_occupied']
            )]
        
        return None  # Action could not be created
    
    def find_alternative_location(self, target_object: str) -> Optional[List[float]]:
        """Find an alternative location to search for an object"""
        # This would use spatial knowledge to find alternative locations
        # For simulation, returning a fixed alternative
        return [2.0, 2.0, 0.0]
    
    def find_path_around_obstacle(self, start: List[float], goal: List[float]) -> Optional[List[float]]:
        """Find a path around an obstacle"""
        # This would use path planning algorithms
        # For simulation, returning a simple detour
        return [start[0], start[1], 0.0]  # Same start position as fallback
    
    def calculate_alternative_grasp_pose(self, object_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate an alternative grasp pose"""
        # This would analyze object shape and calculate better grasp
        return {'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}
```

**Step 4: Integration with NVIDIA Isaac Platform**

```python
# NVIDIA Isaac specific action planning module
import omni
import carb
from omni.isaac.core import World
from omni.isaac.core.utils import stage, assets
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.sensors import Camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from pxr import Gf
import torch

class IsaacActionPlanner:
    def __init__(self):
        # Initialize Isaac environment
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.articulation_controller = None
        self.navigation_map = None
        
        # Set up the environment
        self.setup_isaac_environment()
    
    def setup_isaac_environment(self):
        """
        Set up the Isaac environment with robot and objects
        """
        # Add ground plane
        stage.add_ground_plane("/ground_plane", "XZ", 1000, [0, 0, 1], 0.5)
        
        # Add robot
        add_reference_to_stage(
            usd_path="path/to/humanoid_model.usd",
            prim_path="/World/HumanoidRobot"
        )
        
        # Add objects for interaction
        self.add_interactable_objects()
        
        # Initialize robot
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/HumanoidRobot",
                name="HumanoidRobot",
                usd_path="path/to/humanoid_model.usd"
            )
        )
        
        # Get articulation controller
        self.articulation_controller = ArticulationView(
            prim_paths_expr="/World/HumanoidRobot/.*",
            name="articulation_view",
            reset_xform_properties=False,
        )
        
        # Initialize world
        self.world.reset()
    
    def add_interactable_objects(self):
        """
        Add objects that the robot can interact with
        """
        # Add a red cube (mug alternative)
        DynamicCuboid(
            prim_path="/World/RedMug",
            name="RedMug",
            position=[1.5, 0.5, 0.1],
            size=0.1,
            color=torch.tensor([1.0, 0.0, 0.0])  # Red
        )
        
        # Add a table
        FixedCuboid(
            prim_path="/World/Table",
            name="Table",
            position=[2.5, 0.0, 0.4],
            size=0.8,
            color=torch.tensor([0.5, 0.3, 0.1])  # Brown
        )
    
    def plan_and_execute_command(self, command: str):
        """
        Plan and execute a command in Isaac simulation
        """
        print(f"Processing command: {command}")
        
        # Parse the command using our planning components
        # In a real implementation, this would connect to the full NLU pipeline
        command_parser = CommandParser()
        parsed_command = command_parser.parse_command(command)
        
        # Generate a plan based on the parsed command
        # This is simplified - in reality, we would connect to a full perception system
        plan = self.generate_simulated_plan(parsed_command)
        
        # Execute the plan in Isaac
        execution_result = self.execute_plan_in_simulation(plan)
        
        return execution_result
    
    def generate_simulated_plan(self, parsed_command: ParsedCommand) -> List[RobotAction]:
        """
        Generate a simple plan for demonstration purposes
        """
        actions = []
        
        # If the command involves navigation
        if any(action in parsed_command.action_sequence for action in ['navigate', 'go', 'move']):
            actions.append(RobotAction(
                name='navigate_to_position',
                parameters={'position': [2.0, 1.0, 0.0]},
                preconditions=['robot_mobile'],
                effects=['robot_at_destination']
            ))
        
        # If the command involves grasping
        if any(action in parsed_command.action_sequence for action in ['grasp', 'pick', 'take']):
            actions.append(RobotAction(
                name='grasp_object',
                parameters={'object_id': 'RedMug', 'position': [1.5, 0.5, 0.1]},
                preconditions=['object_reachable'],
                effects=['object_grasped']
            ))
        
        # If the command involves placing
        if any(action in parsed_command.action_sequence for action in ['place', 'put', 'set']):
            actions.append(RobotAction(
                name='place_object',
                parameters={'position': [2.5, 0.0, 0.5]},
                preconditions=['object_grasped'],
                effects=['object_placed']
            ))
        
        return actions
    
    def execute_plan_in_simulation(self, plan: List[RobotAction]):
        """
        Execute an action plan in the Isaac simulation
        """
        print("Starting plan execution in Isaac simulation...")
        
        for i, action in enumerate(plan):
            print(f"Executing action {i+1}/{len(plan)}: {action.name}")
            
            if action.name == 'navigate_to_position':
                self.execute_navigation_action(action)
            elif action.name == 'grasp_object':
                self.execute_grasp_action(action)
            elif action.name == 'place_object':
                self.execute_place_action(action)
            
            # Step the simulation
            for _ in range(100):  # Run simulation for a while to complete action
                self.world.step(render=True)
        
        print("Plan execution completed!")
        return {'status': 'completed', 'actions_executed': len(plan)}
    
    def execute_navigation_action(self, action: RobotAction):
        """
        Execute a navigation action in simulation
        """
        target_pos = action.parameters['position']
        print(f"Navigating to position: {target_pos}")
        
        # In a real implementation, this would call a navigation system
        # For simulation, we'll just move the robot
        self.articulation_controller.set_world_poses(
            positions=torch.tensor([target_pos]),
            orientations=torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # Default orientation
        )
    
    def execute_grasp_action(self, action: RobotAction):
        """
        Execute a grasping action in simulation
        """
        obj_id = action.parameters['object_id']
        obj_pos = action.parameters['position']
        print(f"Attempting to grasp object {obj_id} at position {obj_pos}")
        
        # In a real implementation, this would call a manipulation system
        # For simulation, we'll just log the action
        pass
    
    def execute_place_action(self, action: RobotAction):
        """
        Execute a placement action in simulation
        """
        pos = action.parameters['position']
        print(f"Placing object at position: {pos}")
        
        # In a real implementation, this would call a manipulation system
        # For simulation, we'll just log the action
        pass
    
    def run_command_demo(self):
        """
        Run a demonstration of command processing and execution
        """
        demo_commands = [
            "Go to the table",
            "Grasp the red mug",
            "Place the mug on the table",
            "Navigate to position (2, 1)"
        ]
        
        for command in demo_commands:
            print(f"\n--- Processing Command: '{command}' ---")
            result = self.plan_and_execute_command(command)
            print(f"Result: {result}\n")
    
    def cleanup(self):
        """
        Clean up Isaac resources
        """
        self.world.clear()

# Example usage for VLA system integration
class VLAPipelineWithActionPlanning:
    """Complete VLA pipeline with integrated action planning"""
    
    def __init__(self):
        self.nlu_system = None
        self.action_planner = None
        self.executor = None
    
    def setup_pipeline(self):
        """
        Set up the complete VLA pipeline with action planning
        """
        # Initialize components would happen here
        # In practice, this would connect to real perception, NLU, and action systems
        pass
    
    def process_language_command(self, command: str) -> Dict[str, Any]:
        """
        Process a complete language command through the VLA pipeline
        """
        # 1. Natural Language Understanding
        # nlu_result = self.nlu_system.process_utterance(command)
        
        # 2. Action Planning
        # action_plan = self.action_planner.plan_from_command(command)
        
        # 3. Execution
        # execution_result = self.executor.execute_plan(action_plan)
        
        # For simulation, returning a placeholder result
        return {
            'input_command': command,
            'nlu_result': 'parsed successfully',
            'action_plan': ['navigate', 'grasp', 'place'],
            'execution_result': {'status': 'completed', 'actions_executed': 3},
            'success': True
        }
    
    def run_vla_demo(self):
        """
        Run a complete VLA demonstration
        """
        print("Starting Vision-Language-Action demonstration...")
        
        commands = [
            "Bring me the red mug from the counter",
            "Go to the kitchen and find my keys",
            "Pick up the book and place it on the shelf"
        ]
        
        for command in commands:
            result = self.process_language_command(command)
            print(f"Command: {command}")
            print(f"Success: {result['success']}")
            print(f"Actions executed: {result['execution_result']['actions_executed']}")
            print("---")
        
        print("VLA demonstration completed!")

# Example usage
def run_action_planning_demo():
    """Run the action planning demonstration"""
    print("Setting up action planning system...")
    
    # Initialize components
    # command_parser = CommandParser()
    # action_planner = ActionPlanner(perception_system, nav_system, manip_system)
    # plan_executor = PlanExecutor(robot_interface)
    # reactive_planner = ReactivePlanner(action_planner, plan_executor)
    
    # Process a sample command
    # parsed = command_parser.parse_command("Bring me the red mug from the kitchen counter")
    # plan = action_planner.plan_from_command("Bring me the red mug from the kitchen counter")
    
    print("Action planning system ready!")
    
    # Run VLA pipeline demo
    # vla_pipeline = VLAPipelineWithActionPlanning()
    # vla_pipeline.run_vla_demo()

if __name__ == "__main__":
    run_action_planning_demo()
```

This comprehensive implementation provides a complete action planning system that translates language commands into executable robot actions, with integration into the Vision-Language-Action pipeline as required for User Story 4.

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Action Planning from Language                    │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Language      │    │   Planning      │    │   Execution     │ │
│  │   Understanding │───▶│   Component     │───▶│   Component     │ │
│  │                 │    │                 │    │                 │ │
│  │ • Command       │    │ • Plan          │    │ • Action        │ │
│  │   parsing       │    │   generation    │    │   execution     │ │
│  │ • Semantic      │    │ • Constraint    │    │ • Monitoring    │ │
│  │   grounding     │    │   validation    │    │ • Adaptation    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│         │                        │                       │         │
│         ▼                        ▼                       ▼         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   World Model   │    │   Action        │    │   Robot         │ │
│  │   (Environment) │    │   Sequences     │    │   Interface     │ │
│  │ • Object states │    │ • Primitive     │    │ • Navigation    │ │
│  │ • Locations     │    │ • Compound      │    │ • Manipulation  │ │
│  │ • Constraints   │    │ • Contingent    │    │ • Sensors       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Reactive Planning                            ││
│  │  • Exception handling    • Plan adaptation                     ││
│  │  • Uncertainty handling  • Contingency plans                   ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│                    Language to Action Pipeline                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Checklist

- [ ] Design action planning systems that translate language commands into primitive robot actions
- [ ] Implement symbolic planners that handle multi-step tasks and constraints
- [ ] Develop neural planning approaches for complex manipulation and navigation
- [ ] Integrate language understanding with action selection and execution
- [ ] Handle uncertainty and adapt planning during execution
- [ ] Evaluate the effectiveness of language-to-action translation systems
- [ ] Include voice-command processing examples
- [ ] Implement complete VLA pipeline examples
- [ ] Include diagrams showing perception → planning → action pipeline
- [ ] Implement examples of AI perception modules