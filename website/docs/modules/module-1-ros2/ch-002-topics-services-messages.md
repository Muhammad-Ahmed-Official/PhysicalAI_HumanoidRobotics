---
id: ch-002
title: "Chapter 1.2: Topics, Services, and Messages"
description: "Exploring the communication mechanisms in ROS 2: topics, services, and messages"
tags: [ros2, topics, services, messages]
---

# Chapter 1.2: Topics, Services, and Messages

## Introduction

In ROS 2, communication between nodes is facilitated through three primary mechanisms: topics, services, and actions. Understanding these communication patterns is crucial for developing humanoid robots, as they determine how different parts of the robot system exchange information. Topics enable asynchronous communication through publish-subscribe pattern, services provide synchronous request-response interaction, and actions extend services with feedback and goal preemption capabilities.

## Learning Outcomes

- Students will understand the concept of topics and the publish-subscribe communication pattern
- Learners will be able to implement services for synchronous request-response communication
- Readers will be familiar with ROS 2 message types and how to define custom messages
- Students will be able to distinguish when to use topics versus services versus actions

## Core Concepts

### Topics and Publish-Subscribe Pattern
Topics are named buses over which nodes exchange messages. In the publish-subscribe pattern:
- Publishers send messages to a topic without knowing who receives them
- Subscribers receive messages from a topic without knowing who sent them
- Multiple publishers and subscribers can exist for the same topic
- Communication is asynchronous and data is distributed by the DDS middleware

### Services and Request-Response Pattern
Services provide synchronous communication with request-response pattern:
- A client sends a request and waits for a response
- A service processes the request and sends back the response
- Communication is synchronous and blocking
- Only one service server can exist for a specific service name

### Actions
Actions are an extension of services that support long-running tasks:
- Allow clients to send goals to a server
- Provide feedback during goal execution
- Support goal preemption (canceling ongoing goals)
- Include status reporting throughout the process

### Messages
Messages are the data structures exchanged between nodes:
- Defined in .msg files with specific fields and types
- Automatically converted to language-specific structures (Python, C++, etc.)
- Can include basic types (int, float, string) and other message types
- Used by both topics and services with slightly different patterns

## Simulation Walkthrough

Let's look at practical examples of each communication pattern in ROS 2:

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from example_interfaces.srv import AddTwoInts


    class CommunicationDemoNode(Node):

        def __init__(self):
            super().__init__('communication_demo')
            
            # Topic: Publisher
            self.publisher_ = self.create_publisher(String, 'topic_demo', 10)
            
            # Topic: Subscriber
            self.subscription = self.create_subscription(
                String,
                'topic_demo',
                self.listener_callback,
                10)
            
            # Service: Server
            self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
            
            # Service: Client
            self.cli = self.create_client(AddTwoInts, 'add_two_ints')
            
            timer_period = 1  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            
            # Wait for service to be available
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service not available, waiting again...')

        def timer_callback(self):
            msg = String()
            msg.data = 'Hello from publisher: %d' % self.get_clock().now().nanoseconds
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)

        def listener_callback(self, msg):
            self.get_logger().info('I heard: "%s"' % msg.data)

        def add_two_ints_callback(self, request, response):
            response.sum = request.a + request.b
            self.get_logger().info('Incoming request\na: %d, b: %d' % (request.a, request.b))
            self.get_logger().info('Sending back response: [%d]' % response.sum)
            return response


    def main(args=None):
        rclpy.init(args=args)
        communication_demo = CommunicationDemoNode()
        
        # Example of client calling the service
        future = communication_demo.cli.call_async(AddTwoInts.Request(a=2, b=3))
        rclpy.spin_until_future_complete(communication_demo, future)
        result = future.result()
        communication_demo.get_logger().info(f'Result of service call: {result.sum}')
        
        rclpy.spin(communication_demo)
        
        communication_demo.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    // Initialize ROS 2 system
    Initialize ROS
    Create Node "communication_demo"
    
    // Create publisher for topic
    publisher = CreatePublisher(String, "topic_demo", queue_size=10)
    
    // Create subscriber for topic
    subscriber = CreateSubscriber(String, "topic_demo", callback_function)
    
    // Create service server
    service = CreateService("add_two_ints", add_two_ints_handler)
    
    // Create service client
    client = CreateClient("add_two_ints")
    
    // Timer to publish periodically
    timer_period = 1 second
    counter = 0
    
    While ROS is running:
        message = "Hello from publisher: " + timestamp
        Publish message on publisher
        Log "Publishing: " + message
        Sleep for timer_period
    End While
    
    // Example of client calling service
    result = CallService(client, a=2, b=3)
    Log "Result of service call: " + result
    ```
  </TabItem>
</Tabs>

This example demonstrates how to implement all three communication patterns in a single node. The publisher sends messages to a topic, the subscriber receives messages from the same topic, and the service handles requests to add two integers together.

## Visual Explanation

```
Communication Patterns in ROS 2

1. Topic (Publish-Subscribe):
   Publisher A ----> Topic ----> Subscriber B
                    (DDS)       Subscriber C
                               Multiple subscribers can receive same data

2. Service (Request-Response):
   Client A ----> Service Server
      |                 |
   Request         Process
      |                 |
   Response <---------| 
   Synchronous communication

3. Action (Extended Service):
   Client ----> Goal          Feedback ----> Client
                |                 |
                v                 ^
         Action Server -----> Running Process
                |                 |
                +------ Status ----+
   Long-running tasks with feedback and cancelation
```

The communication patterns serve different purposes: topics for streaming data, services for direct requests, and actions for complex, long-running operations with feedback.

## Checklist

- [x] Understand the publish-subscribe communication pattern
- [x] Know when to use services vs topics
- [x] Can implement a service server and client
- [x] Understand the purpose of actions in ROS 2
- [ ] Self-assessment: Can explain which communication pattern to use in different scenarios
- [ ] Self-assessment: Understand how DDS handles message distribution