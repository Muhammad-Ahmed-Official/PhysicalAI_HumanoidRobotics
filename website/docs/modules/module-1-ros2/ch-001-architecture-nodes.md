---
id: ch-001
title: "Chapter 1.1: ROS 2 Architecture and Nodes"
description: "Understanding the fundamental architecture of ROS 2 and the concept of nodes"
tags: [ros2, architecture, nodes]
---

# Chapter 1.1: ROS 2 Architecture and Nodes

## Introduction

The Robot Operating System 2 (ROS 2) is an open-source framework designed to help developers build robotic applications. Understanding its architecture and the concept of nodes is fundamental to developing humanoid robots. This chapter introduces the core architecture of ROS 2, focusing on the node-based approach that allows for distributed robotics software development.

## Learning Outcomes

- Students will understand the fundamental architecture of ROS 2
- Learners will be able to explain what a ROS 2 node is and its purpose
- Readers will be familiar with the communication patterns in ROS 2 architecture
- Students will be able to create a simple ROS 2 node in Python

## Core Concepts

ROS 2 follows a distributed architecture where computation is spread across multiple processes and potentially multiple devices. The basic building blocks of this architecture are:

1. **Nodes**: Processes that perform computation. Nodes are the fundamental units of a ROS 2 program.
2. **Topics**: Named buses over which nodes exchange messages.
3. **Services**: Synchronous request/response communication between nodes.
4. **Actions**: Extended services that include feedback and goal preemption.

The architecture uses a Data Distribution Service (DDS) as the underlying middleware to enable communication between nodes across different devices and networks. This provides features like discovery, fault tolerance, and real-time performance.

A key improvement of ROS 2 over ROS 1 is the use of DDS, which makes ROS 2 more suitable for production environments, especially in applications like humanoid robotics where reliability and real-time performance are crucial.

## Simulation Walkthrough

Let's walk through creating a simple ROS 2 node that publishes messages. This example will help illustrate the node concept in practice:

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String


    class MinimalPublisher(Node):

        def __init__(self):
            super().__init__('minimal_publisher')
            self.publisher_ = self.create_publisher(String, 'topic', 10)
            timer_period = 0.5  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            self.i = 0

        def timer_callback(self):
            msg = String()
            msg.data = 'Hello World: %d' % self.i
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            self.i += 1


    def main(args=None):
        rclpy.init(args=args)

        minimal_publisher = MinimalPublisher()

        rclpy.spin(minimal_publisher)

        # Destroy the node explicitly
        minimal_publisher.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    // Initialize ROS 2 system
    Initialize ROS
    Create Node "minimal_publisher"
    
    // Create publisher
    publisher = CreatePublisher(String, "topic", queue_size=10)
    
    // Create timer to periodically publish messages
    timer_period = 0.5 seconds
    counter = 0
    
    While ROS is running:
        message = "Hello World: " + counter
        Publish message on publisher
        Log "Publishing: " + message
        counter = counter + 1
        Sleep for timer_period
    End While
    
    Cleanup resources
    Shutdown ROS
    ```
  </TabItem>
</Tabs>

This code creates a ROS 2 node that publishes "Hello World" messages to a topic named "topic". The node demonstrates the basic structure of a ROS 2 node which includes initialization, creating publishers/subscribers, and spinning to process callbacks.

## Visual Explanation

```
[ROS 2 Architecture Diagram - Nodes and Communication]

Node A (Publisher)        DDS Middleware Layer        Node B (Subscriber)
     |------------------->|------------------->|
     |  Topic: /topic     |  Message Transport |
     |  Message: "Hello"  |                    |
     |<-------------------|<-------------------|
     |  Service Request   |  Service Response  |

Each node runs independently and communicates through the DDS layer, 
which handles message routing, discovery, and delivery.
```

The diagram illustrates how nodes communicate through ROS 2's middleware layer. Nodes do not directly communicate with each other; instead, messages flow through the DDS middleware which handles routing, discovery, and delivery.

## Checklist

- [x] Understand the concept of a ROS 2 node
- [x] Know the different communication patterns in ROS 2
- [x] Can create a basic publisher node
- [x] Understand how the DDS middleware enables communication
- [ ] Self-assessment: Can explain why nodes are important for distributed robotics
- [ ] Self-assessment: Understand the role of DDS in ROS 2