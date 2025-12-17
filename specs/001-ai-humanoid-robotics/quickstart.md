# Quickstart Guide: AI/Humanoid Robotics Book

**Feature**: 001-ai-humanoid-robotics  
**Date**: 2025-12-16  
**Purpose**: Get up and running with the Physical AI & Humanoid Robotics educational book project

## Prerequisites

Before starting, ensure you have the following installed:

- **Node.js** (v18 or higher)
- **npm** or **yarn** package manager
- **Git** for version control
- Basic familiarity with **Markdown** and **MDX**
- Access to **Python 3.8+** (for ROS 2 examples, though pseudocode will be used primarily)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Docusaurus Dependencies

```bash
npm install
# OR if using yarn
yarn install
```

### 3. Start the Development Server

```bash
npm run start
# OR if using yarn
yarn start
```

This will start the development server at `http://localhost:3000` with hot reloading.

## Project Structure

The educational content is organized as follows:

```
docs/
├── modules/                    # Content for each module
│   ├── module-1-ros2/          # Module 1: ROS 2 fundamentals
│   ├── module-2-digital-twin/  # Module 2: Simulation environments
│   ├── module-3-ai-brain/      # Module 3: AI perception and planning
│   └── module-4-vla/           # Module 4: Vision-Language-Action
├── components/                 # Custom Docusaurus components
├── static/                     # Static assets (images, diagrams)
│   └── images/                 # Visual diagrams and other images
└── docusaurus.config.js        # Docusaurus configuration
```

## Creating New Content

### Chapter Template

Each chapter follows a consistent structure:

```mdx
---
id: ch-xxx
title: "Chapter Title"
description: "Brief description of the chapter"
tags: [tag1, tag2]
---

# Chapter Title

## Introduction

Brief overview of the chapter's importance and what will be covered.

## Learning Outcomes

- Students will understand ...
- Learners will be able to ...
- Readers will be familiar with ...

## Core Concepts

Detailed explanation of the key concepts with supporting visuals and examples.

## Simulation Walkthrough

Practical implementation using pseudocode and simulation examples:

```python
# Python example
def example_function():
    pass
```

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    # Python implementation
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    # Pseudocode representation
    ```
  </TabItem>
</Tabs>

## Visual Explanation

Diagrams, flowcharts, and visual aids to reinforce concepts.

<Admonition type="note" title="Key Concept">
  Important concept explanation.
</Admonition>

## Checklist

- [ ] Key concept understood
- [ ] Example implemented
- [ ] Self-assessment completed

```

### Available Components

#### Tabs for Multiple Examples
```jsx
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    # Python code example
    ```
  </TabItem>
  <TabItem value="pseudocode" label="Pseudocode">
    ```
    # Pseudocode representation
    ```
  </TabItem>
</Tabs>
```

#### Admonitions for Special Notes
```jsx
import Admonition from '@theme/Admonition';

<Admonition type="note" title="Key Concept">
  Explanation of an important concept.
</Admonition>

<Admonition type="tip" title="Pro Tip">
  Helpful suggestion for the reader.
</Admonition>

<Admonition type="caution" title="Warning">
  Important cautionary information.
</Admonition>
```

## Content Guidelines

### Writing Style
- Write for beginner-friendly understanding while maintaining technical accuracy
- Use official documentation sources only
- Ensure all examples can be reproduced in simulation environments
- Follow consistent structure: Summary → Objectives → Explanation → Example → Checklist

### Code Examples
- Use Python for ROS 2 examples with `rclpy`
- Include pseudocode alternatives for complex implementations
- Ensure all examples are testable in simulation
- Add appropriate comments and explanations

### Visual Elements
- Include diagrams for complex concepts
- Use accessible alt text for all images
- Ensure high contrast for readability
- Maintain consistent styling

## Building the Book

To build the static site for deployment:

```bash
npm run build
# OR if using yarn
yarn build
```

The build output will be in the `build/` directory.

## Running Tests

Validate the book content:

```bash
npm run serve
# This serves the built site locally to verify all links and content work properly
```

## Deployment

The book can be deployed to GitHub Pages or other static hosting services. The default configuration is set up for GitHub Pages deployment.

1. Build the site: `npm run build`
2. The output in the `build/` directory can be deployed to any static hosting service
3. For GitHub Pages, push to the `gh-pages` branch or use GitHub Actions

## Troubleshooting

### Common Issues

1. **Content not updating**: Clear Docusaurus cache with `npm run clear`
2. **Build errors**: Check for syntax errors in Markdown/MDX files
3. **Missing images**: Ensure images are in the `static/images/` directory and use proper paths

### Getting Help

- Check the feature specification in `specs/001-ai-humanoid-robotics/spec.md`
- Review the implementation plan in `specs/001-ai-humanoid-robotics/plan.md`
- Consult the data model in `specs/001-ai-humanoid-robotics/data-model.md`