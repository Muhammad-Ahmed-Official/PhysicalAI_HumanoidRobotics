import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Custom sidebar to organize the AI/Humanoid Robotics Book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'modules/chapter-template',
        'modules/content-schemas',
        'modules/id-system',
        'modules/accessibility'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      items: [
        'modules/module-1-ros2/index',
            "modules/module-1-ros2/ch-001",
            "modules/module-1-ros2/ch-002",
            "modules/module-1-ros2/ch-003",
            "modules/module-1-ros2/ch-004",
            "modules/module-1-ros2/ch-005",
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin - Simulation Environments',
      items: [
        'modules/module-2-digital-twin/index',
        'modules/module-2-digital-twin/chapter-2-1',
        'modules/module-2-digital-twin/chapter-2-2',
        'modules/module-2-digital-twin/chapter-2-3',
        'modules/module-2-digital-twin/chapter-2-4',
        'modules/module-2-digital-twin/chapter-2-5',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain - Perception and Planning',
      items: [
        'modules/module-3-ai-brain/index',
        'modules/module-3-ai-brain/chapter-3-1',
        'modules/module-3-ai-brain/chapter-3-2',
        'modules/module-3-ai-brain/chapter-3-3',
        'modules/module-3-ai-brain/chapter-3-4',
        'modules/module-3-ai-brain/chapter-3-5',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) for Humanoid Robots',
      items: [
        'modules/module-4-vla/index',
            "modules/module-4-vla/chapter-4-1",
            "modules/module-4-vla/chapter-4-2",
            "modules/module-4-vla/chapter-4-3",
            "modules/module-4-vla/chapter-4-4",
            "modules/module-4-vla/chapter-4-5",
            "modules/module-4-vla/capstone-project-integration",
      ],
      collapsed: false,
    },
  ],
};

export default sidebars;
