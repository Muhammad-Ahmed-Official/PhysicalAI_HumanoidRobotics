--- 
id: 1
title: Fix Docusaurus Build Errors
stage: red
date: 2025-12-17
surface: agent
model: claude-3-opus
feature: docusaurus-fixes
branch: users/example-user/docusaurus-fixes
user: Example User
command: solve the error [INFO] Starting the development server...
labels: ["docusaurus", "bug-fix", "configuration"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - website/sidebars.ts
  - website/docusaurus.config.ts
tests_yaml: []
prompt_text: |- 
  solve the error [INFO] Starting the development server...
  [WARNING] The `siteConfig.onBrokenMarkdownLinks` config option is deprecated and will be removed in Docusaurus v4.
  Please migrate and move this option to `siteConfig.markdown.hooks.onBrokenMarkdownLinks` instead.
  [ERROR] Loading of version failed for version current

  [ERROR] Error: Invalid sidebar file at "sidebars.ts".
  These sidebar document ids do not exist:
  - modules/module-1-ros2/ch-001-architecture-nodes
  - modules/module-1-ros2/ch-002-topics-services-messages
  - modules/module-1-ros2/ch-003-middleware-design-patterns
  - modules/module-1-ros2/ch-004-sensor-actuator-simulation
  - modules/module-1-ros2/ch-005-debugging-logging-tools
  - modules/module-4-vla/chapter-4-2
  - modules/module-4-vla/chapter-4-3
  - modules/module-4-vla/chapter-4-4
  - modules/module-4-vla/chapter-4-5

  Available document ids are:
  - intro
  - modules/accessibility
  - modules/chapter-template
  - modules/content-schemas
  - modules/id-system
  - modules/module-1-ros2/ch-001
  - modules/module-1-ros2/ch-002
  - modules/module-1-ros2/ch-003
  - modules/module-1-ros2/ch-004
  - modules/module-1-ros2/ch-005
  - modules/module-1-ros2/index
  - modules/module-2-digital-twin/chapter-2-1
  - modules/module-2-digital-twin/chapter-2-2
  - modules/module-2-digital-twin/chapter-2-3
  - modules/module-2-digital-twin/chapter-2-4
  - modules/module-2-digital-twin/chapter-2-5
  - modules/module-2-digital-twin/index
  - modules/module-3-ai-brain/chapter-3-1
  - modules/module-3-ai-brain/chapter-3-2
  - modules/module-3-ai-brain/chapter-3-3
  - modules/module-3-ai-brain/chapter-3-4
  - modules/module-3-ai-brain/chapter-3-5
  - modules/module-3-ai-brain/index
  - modules/module-4-vla/chapter-4-1
  - modules/module-4-vla/index
  - tutorial-basics/congratulations
  - tutorial-basics/create-a-blog-post
  - tutorial-basics/create-a-document
  - tutorial-basics/create-a-page
  - tutorial-basics/deploy-your-site
  - tutorial-basics/markdown-features
  - tutorial-extras/manage-docs-versions
  - tutorial-extras/translate-your-site

      at Object.checkSidebarsDocIds (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\plugin-content-docs\lib\sidebars\utils.js:245:19)
      at doLoadVersion (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:102:19)
      at async loadVersion (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\plugin-content-docs\lib\versions\loadVersion.js:119:16)
      at async Promise.all (index 0)
      at async Object.loadContent (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\plugin-content-docs\lib\index.js:152:33)
      at async C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\core\lib\server\plugins\plugins.js:40:23
      at async loadSite (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\core\lib\server\site.js:155:45)
      at async createReloadableSite (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\core\lib\commands\start\utils.js:62:16)
      at async doStart (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\core\lib\commands\start\start.js:22:28)
      at async Promise.all (index 0)
      at async runCLI (C:\Users\DELL\Documents\google\website\node_modules\@docusaurus\core\lib\commands\cli.js:56:5)
      at async file:///C:/Users/DELL/Documents/google/website/node_modules/@docusaurus/core/bin/docusaurus.mjs:44:3
  [INFO] Docusaurus version: 3.9.2
  Node version: v24.11.1
  PS C:\Users\DELL\Documents\google\website> 
response_text: |-
  Fixed invalid document IDs in `website/sidebars.ts` and addressed the deprecated `onBrokenMarkdownLinks` configuration in `website/docusaurus.config.ts`.
outcome: All Docusaurus build errors related to sidebar links and deprecated configuration should now be resolved, allowing the development server to start correctly.
evaluation: Successfully identified and corrected all reported Docusaurus configuration errors.
---