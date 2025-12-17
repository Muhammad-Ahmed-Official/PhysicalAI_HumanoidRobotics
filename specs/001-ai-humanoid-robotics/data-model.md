# Data Model: AI/Humanoid Robotics Book

**Feature**: 001-ai-humanoid-robotics  
**Date**: 2025-12-16  
**Input**: Feature specification from `/specs/001-ai-humanoid-robotics/spec.md`

## Overview

This document defines the data structures and relationships for the educational content of the Physical AI & Humanoid Robotics book. The "data model" represents the organizational structure of the learning materials rather than application data.

## Content Entities

### Book
- **Fields**:
  - id: string (sequential ID with prefix "book-")
  - title: string
  - description: string
  - modules: [Module]
  - learning_outcomes: [string]
  - target_audience: string
  - prerequisites: [string]

### Module
- **Fields**:
  - id: string (sequential ID with prefix "mod-")
  - title: string
  - description: string
  - chapters: [Chapter]
  - learning_outcomes: [string]
  - duration_estimate: string
- **Relationships**:
  - Belongs to: Book
  - Contains: Chapters

### Chapter
- **Fields**:
  - id: string (sequential ID with prefix "ch-")
  - title: string
  - description: string
  - sections: [Section]
  - learning_outcomes: [string]
  - prerequisites: [string]
  - duration_estimate: string
- **Relationships**:
  - Belongs to: Module
  - Contains: Sections

### Section
- **Fields**:
  - id: string (sequential ID with prefix "sect-")
  - title: string
  - content: MDXContent
  - section_type: SectionType (enum: "introduction", "learning_outcomes", "core_concepts", "simulation_walkthrough", "visual_explanation", "checklist")
  - order: integer
- **Relationships**:
  - Belongs to: Chapter

### MDXContent
- **Fields**:
  - raw_content: string (MDX syntax)
  - compiled_html: string (rendered HTML)
  - embedded_components: [Component]
  - code_snippets: [CodeSnippet]

### CodeSnippet
- **Fields**:
  - id: string (sequential ID with prefix "code-")
  - language: string
  - code: string
  - description: string
  - is_pseudocode: boolean
  - simulation_relevance: string

### Component
- **Fields**:
  - component_type: ComponentType (enum: "tabs", "admonition", "diagram", "callout")
  - props: JSON
  - content: [MDXContent]

### Diagram
- **Fields**:
  - id: string (sequential ID with prefix "diag-")
  - title: string
  - description: string
  - image_path: string
  - alt_text: string
  - caption: string

### LearningOutcome
- **Fields**:
  - id: string (sequential ID with prefix "lo-")
  - description: string
  - measurable: boolean
  - success_criteria: string
  - difficulty_level: Level (enum: "beginner", "intermediate", "advanced")
- **Relationships**:
  - Associated with: Book, Module, Chapter

## Validations and Constraints

1. **Sequential ID Format**: All IDs must follow the format `{prefix}-{3-digit-number}` (e.g., "mod-001", "ch-015")
2. **Chapter Structure**: Each chapter must contain exactly 6 sections following the required structure:
   - introduction (order: 0)
   - learning_outcomes (order: 1)
   - core_concepts (order: 2)
   - simulation_walkthrough (order: 3)
   - visual_explanation (order: 4)
   - checklist (order: 5)
3. **Content Requirements**: All content must be reproducible in simulation environments without requiring physical hardware
4. **Accessibility Compliance**: All content must maintain WCAG 2.1 AA compliance
5. **Module Dependencies**: Later modules can depend on concepts from earlier modules but should remain as independent as possible
6. **Duration Limits**: Individual chapters should not exceed 2 hours of learning time

## State Transitions

### ContentState
- **Draft**: Initial creation state, content not yet validated
- **Reviewed**: Content reviewed by technical expert but not yet tested
- **Tested**: Content tested in simulation environment
- **Published**: Final content ready for deployment

### ContentLifecycle
```
Draft → Reviewed → Tested → Published
```

## Relationships Summary

- Book contains many Modules
- Module contains many Chapters
- Chapter contains many Sections
- Section contains many CodeSnippets and Components
- Diagrams are embedded within Sections via Component references
- LearningOutcomes are associated with Book, Module, and Chapter entities

## Content Organization Schema

```
Book
├── Module (4 total)
│   ├── Module Metadata
│   └── Chapter (5 per Module, 20 total)
│       ├── Chapter Metadata
│       └── Section (6 per Chapter, 120 total)
│           ├── Introduction
│           ├── Learning Outcomes
│           ├── Core Concepts
│           ├── Simulation Walkthrough
│           ├── Visual Explanation
│           └── Checklist
```

This structure ensures consistent learning progression and predictable navigation for readers while maintaining the required educational standards.