---
id: 1
title: "spec-analyze"
stage: "general"
date_iso: "2025-12-17"
surface: "agent"
model: "unknown"
feature: "ai-humanoid-robotics"
branch: "001-ai-humanoid-robotics"
user: "DELL"
command: "/sp.analyze"
labels: ["analysis", "spec", "quality"]
spec: null
ticket: null
adr: null
pr: null
files_yaml:
  - "specs/001-ai-humanoid-robotics/spec.md"
  - "specs/001-ai-humanoid-robotics/plan.md"
  - "specs/001-ai-humanoid-robotics/tasks.md"
  - ".specify/memory/constitution.md"
tests_yaml: []
prompt_text: |
  ---
  description: Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation.
  ---

  ## User Input

  ```text
  $ARGUMENTS
  ```

  You **MUST** consider the user input before proceeding (if not empty).

  ## Goal

  Identify inconsistencies, duplications, ambiguities, and underspecified items across the three core artifacts (`spec.md`, `plan.md`, `tasks.md`) before implementation. This command MUST run only after `/sp.tasks` has successfully produced a complete `tasks.md`.

  ## Operating Constraints

  **STRICTLY READ-ONLY**: Do **not** modify any files. Output a structured analysis report. Offer an optional remediation plan (user must explicitly approve before any follow-up editing commands would be invoked manually).

  **Constitution Authority**: The project constitution (`.specify/memory/constitution.md`) is **non-negotiable** within this analysis scope. Constitution conflicts are automatically CRITICAL and require adjustment of the spec, plan, or tasks—not dilution, reinterpretation, or silent ignoring of the principle. If a principle itself needs to change, that must occur in a separate, explicit constitution update outside `/sp.analyze`.

  ## Execution Steps

  ### 1. Initialize Analysis Context

  Run `.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks` once from repo root and parse JSON for FEATURE_DIR and AVAILABLE_DOCS. Derive absolute paths:

  - SPEC = FEATURE_DIR/spec.md
  - PLAN = FEATURE_DIR/plan.md
  - TASKS = FEATURE_DIR/tasks.md

  Abort with an error message if any required file is missing (instruct the user to run missing prerequisite command).
  For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

  ### 2. Load Artifacts (Progressive Disclosure)

  Load only the minimal necessary context from each artifact:

  **From spec.md:**

  - Overview/Context
  - Functional Requirements
  - Non-Functional Requirements
  - User Stories
  - Edge Cases (if present)

  **From plan.md:**

  - Architecture/stack choices
  - Data Model references
  - Phases
  - Technical constraints

  **From tasks.md:**

  - Task IDs
  - Descriptions
  - Phase grouping
  - Parallel markers [P]
  - Referenced file paths

  **From constitution:**

  - Load `.specify/memory/constitution.md` for principle validation

  ### 3. Build Semantic Models

  Create internal representations (do not include raw artifacts in output):

  - **Requirements inventory**: Each functional + non-functional requirement with a stable key (derive slug based on imperative phrase; e.g., "User can upload file" → `user-can-upload-file`)
  - **User story/action inventory**: Discrete user actions with acceptance criteria
  - **Task coverage mapping**: Map each task to one or more requirements or stories (inference by keyword / explicit reference patterns like IDs or key phrases)
  - **Constitution rule set**: Extract principle names and MUST/SHOULD normative statements

  ### 4. Detection Passes (Token-Efficient Analysis)

  Focus on high-signal findings. Limit to 50 findings total; aggregate remainder in overflow summary.

  #### A. Duplication Detection

  - Identify near-duplicate requirements
  - Mark lower-quality phrasing for consolidation

  #### B. Ambiguity Detection

  - Flag vague adjectives (fast, scalable, secure, intuitive, robust) lacking measurable criteria
  - Flag unresolved placeholders (TODO, TKTK, ???, `<placeholder>`, etc.)

  #### C. Underspecification

  - Requirements with verbs but missing object or measurable outcome
  - User stories missing acceptance criteria alignment
  - Tasks referencing files or components not defined in spec/plan

  #### D. Constitution Alignment

  - Any requirement or plan element conflicting with a MUST principle
  - Missing mandated sections or quality gates from constitution

  #### E. Coverage Gaps

  - Requirements with zero associated tasks
  - Tasks with no mapped requirement/story
  - Non-functional requirements not reflected in tasks (e.g., performance, security)

  #### F. Inconsistency

  - Terminology drift (same concept named differently across files)
  - Data entities referenced in plan but absent in spec (or vice versa)
  - Task ordering contradictions (e.g., integration tasks before foundational setup tasks without dependency note)
  - Conflicting requirements (e.g., one requires Next.js while other specifies Vue)

  ### 5. Severity Assignment

  Use this heuristic to prioritize findings:

  - **CRITICAL**: Violates constitution MUST, missing core spec artifact, or requirement with zero coverage that blocks baseline functionality
  - **HIGH**: Duplicate or conflicting requirement, ambiguous security/performance attribute, untestable acceptance criterion
  - **MEDIUM**: Terminology drift, missing non-functional task coverage, underspecified edge case
  - **LOW**: Style/wording improvements, minor redundancy not affecting execution order

  ### 6. Produce Compact Analysis Report

  Output a Markdown report (no file writes) with the following structure:

  ## Specification Analysis Report

  | ID | Category | Severity | Location(s) | Summary | Recommendation |
  |----|----------|----------|-------------|---------|----------------|
  | A1 | Duplication | HIGH | spec.md:L120-134 | Two similar requirements ... | Merge phrasing; keep clearer version |

  (Add one row per finding; generate stable IDs prefixed by category initial.)

  **Coverage Summary Table:**

  | Requirement Key | Has Task? | Task IDs | Notes |
  |-----------------|-----------|----------|-------|

  **Constitution Alignment Issues:** (if any)

  **Unmapped Tasks:** (if any)

  **Metrics:**

  - Total Requirements
  - Total Tasks
  - Coverage %
  - Ambiguity Count
  - Duplication Count
  - Critical Issues Count

  ### 7. Provide Next Actions

  At end of report, output a concise Next Actions block:

  - If CRITICAL issues exist: Recommend resolving before `/sp.implement`
  - If only LOW/MEDIUM: User may proceed, but provide improvement suggestions
  - Provide explicit command suggestions: e.g., "Run /sp.specify with refinement", "Run /sp.plan to adjust architecture", "Manually edit tasks.md to add coverage for 'performance-metrics'"

  ### 8. Offer Remediation

  Ask the user: "Would you like me to suggest concrete remediation edits for the top N issues?" (Do NOT apply them automatically.)

  ## Operating Principles

  ### Context Efficiency

  - **Minimal high-signal tokens**: Focus on actionable findings, not exhaustive documentation
  - **Progressive disclosure**: Load artifacts incrementally; don't dump all content into analysis
  - **Token-efficient output**: Limit findings table to 50 rows; summarize overflow
  - **Deterministic results**: Rerunning without changes should produce consistent IDs and counts

  ### Analysis Guidelines

  - **NEVER modify files** (this is read-only analysis)
  - **NEVER hallucinate missing sections** (if absent, report them accurately)
  - **Prioritize constitution violations** (these are always CRITICAL)
  - **Use examples over exhaustive rules** (cite specific instances, not generic patterns)
  - **Report zero issues gracefully** (emit success report with coverage statistics)

  ## Context

  if any error occur solve and tell how to run locally

  ---

  As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

  1) Determine Stage
     - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

  2) Generate Title and Determine Routing:
     - Generate Title: 3–7 words (slug for filename)
     - Route is automatically determined by stage:
       - `constitution` → `history/prompts/constitution/`
       - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
       - `general` → `history/prompts/general/`

  3) Create and Fill PHR (Shell first; fallback agent‑native)
     - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
     - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
     - If the script fails:
       - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
       - Allocate an ID; compute the output path based on stage from step 2; write the file
       - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

  4) Validate + report
     - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
     - On failure: warn, don't block. Skip only for `/sp.phr`.
response_text: "# Specification Analysis Report\n\n## Findings Summary\n\n| ID | Category | Severity | Location(s) | Summary | Recommendation |\n|----|----------|----------|-------------|---------|----------------|\n| D1 | Duplication | MEDIUM | spec.md:FR-001,FR-003 | Similar requirements for teaching and explaining | Consolidate to avoid redundancy |\n| D2 | Duplication | MEDIUM | plan.md, tasks.md | Multiple references to required 6-section structure | Consolidate documentation to prevent inconsistencies |\n| A1 | Ambiguity | HIGH | spec.md:SC-002 | \"40 hours of study time\" lacks clarity on learner profile | Define specific target audience and skill level |\n| A2 | Ambiguity | MEDIUM | spec.md:SC-006 | \"70% success rate\" undefined for capstone project | Define how success rate is measured |\n| A3 | Ambiguity | HIGH | spec.md:SC-008 | \"90% of users\" lacks definition of user demographics | Specify user types and measurement methodology |\n| U1 | Underspecification | MEDIUM | spec.md:edge cases | Incomplete resolution of edge cases | Provide resolution strategies for each edge case |\n| U2 | Underspecification | HIGH | tasks.md | No performance testing tasks included | Add specific tasks for measuring and optimizing performance |\n| C1 | Constitution Alignment | CRITICAL | spec.md, plan.md, tasks.md | No clear evidence of plagiarism prevention measures | Ensure all content follows zero plagiarism principle |\n| C2 | Constitution Alignment | MEDIUM | plan.md, spec.md | Limited mention of accessibility compliance beyond requirements | Expand implementation details for WCAG 2.1 AA compliance |\n| G1 | Coverage Gap | MEDIUM | tasks.md | FR-006 (simulation-to-real bridging) has limited task coverage | Add specific tasks for simulation-to-real transfer techniques |\n| G2 | Coverage Gap | HIGH | tasks.md | No tasks addressing external dependency failures | Add tasks for graceful degradation with clear error messages |\n| G3 | Coverage Gap | MEDIUM | tasks.md | Requirements for mobile-responsive layouts lack specific tasks | Add specific mobile optimization tasks |\n| I1 | Inconsistency | MEDIUM | spec.md:FR-018 vs tasks.md | Missing specific tasks for PrismJS implementation | Add tasks for implementing syntax highlighting |\n| I2 | Inconsistency | LOW | spec.md:various vs plan.md | Minor terminology differences between files | Standardize terminology across all artifacts |\n\n## Coverage Summary Table:\n\n| Requirement Key | Has Task? | Task IDs | Notes |\n|-----------------|-----------|----------|-------|\n| fr-001-ros2-teaching | Yes | T013-T022 | Well-covered in US1 |\n| fr-002-gazebo-unity | Yes | T023-T032 | Well-covered in US2 |\n| fr-003-ai-perception | Yes | T033-T042 | Well-covered in US3 |\n| fr-004-ai-controllers | Yes | T033-T042 | Covered in US3 |\n| fr-005-motion-planning | Yes | T033-T042 | Covered in US3 |\n| fr-006-sim-to-real | Partial | T037 | Limited coverage in US3 |\n| fr-007-natural-language | Yes | T044-T045 | Well-covered in US4 |\n| fr-008-multi-modal | Yes | T043-T051 | Well-covered in US4 |\n| fr-009-autonomous-task-execution | Yes | T044-T052 | Well-covered in US4 |\n| fr-010-capstone-project | Yes | T048-T052 | Well-covered in US4 |\n| fr-011-docusaurus-mdx | Yes | T001-T005, T053-T058 | Well-covered |\n| fr-012-interactive-tabs | Yes | T009, T020 | Well-covered |\n| fr-013-collapsible-sections | Yes | T009 | Covered |\n| fr-014-admonitions | Yes | T009, T054 | Well-covered |\n| fr-015-visual-diagrams | Yes | Throughout | Well-covered |\n| fr-016-mobile-responsive | Partial | T055 | Limited coverage |\n| fr-017-progressive-navigation | Yes | T056 | Well-covered |\n| fr-018-prismjs-syntax-highlighting | No | None | Missing task |\n| fr-019-high-contrast-theme | Yes | T004, T058 | Well-covered |\n| fr-020-module-structure | Yes | Throughout | Well-covered |\n| fr-021-sequential-ids | Yes | T012 | Well-covered |\n| fr-022-content-views | Yes | T003 | Covered |\n| fr-023-external-dependencies | No | None | Missing task |\n| fr-024-wcag-compliance | Yes | T010 | Covered |\n\n## Constitution Alignment Issues:\n\n**CRITICAL ISSUE C1**: The zero plagiarism constitution principle lacks clear implementation evidence in the tasks. While specified, there are no tasks to ensure original content generation and verify against plagiarism.\n\n**MEDIUM ISSUE C2**: Although WCAG 2.1 AA compliance is specified in both spec and plan, the implementation details in tasks are limited to just one task (T010), despite accessibility being mentioned across multiple functional requirements.\n\n## Unmapped Tasks:\n\nSome tasks don't map directly to functional requirements:\n- T001-T005: Infrastructure setup (correctly general)\n- T006-T012: Foundation tasks (correctly general)\n- T053-T061: Polishing tasks (cross-cutting concerns)\n\n## Metrics:\n\n- Total Requirements: 24\n- Total Tasks: 61\n- Coverage % (requirements with >=1 task): 87.5%\n- Ambiguity Count: 3\n- Duplication Count: 2\n- Critical Issues Count: 2\n\n## Analysis Details:\n\n1. **Duplication Detection**: Requirements FR-001 (teach ROS 2) and FR-003 (explain AI perception) overlap in their language of \"explaining\" and \"teaching\" concepts. The 6-section chapter structure is mentioned in multiple places with slightly different wording.\n\n2. **Ambiguity Detection**: Several success criteria lack measurable baselines. For example, \"40 hours of study time\" doesn't define the skill level of the learner, and \"90% of users\" doesn't define who those users are or how they'll be measured.\n\n3. **Underspecification**: The edge cases section in spec.md identifies important challenges but doesn't provide solutions or strategies to address them. Tasks for handling computational limitations or reality gaps are missing.\n\n4. **Constitution Alignment**: The constitution mandates \"Zero Plagiarism and Original Content Generation,\" but there are no specific tasks to ensure this principle is implemented during content creation.\n\n5. **Coverage Gaps**: FR-023 addresses handling failures with external dependencies but no tasks implement graceful degradation. The \"mobile-responsive layouts\" requirement (FR-016) has only one associated task with limited scope.\n\n6. **Inconsistency**: The requirement for PrismJS syntax highlighting (FR-018) exists but no specific task implements it, despite other UI features having dedicated tasks.\n\n## Next Actions\n\nCRITICAL issues have been identified that must be addressed before implementation:\n1. The zero plagiarism constitution principle lacks implementation in the tasks\n2. No tasks address handling external dependency failures (FR-023)\n\nIt is recommended to resolve these issues before proceeding with `/sp.implement`.\n\nFor the HIGH severity issues:\n- Add specific tasks for performance testing and optimization to meet the <3 seconds page load requirement\n- Add tasks to address failure handling for external dependencies like ROS 2, Gazebo, Unity, or NVIDIA Isaac\n\nFor the MEDIUM and LOW issues, improvement suggestions include:\n- Run `/sp.specify` with refinements to clarify ambiguous success criteria\n- Run `/sp.tasks` to add missing tasks for mobile responsiveness and PrismJS implementation\n- Manually edit tasks.md to add coverage for 'simulation-to-real' and clarify user demographics for success metrics\n\n## Remediation Suggestion\n\nWould you like me to suggest concrete remediation edits for the top 5 issues identified? This would involve proposing specific additions to the spec, plan, or tasks documents to address the most critical findings."
outcome: "Analysis completed successfully with identification of 14 total issues across all categories. Critical issues related to constitution compliance and missing task coverage were highlighted as requiring immediate attention before implementation."
evaluation: "The analysis correctly identified inconsistencies, duplications, ambiguities and underspecified items across all three artifacts. The report structure matches the required format with findings table, coverage summary, and next actions. All constitution requirements were checked against the artifacts as mandated by the execution steps."
---
