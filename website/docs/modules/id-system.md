---
id: id-system
title: Sequential ID System
description: Documentation of the sequential ID system for content entities
---

# Sequential ID System

This document defines the sequential ID system used throughout the AI/Humanoid Robotics educational content.

## ID Format

All content entities use the following format:

`{prefix}-{3-digit-number}`

Where:
- `{prefix}` is a short identifier for the entity type
- `{3-digit-number}` is a sequential number starting from 001

## ID Prefixes

| Entity Type | Prefix | Example |
|-------------|--------|---------|
| Book | `book-` | `book-001` |
| Module | `mod-` | `mod-001` |
| Chapter | `ch-` | `ch-001` |
| Section | `sect-` | `sect-001` |
| Code Snippet | `code-` | `code-001` |
| Diagram | `diag-` | `diag-001` |
| Learning Outcome | `lo-` | `lo-001` |

## Sequential Numbering

- Numbers start from 001 and increment sequentially
- Each entity type has its own sequential numbering
- Numbers are padded with leading zeros to ensure 3 digits (e.g., 001, 002, ..., 105)

## Usage Guidelines

1. When creating new content entities, assign the next available sequential ID
2. Maintain the same prefix for the same entity type
3. Do not reuse IDs for different entities
4. Document any ID changes or reassignments

## Example IDs

- `mod-001`, `mod-002`, `mod-003` - Sequential modules
- `ch-001`, `ch-002`, `ch-003` - Sequential chapters
- `sect-001`, `sect-002`, `sect-003` - Sequential sections within a chapter
- `code-001`, `code-002`, `code-003` - Sequential code snippets