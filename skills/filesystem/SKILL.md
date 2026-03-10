---
name: filesystem
description: File and codebase navigation skill for listing directories, reading files, writing files, and searching code.
tools:
  - workspace_info
  - list_directory
  - read_file
  - write_file
  - edit_file
  - search_files
  - grep_text
triggers:
  - file
  - files
  - folder
  - directory
  - repo
  - project
  - codebase
  - read
  - write
  - edit
  - search
  - grep
  - find
always_on: false
---

# Filesystem Skill

Preferred workflow for code and file tasks:
1. `list_directory` to orient in the repo
2. `grep_text` for literal text search, or `search_files` for regex search
3. `read_file` for exact content
4. `edit_file` for surgical edits (replace old_text with new_text; saves tokens vs full rewrite)
5. `write_file` to create or overwrite entire files

Never fabricate file content; rely on tool outputs.
