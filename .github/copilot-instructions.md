# GitHub Copilot Instructions for Tracker Repository

## Task Management
Proactively use the task-manager MCP server to manage work. When the user describes work to do, automatically create tasks. When starting a session, check getNextTask. When work is completed, mark tasks as done with setTaskStatus. Break down complex requests into subtasks using expandTask. Always track progress through the task system without being asked.

**Important**: The task database is stored at `data/taskmanager.db`. When completing work, ensure any task status changes are committed to the repository so they sync back to the local development environment.

## Sequential Thinking
For complex problems, multi-step tasks, or when planning is needed, use the sequential-thinking MCP server. Break down problems into logical steps, get tool recommendations, and follow structured reasoning. Use it automatically when facing architectural decisions, debugging complex issues, or implementing multi-part features.

## Documentation Lookup
Use the Context7 MCP server to fetch up-to-date documentation for any library or framework. Before implementing features with external libraries, always resolve the library ID with `resolve-library-id` and then fetch relevant docs with `get-library-docs`. This ensures code follows current best practices and API usage patterns.

## Project Context
This is a vehicle tracking system using:
- Python with OpenCV and YOLOv8 for object detection
- Video processing for truck/vehicle identification
- Debug image storage organized by vehicle ID
