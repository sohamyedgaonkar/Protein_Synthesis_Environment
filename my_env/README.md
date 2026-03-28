---
title: Code Reviewer Environment Server
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - code-review
  - reinforcement-learning
---

# Code Reviewer Environment

A reinforcement learning environment for training AI agents to review and improve Python code from GitHub repositories. Agents learn to fix bugs, remove unused imports, and optimize code quality while receiving reward signals for each improvement.

## Quick Start

The simplest way to use the Code Reviewer environment is through the `CodeReviewerEnv` class:

```python
from my_env import CodeReviewerAction, CodeReviewerEnv

try:
    # Create environment from Docker image
    env = CodeReviewerEnv.from_docker_image("code-reviewer-env:latest")

    # Reset with a GitHub repository
    repo_url = "https://github.com/user/python-project"
    result = env.reset(repo_url=repo_url)
    
    print(f"Repository: {result.observation.repo_url}")
    print(f"Initial Errors: {len(result.observation.errors)}")
    print(f"Code Files: {result.observation.code_metrics.total_files}")
    print(f"Code Lines: {result.observation.code_metrics.total_lines}")

    # Propose code improvements
    action = CodeReviewerAction(
        file_path="main.py",
        modified_code="improved code here...",
        description="Fixed bug in line 42"
    )

    result = env.step(action)
    print(f"\nAfter Fix:")
    print(f"Reward: {result.reward}")
    print(f"Errors Remaining: {len(result.observation.errors)}")
    print(f"Done: {result.done}")

finally:
    # Always clean up
    env.close()
```

That's it! The `CodeReviewerEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t code-reviewer-env:latest -f server/Dockerfile .
```

## Key Features

### Environment Functions

**reset(repo_url: str) → CodeReviewerObservation**
- Clones a GitHub Python repository
- Collects all source code
- Analyzes code quality (imports, syntax, etc.)
- Executes main.py and captures output/errors
- Returns full observation with code and initial errors

**step(action) → CodeReviewerObservation**

Supports **two types of actions**:

1. **CodeReviewerAction** - Full file replacement
   - Provide entire modified file content
   - Simpler but less efficient
   - Use when replacing entire file

2. **LineEditAction** ⭐ NEW - Granular line-by-line edits
   - Make targeted changes without full file content
   - Support operations: replace, add, delete
   - More efficient - agent only specifies changed lines
   - Multiple edits in one action
   - Perfect for fixing specific errors

Example with CodeReviewerAction:
```python
action = CodeReviewerAction(
    file_path="main.py",
    modified_code="entire file content...",
    description="Fixed bug"
)
```

Example with LineEditAction:
```python
action = LineEditAction(
    edits=[
        LineEdit(file_path="main.py", line_number=51, 
                operation="replace", new_code="def add_numbers(a, b):\n"),
        LineEdit(file_path="main.py", line_number=73, 
                operation="replace", new_code="def thread_task():\n"),
    ],
    description="Fixed 2 syntax errors"
)
```

- Applies proposed code changes
- Re-analyzes code metrics
- Re-executes main.py
- Calculates reward based on improvements
- Returns observation with new results and reward

**state → State**
- Returns current episode ID and step count

### Observation Structure

The observation includes:
- `repo_url` - URL of the repository
- `all_code` - Complete code from all Python files
- `execution_logs` - Output from running main.py
- `errors` - List of errors with type, message, line number
- `code_metrics` - Code analysis (unused imports, syntax errors, etc.)
- `reward` - Reward signal for this step
- `done` - Episode completion flag
- `metadata` - Additional tracking information

### Reward System

Agents receive rewards for improvements:
| Achievement | Reward |
|-------------|--------|
| Fix an error | +0.5 |
| Remove unused import | +0.1 |
| Reduce warning count | +0.1 per warning |
| Successful execution | +0.2 |
| Introduce syntax error | -0.2 |

### Code Analysis Features

The environment automatically:
- ✅ Detects unused imports using AST parsing
- ✅ Counts syntax and runtime errors
- ✅ Parses exception tracebacks
- ✅ Extracts line numbers from errors
- ✅ Tracks code quality metrics
- ✅ Manages repository versions

## How It Works

```
1. Agent initializes with GitHub repo URL
   └─ reset("https://github.com/user/project")
   
2. Environment returns initial code state
   └─ all_code, errors, metrics, execution_logs
   
3. Agent analyzes errors and proposes fix
   └─ Option A: CodeReviewerAction(file_path, modified_code)
   └─ Option B: LineEditAction(edits=[LineEdit(...), ...])  ⭐ NEW
   
4. Environment applies changes and tests
   └─ Runs main.py, analyzes metrics, calculates reward
   
5. Agent receives feedback
   └─ reward signal, new errors, updated metrics
   
6. Repeat until episode complete
   └─ done=True when all errors/warnings fixed
```

## Environment Details

### Action - CodeReviewerAction
```python
{
    "file_path": "main.py",           # File to modify (relative path)
    "modified_code": "import sys\n...",  # Complete new file content
    "description": "Fixed bug"        # Change description
}
```

### Action - LineEditAction ⭐ NEW
```python
{
    "edits": [
        {
            "file_path": "main.py",
            "line_number": 51,                    # 1-indexed line number
            "operation": "replace",               # replace | add | delete
            "new_code": "def add_numbers(a, b):\n"  # Required for replace/add
        },
        {
            "file_path": "main.py",
            "line_number": 73,
            "operation": "replace",
            "new_code": "def thread_task():\n"
        }
    ],
    "description": "Fixed 2 syntax errors"
}
```

**LineEditAction Operations:**
- `replace` - Modify existing line (line must exist)
- `add` - Insert new line before specified line number
- `delete` - Remove a line (new_code not used)

**Advantages:**
- No need to provide entire file content
- Multiple edits in one action
- More efficient - targeted changes only
- Clear indication of what changed

### Observation - CodeReviewerObservation
```python
{
    "repo_url": "https://github.com/...",
    "all_code": "# All code from repo",
    "code_file_path": "Repo/{id}/code_summary.txt",
    "execution_logs": "stdout/stderr from main.py",
    "errors": [
        {
            "error_type": "RuntimeError",
            "error_message": "...",
            "line_number": 42,
            "traceback": "..."
        }
    ],
    "code_metrics": {
        "unused_imports": ["os", "sys"],
        "syntax_errors": 0,
        "runtime_errors": 1,
        "warnings": [],
        "total_lines": 150,
        "total_files": 3
    },
    "done": false,
    "reward": 0.5,
    "step_count": 1,
    "episode_id": "uuid"
}
```

### Reward
The reward signal guides learning:
- **Positive**: Fixing errors, removing unused imports, succeeding
- **Negative**: Introducing new errors, syntax violations
- **Episodes end**: When all errors and warnings are fixed

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --repo-id my-org/code-reviewer --private
```

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for testing
- **API Documentation** at `/docs` - Full OpenAPI interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint

## Advanced Usage

### Connecting to an Existing Server

```python
from my_env import CodeReviewerEnv

# Connect to running server
env = CodeReviewerEnv(base_url="http://localhost:8000")

result = env.reset("https://github.com/user/repo")
```

### Local Testing Without Server

```python
from my_env.server.my_env_environment import CodeReviewerEnvironment

# Direct environment access
env = CodeReviewerEnvironment()
obs = env.reset("https://github.com/user/repo")

# Apply action
action = CodeReviewerAction(
    file_path="main.py",
    modified_code="...",
    description="Fix"
)
obs = env.step(action)
```

### Training an RL Agent

```python
from my_env import CodeReviewerEnv, CodeReviewerAction
import random

env = CodeReviewerEnv(base_url="http://localhost:8000")
obs = env.reset("https://github.com/target/repo")

total_reward = 0
for step in range(10):  # Max 10 improvement attempts
    # Agent analyzes observation and proposes fix
    # (Replace with your RL model)
    action = CodeReviewerAction(
        file_path="main.py",
        modified_code="improved code...",
        description=f"Improvement {step+1}"
    )
    
    result = env.step(action)
    total_reward += result.reward
    
    if result.done:
        print(f"Episode completed in {step+1} steps!")
        break

print(f"Total reward: {total_reward}")
```

## Repository Requirements

The Code Reviewer environment expects repositories to have:
- **main.py** in the root directory - This will be executed to test changes
- **Python files** - Source code to analyze and improve

The environment will:
1. Clone the repository
2. Collect all Python source files
3. Run main.py and capture any errors
4. Track metrics across improvements

## Testing

Run the test suite to verify the installation:

```bash
python test_environment.py
```

Try the example usage:

```bash
python example_usage.py
```

## Documentation

- **Full Guide**: See `CODE_REVIEWER_GUIDE.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **LineEditAction Guide**: See `LINE_EDIT_EXAMPLES.md` ⭐ NEW
- **Examples**: See `example_usage.py`
- **Tests**: Run `test_environment.py`

## Configuration

### max_concurrent_envs
Edit `server/app.py`:
```python
app = create_app(
    ...,
    max_concurrent_envs=4  # Adjust for system resources
)
```

### Execution Timeout
Edit `server/my_env_environment.py`:
```python
timeout=30  # Seconds before execution times out
```

### Repository Cleanup
The environment automatically keeps the 3 most recent repositories
in the `Repo/` folder.

## What Agents Learn

This environment helps train agents to:
- ✓ Identify Python syntax errors
- ✓ Fix runtime errors
- ✓ Detect and remove unused imports
- ✓ Improve code quality
- ✓ Test code changes iteratively
- ✓ Optimize based on reward signals
- ✓ Develop code review strategies

## Next Steps

1. **Build Docker Image**: `docker build -t code-reviewer-env:latest -f server/Dockerfile .`
2. **Run Tests**: `python test_environment.py`
3. **Start Server**: `uvicorn server.app:app --reload`
4. **Connect Agent**: Use `CodeReviewerEnv` client
5. **Train Model**: Implement your RL agent

## Support

For issues, examples, and documentation:
- See `CODE_REVIEWER_GUIDE.md` for complete API reference
- See `IMPLEMENTATION_SUMMARY.md` for architecture details
- Run `test_environment.py` to verify installation
- Check `example_usage.py` for usage patterns

# Use as normal
result = my_envenv.reset()
result = my_envenv.step(MyAction(message="Hello!"))
```

Note: When connecting to an existing server, `my_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from my_env import MyAction, MyEnv

# Connect with context manager (auto-connects and closes)
with MyEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(MyAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    MyEnvironment,  # Pass class, not instance
    MyAction,
    MyObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from my_env import MyAction, MyEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with MyEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(MyAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/my_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
my_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # MyEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── my_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
