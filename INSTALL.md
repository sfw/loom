# Installation Guide

Complete setup instructions for Loom, from prerequisites through first task execution.

## Prerequisites

### Python 3.11+

Loom requires Python 3.11 or later.

```bash
# Check your version
python3 --version

# macOS (via Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt update && sudo apt install python3.12 python3.12-venv

# Fedora
sudo dnf install python3.12
```

### A Local Model Backend

Loom needs at least one local LLM backend running. Choose one:

#### Option A: Ollama (Recommended for beginners)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen3:14b      # Primary model (planning, execution)
ollama pull qwen3:8b        # Utility model (extraction, verification)

# Verify it's running
curl http://localhost:11434/api/tags
```

Ollama runs on port 11434 by default.

#### Option B: LM Studio

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load a model (e.g., Qwen 2.5 14B, Mistral 7B, or any GGUF model)
3. Start the local server (default port: 1234)
4. Verify: `curl http://localhost:1234/v1/models`

#### Option C: vLLM / Other OpenAI-Compatible Server

Any server that exposes the `/v1/chat/completions` endpoint works:

```bash
# Example with vLLM
pip install vllm
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000
```

---

## Install Loom

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-org/loom.git
cd loom

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
loom --version
```

### Using uv (Faster)

```bash
git clone https://github.com/your-org/loom.git
cd loom

# Install with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

loom --version
```

---

## Configuration

### Create the Config File

```bash
# Option 1: Copy the example config to your home directory
mkdir -p ~/.loom
cp loom.toml ~/.loom/loom.toml

# Option 2: Keep it in your project directory (Loom checks ./loom.toml first)
```

### Configure Your Models

Edit `~/.loom/loom.toml` (or `./loom.toml`) to match your backend:

#### For Ollama

```toml
[server]
host = "127.0.0.1"
port = 9000

[models.primary]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:14b"
max_tokens = 4096
temperature = 0.1
roles = ["planner", "executor"]

[models.utility]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
max_tokens = 2048
temperature = 0.0
roles = ["extractor", "verifier"]
```

#### For LM Studio

```toml
[models.primary]
provider = "openai_compatible"
base_url = "http://localhost:1234/v1"
model = "qwen2.5-14b-instruct"
max_tokens = 4096
temperature = 0.1
roles = ["planner", "executor"]
```

#### Single Model Setup (Simplest)

If you only have one model, assign it all roles:

```toml
[models.primary]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:14b"
max_tokens = 4096
temperature = 0.1
roles = ["planner", "executor", "extractor", "verifier"]
```

### Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `server` | `host` | `127.0.0.1` | Bind address |
| `server` | `port` | `9000` | Server port |
| `models.<name>` | `provider` | -- | `ollama` or `openai_compatible` |
| `models.<name>` | `base_url` | -- | Model API URL |
| `models.<name>` | `model` | -- | Model identifier |
| `models.<name>` | `max_tokens` | `4096` | Max response tokens |
| `models.<name>` | `temperature` | `0.1` | Sampling temperature |
| `models.<name>` | `roles` | `["executor"]` | Assigned roles |
| `workspace` | `default_path` | `~/projects` | Default workspace |
| `workspace` | `scratch_dir` | `~/.loom/scratch` | Temp state storage |
| `execution` | `max_subtask_retries` | `3` | Retries per subtask |
| `execution` | `max_loop_iterations` | `50` | Max tool-call loops per subtask |
| `memory` | `database_path` | `~/.loom/loom.db` | SQLite database location |
| `logging` | `level` | `INFO` | Log verbosity |
| `logging` | `event_log_path` | `~/.loom/logs` | Event log directory |

---

## Verify Installation

### 1. Check the CLI

```bash
loom --version
# loom, version 0.1.0

loom models
# Should list your configured models
```

### 2. Check Model Connectivity

```bash
# For Ollama
curl http://localhost:11434/api/tags

# For OpenAI-compatible
curl http://localhost:1234/v1/models
```

### 3. Run the Test Suite

```bash
pytest
# Should show 262 passed
```

### 4. Start the Server

```bash
loom serve
# Starting Loom server on 127.0.0.1:9000
```

### 5. Test the Health Endpoint

```bash
# In another terminal
curl http://localhost:9000/health
# {"status":"ok","version":"0.1.0"}
```

---

## First Task

With the server running:

```bash
# Create a workspace
mkdir -p /tmp/loom-demo

# Submit a task
curl -X POST http://localhost:9000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Create a Python script that prints the Fibonacci sequence up to 100",
    "workspace": "/tmp/loom-demo"
  }'

# Note the task_id from the response, then check status:
curl http://localhost:9000/tasks/<task_id>

# Or stream events in real-time:
curl -N http://localhost:9000/tasks/<task_id>/stream
```

Or use the CLI directly:

```bash
loom run "Create a Python script that prints Fibonacci numbers up to 100" \
  --workspace /tmp/loom-demo
```

---

## Troubleshooting

### "No models configured"

Your `loom.toml` isn't being found. Either:
- Place it in the current directory, or
- Place it at `~/.loom/loom.toml`, or
- Pass it explicitly: `loom --config /path/to/loom.toml serve`

### Connection refused to model backend

Make sure your model server is running:
```bash
# Ollama
systemctl status ollama   # or just: ollama serve

# LM Studio
# Open the app and click "Start Server"
```

### "Task failed" immediately

Check the task details for error info:
```bash
curl http://localhost:9000/tasks/<task_id>
```

Common causes:
- Model returned empty response (model may be too small for the task)
- Workspace path doesn't exist (create it first)
- Model server went down mid-task

### Port already in use

Change the port in your config or use the CLI override:
```bash
loom serve --port 9001
```

---

## Uninstall

```bash
pip uninstall loom
rm -rf ~/.loom          # Remove config and data
```
