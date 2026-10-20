#!/bin/bash
#
# Start the Claude API proxy.
#
# This proxy translates llama.cpp /completion format to Claude /v1/messages format,
# so any client that speaks to llama-server can use Claude as a backend.
#
# Usage:
#   ./start-claude-proxy.sh [extra claude-proxy.py flags...]
#
# Requirements:
#   - ANTHROPIC_API_KEY environment variable must be set
#   - Python 3.x
#
# Configuration:
#   Point your client's llama.cpp server URL at http://localhost:8081
#   (or whatever CLAUDE_PROXY_PORT you choose).
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/claude-proxy.py"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY environment variable not set"
    echo ""
    echo "To set it:"
    echo "  export ANTHROPIC_API_KEY=..."
    echo ""
    echo "Or add to your ~/.zshrc or ~/.bashrc"
    exit 1
fi

# Default listen port
PORT=${CLAUDE_PROXY_PORT:-8081}

# Default model - can override with CLAUDE_MODEL env var
MODEL=${CLAUDE_MODEL:-"claude-sonnet-4-20250514"}

echo "Starting Claude API proxy..."
echo "  Port: $PORT"
echo "  Model: $MODEL"
echo "  API Key: set"
echo ""
echo "Point your client's llama.cpp server URL at:"
echo "  http://localhost:$PORT"
echo ""

python3 "$PROXY_SCRIPT" --port "$PORT" --model "$MODEL" "$@"
