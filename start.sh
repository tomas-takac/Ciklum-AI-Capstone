#!/bin/bash
# Make sure we are in the script's directory
cd "$(dirname "$0")"

echo "Starting MyLittleRag System..."
docker compose up -d --build

echo "Attaching to Chat (Press Ctrl+P then Ctrl+Q to detach without stopping)"
docker attach rag-app
