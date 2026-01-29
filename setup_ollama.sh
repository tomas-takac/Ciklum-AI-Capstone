#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to wake up
sleep 5

# Pull the model
echo "--- Pulling llama3 model ---"
ollama pull llama3

# Keep the container running
wait
