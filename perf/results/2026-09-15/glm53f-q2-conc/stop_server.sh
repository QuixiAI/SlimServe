#!/bin/bash
# Stop the glm53f-q2-1 server and wait for port 8000 to be released.
#
# `pkill -f slimserve.cli` is NOT enough: the CLI spawns
# vllm.entrypoints.openai.api_server plus a VLLM::EngineCore child, and only
# the wrapper carries "slimserve.cli" in its argv. Killing the wrapper leaves
# the server holding port 8000, and the next boot dies with
# "OSError: [Errno 48] Address already in use" ~10 minutes later.
set -u
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null
pkill -f 'VLLM::EngineCore' 2>/dev/null
pkill -f 'slimserve.cli' 2>/dev/null
for i in $(seq 1 30); do
  if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port 8000 free after ~${i}s"; exit 0
  fi
  sleep 1
done
echo "PORT 8000 STILL HELD"; lsof -nP -iTCP:8000 -sTCP:LISTEN; exit 1
