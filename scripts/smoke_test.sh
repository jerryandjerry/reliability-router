#!/usr/bin/env bash
set -euo pipefail

python -m reliability_router route --query "What is 2 + 2?"
python -m reliability_router answer \
  --query "Cite the API latency target." \
  --context examples/context.json
python -m reliability_router evaluate
