#!/usr/bin/env bash
set -euo pipefail

output_path="${1:-artifacts/environment.json}"
adaptive-vlm environment --workdir "$(pwd)" --output "${output_path}"
