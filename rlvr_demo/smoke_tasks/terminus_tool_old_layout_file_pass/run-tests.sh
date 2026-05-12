#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest /tests/test_answer.py -q -rA
