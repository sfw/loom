#!/usr/bin/env bash
# Run the full Loom test suite.
# Usage: ./scripts/test.sh [pytest-args...]
#
# Examples:
#   ./scripts/test.sh                    # Run all tests
#   ./scripts/test.sh -v                 # Verbose output
#   ./scripts/test.sh -k test_config     # Run only config tests
#   ./scripts/test.sh --cov=loom         # With coverage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Loom Test Suite ==="
echo "Python: $(python3 --version)"
echo ""

# Run ruff lint check
echo "--- Lint Check ---"
if command -v ruff &>/dev/null || uv run ruff --version &>/dev/null 2>&1; then
    uv run ruff check src/ tests/ && echo "Lint: PASSED" || echo "Lint: FAILED"
else
    echo "Lint: SKIPPED (ruff not installed)"
fi
echo ""

# Run tests
echo "--- Tests ---"
uv run pytest "$@"
