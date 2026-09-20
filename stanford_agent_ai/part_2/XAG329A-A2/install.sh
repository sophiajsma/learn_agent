#!/bin/bash
# Install project dependencies into /autograder/source/.venv for Gradescope.
# Invoked by gradescope/default/setup.sh as: source install.sh -r

set -euo pipefail

# Accept unused legacy flags (e.g. -r) for compatibility with setup.sh.
while getopts "r" opt; do
  case "$opt" in
    r) ;;
    *) ;;
  esac
done

# When sourced from setup.sh, cwd is already /autograder/source.
if [ -f "./pyproject.toml" ]; then
  SCRIPT_DIR="$(pwd)"
elif [ -f "${BASH_SOURCE[0]:-}/../pyproject.toml" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR="/autograder/source"
fi
cd "$SCRIPT_DIR"

export PATH="${HOME}/.local/bin:${PATH}"
if [ -f "${HOME}/.local/bin/env" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/.local/bin/env"
fi

if ! command -v uv >/dev/null 2>&1; then
  wget -qO- https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "${HOME}/.local/bin/env"
fi

# Create / sync the virtualenv from the locked project deps.
if [ -f uv.lock ]; then
  uv sync --frozen
else
  uv sync
fi

# Ensure the venv lives where run_autograder expects it.
if [ ! -d .venv ]; then
  echo "install.sh: expected .venv after uv sync" >&2
  return 1 2>/dev/null || exit 1
fi

echo "install.sh: environment ready at ${SCRIPT_DIR}/.venv"
