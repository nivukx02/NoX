#!/usr/bin/env bash
set -euo pipefail

BASE_REF="${1:-}"
HEAD_REF="${2:-HEAD}"

if [[ -z "${BASE_REF}" ]]; then
  echo "Usage: verify_no_deleted_files.sh <base-ref> [head-ref]" >&2
  exit 2
fi

deleted="$(git diff --name-status --diff-filter=D "${BASE_REF}" "${HEAD_REF}")"

if [[ -n "${deleted}" ]]; then
  echo "ERROR: tracked files were deleted in this upgrade range:"
  printf '%s\n' "${deleted}"
  exit 1
fi

echo "No tracked files deleted between ${BASE_REF} and ${HEAD_REF}."
