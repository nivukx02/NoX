#!/usr/bin/env bash
set -euo pipefail

BASE_REF="${1:-}"
HEAD_REF="${2:-HEAD}"

if [[ -z "${BASE_REF}" ]]; then
  echo "Usage: verify_no_source_line_deletions.sh <base-ref> [head-ref]" >&2
  exit 2
fi

patterns=(
  '*.kt' '*.kts' '*.java' '*.cpp' '*.cc' '*.cxx'
  '*.h' '*.hpp' '*.hh' '*.c' '*.m' '*.mm'
  '*.gradle' '*.gradle.kts' '*.cmake' 'CMakeLists.txt'
  '*.xml' '*.glsl' '*.md' '*.py' '*.sh' '*.yml' '*.yaml'
  '*.json' '*.toml' '*.properties' '*.pro'
)

set +e
changes="$(git diff --no-ext-diff --unified=0 "${BASE_REF}" "${HEAD_REF}" -- "${patterns[@]}" |
  awk '
    /^diff --git / {
      path = $0
      sub(/^diff --git a\//, "", path)
      split(path, parts, " b/")
      current = parts[2]
      next
    }
    /^--- / || /^\+\+\+ / { next }
    /^-/ {
      line = substr($0, 2)
      if (line !~ /^[[:space:]]*$/) deleted[current]++
      next
    }
    /^\+/ {
      line = substr($0, 2)
      if (line !~ /^[[:space:]]*$/) added[current]++
      next
    }
    END {
      for (path in deleted) {
        if (deleted[path] > added[path]) {
          printf "%s: %d non-empty lines removed, %d non-empty lines added\n",
            path, deleted[path], added[path]
        }
      }
    }
  ')"
status=$?
set -e

if (( status != 0 )); then
  echo "ERROR: unable to evaluate the source-line preservation policy." >&2
  exit "${status}"
fi

if [[ -n "${changes}" ]]; then
  echo "ERROR: source/test/documentation files have a net loss of non-empty lines:"
  printf '%s\n' "${changes}"
  echo "Restore missing implementation or provide at least equal source-level replacement/addition within each changed file."
  exit 1
fi

echo "No changed source/test/documentation file has a net loss of non-empty lines between ${BASE_REF} and ${HEAD_REF}."
