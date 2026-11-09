#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="${SCRIPT_DIR}/verify_no_source_line_deletions.sh"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "${FIXTURE}"' EXIT

git -C "${FIXTURE}" init -q
git -C "${FIXTURE}" config user.name "NoX CI test"
git -C "${FIXTURE}" config user.email "nox-ci-test@example.invalid"
printf 'alpha\nbeta\ngamma\n' > "${FIXTURE}/fixture.kt"
git -C "${FIXTURE}" add fixture.kt
git -C "${FIXTURE}" commit -qm 'baseline fixture'
BASE="$(git -C "${FIXTURE}" rev-parse HEAD)"

printf 'alpha\nreplacement\ngamma\nadded\n' > "${FIXTURE}/fixture.kt"
git -C "${FIXTURE}" add fixture.kt
git -C "${FIXTURE}" commit -qm 'same-size replacement plus additive improvement'
HEAD="$(git -C "${FIXTURE}" rev-parse HEAD)"
if ! (cd "${FIXTURE}" && bash "${POLICY}" "${BASE}" "${HEAD}"); then
  echo 'Expected a behavior-preserving source replacement to pass.' >&2
  exit 1
fi

BASE="${HEAD}"
printf 'alpha\n' > "${FIXTURE}/fixture.kt"
git -C "${FIXTURE}" add fixture.kt
git -C "${FIXTURE}" commit -qm 'deliberate net source loss'
HEAD="$(git -C "${FIXTURE}" rev-parse HEAD)"
if (cd "${FIXTURE}" && bash "${POLICY}" "${BASE}" "${HEAD}" >/dev/null 2>&1); then
  echo 'Expected a net source-line loss to fail the policy.' >&2
  exit 1
fi

echo 'Source preservation policy regression tests passed.'
