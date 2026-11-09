#!/usr/bin/env bash
# NoX self-bootstrapping Gradle launcher.
# Distribution version/checksum intentionally match .github/workflows/full-build.yml.
set -euo pipefail

readonly GRADLE_VERSION="9.6.0"
readonly GRADLE_SHA256="bbaeb2fef8710818cf0e261201dab964c572f92b942812df0c3620d62a529a01"
readonly GRADLE_URL="https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"

if ! command -v java >/dev/null 2>&1; then
    echo "NoX Gradle bootstrap requires a JDK. Install Java 17 or newer and retry." >&2
    exit 1
fi

if [[ -z "${GRADLE_USER_HOME:-}" ]]; then
    if [[ -z "${HOME:-}" ]]; then
        echo "Unable to resolve Gradle user home. Set GRADLE_USER_HOME and retry." >&2
        exit 1
    fi
    GRADLE_USER_HOME="${HOME}/.gradle"
fi

readonly CACHE_ROOT="${GRADLE_USER_HOME}/wrapper/dists/gradle-${GRADLE_VERSION}-bin/nox-bootstrap"
readonly LOCK_DIR="${CACHE_ROOT}/.install.lock"
readonly GRADLE_HOME="${CACHE_ROOT}/gradle-${GRADLE_VERSION}"
readonly GRADLE_BIN="${GRADLE_HOME}/bin/gradle"

mkdir -p "$CACHE_ROOT"

if [[ ! -x "$GRADLE_BIN" ]]; then
    acquired_lock=0
    for attempt in $(seq 1 300); do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            printf '%s\n' "$$" > "${LOCK_DIR}/pid"
            acquired_lock=1
            break
        fi

        if [[ -x "$GRADLE_BIN" ]]; then
            break
        fi

        if [[ -f "${LOCK_DIR}/pid" ]]; then
            holder_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
            if [[ "$holder_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$holder_pid" 2>/dev/null; then
                stale_lock="${LOCK_DIR}.stale.$$"
                if mv "$LOCK_DIR" "$stale_lock" 2>/dev/null; then
                    rm -rf "$stale_lock"
                fi
                continue
            fi
        fi
        sleep 1
    done

    if [[ ! -x "$GRADLE_BIN" && "$acquired_lock" -ne 1 ]]; then
        echo "Timed out waiting for another Gradle bootstrap process." >&2
        exit 1
    fi

    if [[ "$acquired_lock" -eq 1 ]]; then
        temp_root="$(mktemp -d "${CACHE_ROOT}/.bootstrap.XXXXXX")"
        cleanup() {
            rm -rf "$temp_root"
            rmdir "$LOCK_DIR" 2>/dev/null || true
        }
        trap cleanup EXIT INT TERM

        if [[ ! -x "$GRADLE_BIN" ]]; then
            archive="${temp_root}/gradle-${GRADLE_VERSION}-bin.zip"
            if command -v curl >/dev/null 2>&1; then
                curl --fail --show-error --location \
                    --retry 5 --retry-delay 2 --connect-timeout 20 \
                    --output "$archive" "$GRADLE_URL"
            elif command -v wget >/dev/null 2>&1; then
                wget --tries=5 --timeout=30 --output-document="$archive" "$GRADLE_URL"
            else
                echo "Gradle bootstrap requires curl or wget to download the pinned distribution." >&2
                exit 1
            fi

            if command -v sha256sum >/dev/null 2>&1; then
                actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
            elif command -v shasum >/dev/null 2>&1; then
                actual_sha256="$(shasum -a 256 "$archive" | awk '{print $1}')"
            else
                echo "Gradle bootstrap requires sha256sum or shasum for integrity verification." >&2
                exit 1
            fi

            actual_sha256="$(printf '%s' "$actual_sha256" | tr "A-F" "a-f")"
            if [[ "$actual_sha256" != "$GRADLE_SHA256" ]]; then
                echo "Gradle distribution SHA-256 mismatch." >&2
                echo "Expected: $GRADLE_SHA256" >&2
                echo "Actual:   $actual_sha256" >&2
                exit 1
            fi

            if ! command -v unzip >/dev/null 2>&1; then
                echo "Gradle bootstrap requires unzip to extract the verified distribution." >&2
                exit 1
            fi
            unzip -q "$archive" -d "$temp_root"
            candidate="${temp_root}/gradle-${GRADLE_VERSION}"
            if [[ ! -x "${candidate}/bin/gradle" ]]; then
                echo "Verified Gradle archive did not contain the expected launcher." >&2
                exit 1
            fi

            if ! mv "$candidate" "$GRADLE_HOME" 2>/dev/null; then
                # A concurrent process may have completed the install while this process
                # was downloading; accept that outcome only if the executable is present.
                if [[ ! -x "$GRADLE_BIN" ]]; then
                    echo "Unable to install the verified Gradle distribution." >&2
                    exit 1
                fi
            fi
        fi

        cleanup
        trap - EXIT INT TERM
    fi
fi

if [[ ! -x "$GRADLE_BIN" ]]; then
    echo "Gradle bootstrap finished without an executable at $GRADLE_BIN." >&2
    exit 1
fi

exec "$GRADLE_BIN" "$@"
