#!/usr/bin/env python3
"""Static contract tests for the self-bootstrapping NoX Gradle launchers."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
VERSION = "9.6.0"
SHA256 = "bbaeb2fef8710818cf0e261201dab964c572f92b942812df0c3620d62a529a01"


class GradleBootstrapContractTests(unittest.TestCase):
    def test_properties_pin_distribution_and_checksum(self):
        text = (ROOT / "gradle/wrapper/gradle-wrapper.properties").read_text()
        self.assertIn(f"gradle-{VERSION}-bin.zip", text)
        self.assertIn(f"distributionSha256Sum={SHA256}", text)
        self.assertIn("validateDistributionUrl=true", text)

    def test_unix_launcher_verifies_sha_and_preserves_arguments(self):
        text = (ROOT / "gradlew").read_text()
        self.assertIn(f'GRADLE_VERSION="{VERSION}"', text)
        self.assertIn(f'GRADLE_SHA256="{SHA256}"', text)
        self.assertIn('exec "$GRADLE_BIN" "$@"', text)
        self.assertIn("sha256sum", text)
        self.assertIn("shasum -a 256", text)

    def test_unix_launcher_remains_compatible_with_macos_bash_3(self):
        text = (ROOT / "gradlew").read_text()
        self.assertNotIn("${actual_sha256,,}", text)
        self.assertIn('tr "A-F" "a-f"', text)
        self.assertIn('if [[ -z "${GRADLE_USER_HOME:-}" ]]; then', text)

    def test_unix_launcher_uses_cached_gradle_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            gradle_user_home = temp / "gradle-home"
            cached_gradle = (
                gradle_user_home
                / "wrapper/dists/gradle-9.6.0-bin/nox-bootstrap"
                / "gradle-9.6.0/bin/gradle"
            )
            cached_gradle.parent.mkdir(parents=True)
            cached_gradle.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'fake-gradle\\n'\n"
                "printf '<%s>\\n' \"$@\"\n"
            )
            cached_gradle.chmod(0o755)

            bin_dir = temp / "bin"
            bin_dir.mkdir()
            fake_java = bin_dir / "java"
            fake_java.write_text("#!/usr/bin/env sh\\nexit 0\\n")
            fake_java.chmod(0o755)

            env = os.environ.copy()
            env["GRADLE_USER_HOME"] = str(gradle_user_home)
            env["HOME"] = directory
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

            result = subprocess.run(
                [str(ROOT / "gradlew"), "--version", "--no-daemon", "-Pflag=a b"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                result.stdout,
                "fake-gradle\n<--version>\n<--no-daemon>\n<-Pflag=a b>\n",
            )

    def test_windows_launcher_resolves_and_calls_gradle(self):
        text = (ROOT / "gradlew.bat").read_text()
        self.assertIn("resolve-gradle.ps1", text)
        self.assertIn('call "%NOX_GRADLE_COMMAND%" %*', text)

    def test_windows_resolver_uses_same_distribution_and_checksum(self):
        text = (ROOT / "scripts/ci/resolve-gradle.ps1").read_text()
        self.assertIn(f"$version = '{VERSION}'", text)
        self.assertIn(f"$expectedSha256 = '{SHA256}'", text)
        self.assertIn("Get-FileHash", text)
        self.assertIn("Expand-Archive", text)

    def test_windows_resolver_only_emits_the_gradle_path(self):
        text = (ROOT / "scripts/ci/resolve-gradle.ps1").read_text()
        self.assertIn(
            "Move-Item -LiteralPath $candidate -Destination $gradleHome -ErrorAction Stop | Out-Null",
            text,
        )
        self.assertIn("Write-Output $gradleCommand", text)

    def test_unix_and_windows_launchers_use_the_same_pinned_version(self):
        unix = (ROOT / "gradlew").read_text()
        windows = (ROOT / "scripts/ci/resolve-gradle.ps1").read_text()
        properties = (ROOT / "gradle/wrapper/gradle-wrapper.properties").read_text()
        self.assertIn(VERSION, unix)
        self.assertIn(VERSION, windows)
        self.assertIn(VERSION, properties)


if __name__ == "__main__":
    unittest.main(verbosity=2)
