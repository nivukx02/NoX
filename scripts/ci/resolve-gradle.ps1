$ErrorActionPreference = 'Stop'

$version = '9.6.0'
$expectedSha256 = 'bbaeb2fef8710818cf0e261201dab964c572f92b942812df0c3620d62a529a01'
$url = "https://services.gradle.org/distributions/gradle-$version-bin.zip"

$gradleUserHome = $env:GRADLE_USER_HOME
if ([string]::IsNullOrWhiteSpace($gradleUserHome)) {
    if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        throw 'Unable to resolve Gradle user home. Set GRADLE_USER_HOME and retry.'
    }
    $gradleUserHome = Join-Path $env:USERPROFILE '.gradle'
}

$cacheRoot = Join-Path $gradleUserHome "wrapper\dists\gradle-$version-bin\nox-bootstrap"
$gradleHome = Join-Path $cacheRoot "gradle-$version"
$gradleCommand = Join-Path $gradleHome 'bin\gradle.bat'
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $gradleCommand -PathType Leaf)) {
    $tempRoot = Join-Path $cacheRoot ".bootstrap-$PID"
    $archive = Join-Path $tempRoot "gradle-$version-bin.zip"
    $candidate = Join-Path $tempRoot "gradle-$version"

    try {
        New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
        Invoke-WebRequest -Uri $url -OutFile $archive -MaximumRedirection 10

        $actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $expectedSha256) {
            throw "Gradle distribution SHA-256 mismatch. Expected $expectedSha256, got $actualSha256"
        }

        Expand-Archive -LiteralPath $archive -DestinationPath $tempRoot -Force
        if (-not (Test-Path -LiteralPath (Join-Path $candidate 'bin\gradle.bat') -PathType Leaf)) {
            throw 'Verified Gradle archive did not contain the expected launcher.'
        }

        if (-not (Test-Path -LiteralPath $gradleHome -PathType Container)) {
            try {
                Move-Item -LiteralPath $candidate -Destination $gradleHome -ErrorAction Stop | Out-Null
            } catch {
                if (-not (Test-Path -LiteralPath $gradleCommand -PathType Leaf)) {
                    throw
                }
            }
        }
    } finally {
        if (Test-Path -LiteralPath $tempRoot -PathType Container) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $gradleCommand -PathType Leaf)) {
    throw "Gradle bootstrap finished without an executable at $gradleCommand"
}

Write-Output $gradleCommand
