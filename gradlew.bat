@echo off
setlocal EnableExtensions
set "NOX_APP_HOME=%~dp0"
set "NOX_GRADLE_COMMAND="
set "NOX_GRADLE_RESULT=%TEMP%\nox-gradle-path-%RANDOM%-%RANDOM%.txt"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%NOX_APP_HOME%scripts\ci\resolve-gradle.ps1" > "%NOX_GRADLE_RESULT%"
if errorlevel 1 goto :bootstrap_failed
if not exist "%NOX_GRADLE_RESULT%" goto :bootstrap_failed
set /p NOX_GRADLE_COMMAND=<"%NOX_GRADLE_RESULT%"
del "%NOX_GRADLE_RESULT%" >nul 2>&1

if not defined NOX_GRADLE_COMMAND goto :bootstrap_failed
call "%NOX_GRADLE_COMMAND%" %*
exit /b %ERRORLEVEL%

:bootstrap_failed
if exist "%NOX_GRADLE_RESULT%" del "%NOX_GRADLE_RESULT%" >nul 2>&1
echo NoX Gradle bootstrap failed. Verify PowerShell, Java 17+, and network access. 1>&2
exit /b 1
