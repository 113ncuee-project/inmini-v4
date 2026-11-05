@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_inmin_full_parallel.ps1" %*
if errorlevel 1 pause
