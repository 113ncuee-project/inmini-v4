@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_inmin_full_overnight.ps1" -KeepAwake %*
set "INMIN_EXIT=%ERRORLEVEL%"
pause
exit /b %INMIN_EXIT%
