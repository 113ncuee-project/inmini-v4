@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -X utf8 simplified_rapidchiplet\full_system_gui.py %*
pause
