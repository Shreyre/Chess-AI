@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m chess_ai.screen_player %*
