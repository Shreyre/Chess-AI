@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m chess_ai uci --checkpoint "%~dp0runs\main\model.pt" %*
