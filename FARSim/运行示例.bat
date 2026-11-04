@echo off
chcp 65001 >nul
python "%~dp0run_rate.py" configs/smoke.json
pause
