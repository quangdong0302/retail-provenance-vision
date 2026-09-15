@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
python "%~dp0run_all.py" %*
exit /b %ERRORLEVEL%
